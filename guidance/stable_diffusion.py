from . import BaseGuidance
import torch
import torch.nn.functional as F
import omegaconf

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionPipeline,
    PNDMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)

from utils.typing import *
from utils.ops import perpendicular_component
from utils.misc import C
from rich.console import Console

console = Console()


class StableDiffusionGuidance(BaseGuidance):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.device = self.cfg.device
        self.pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
        self.guidance_scale = cfg.guidance_scale
        
        # 使用下划线前缀避免与nn.Module属性冲突
        self._max_steps = cfg.max_steps
        self._weights_dtype = torch.float16 if cfg.half_precision_weights else torch.float32
        
        # 噪声混合参数设置
        self.mix_ratio = cfg.noise.mix_ratio if hasattr(cfg.noise, "mix_ratio") else 0.5
        self.mix_method = cfg.noise.mix_method if hasattr(cfg.noise, "mix_method") else "weighted"
        self.alternate_step = 0  # 用于alternate模式的步骤计数
        
        # 初始化step变量
        self.step = 0
        self.grad_clip_val = None  # 初始化grad_clip_val

        # Load scheduler, tokenizer and models
        self.scheduler = DDIMScheduler.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self._weights_dtype,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            torch_dtype=self._weights_dtype,
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=self._weights_dtype,
        )

        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="unet",
            torch_dtype=self._weights_dtype,
        )

        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=self._weights_dtype,
        )

        # 设置默认的num_inference_steps
        num_inference_steps = getattr(cfg, "num_inference_steps", 50)
        self.scheduler.set_timesteps(num_inference_steps)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        self.unet.to(self.device)
        self.vae.to(self.device)
        self.text_encoder.to(self.device)

        self.set_min_max_steps()

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self):
        # 获取列表或单个值
        max_step_percent_cfg = self.cfg.max_step_percent
        
        # 如果max_step_percent是一个列表，解析它
        if isinstance(max_step_percent_cfg, (list, omegaconf.ListConfig)):
            if len(max_step_percent_cfg) == 4:  # [步数阈值, 开始值, 结束值, 步数阈值]
                threshold_start, value_start, value_end, threshold_end = max_step_percent_cfg
                
                # 根据当前步数计算插值值
                if self.step < threshold_start:
                    current_max_percent = value_start
                elif self.step > threshold_end:
                    current_max_percent = value_end
                else:
                    # 线性插值
                    ratio = (self.step - threshold_start) / (threshold_end - threshold_start)
                    current_max_percent = value_start + ratio * (value_end - value_start)
            else:
                # 如果格式不正确，使用第一个值或默认值
                current_max_percent = max_step_percent_cfg[0] if len(max_step_percent_cfg) > 0 else 0.98
        else:
            # 如果是单个值直接使用
            current_max_percent = max_step_percent_cfg
        
        # 设置最小和最大步骤
        self.min_step = int(self.scheduler.config.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.scheduler.config.num_train_timesteps * current_max_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents,
        t,
        encoder_hidden_states,
    ):
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self._weights_dtype),
            t.to(self._weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self._weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(self, imgs):
        input_dtype = imgs.dtype
        # 确保所有参数使用相同的数据类型
        
        # 保存原始VAE参数类型
        vae_params_dtype = {}
        for name, param in self.vae.named_parameters():
            vae_params_dtype[name] = param.data.dtype
            param.data = param.data.to(self._weights_dtype)
        
        # 使用统一的数据类型进行编码
        # 将图像归一化到[-1, 1]范围
        imgs = imgs * 2.0 - 1.0
        # 计算posterior
        posterior = self.vae.encode(imgs.to(self._weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        # 恢复原始参数类型
        for name, param in self.vae.named_parameters():
            if name in vae_params_dtype:
                param.data = param.data.to(vae_params_dtype[name])
        
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents,
        latent_height: int = 64,
        latent_width: int = 64,
    ):
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self._weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_embedding,
        elevation,
        azimuth,
        camera_distance,
        noise: Float[Tensor, "B 4 64 64"] = None,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # 如果没有提供噪声，则生成随机噪声
        if noise is None:
            noise = torch.randn_like(latents)
            
        # 获取两个prompt的embeddings
        text_emb1, text_emb2, uncond_text_emb = prompt_embedding.get_text_embedding(
            elevation, azimuth, camera_distance, self.cfg.use_view_dependent_prompt
        )
        
        B = latents.shape[0]
        latent_model_input = torch.cat([latents] * 2)
        
        # 计算第一个prompt的noise prediction
        noise_pred_1 = self.forward_unet(
            latent_model_input,
            torch.cat([t] * 2),
            encoder_hidden_states=torch.cat([text_emb1, uncond_text_emb])
        )
        noise_pred_text_1, noise_pred_uncond_1 = noise_pred_1.chunk(2)
        
        # 计算第二个prompt的noise prediction
        noise_pred_2 = self.forward_unet(
            latent_model_input,
            torch.cat([t] * 2),
            encoder_hidden_states=torch.cat([text_emb2, uncond_text_emb])
        )
        noise_pred_text_2, noise_pred_uncond_2 = noise_pred_2.chunk(2)

        # 如果使用adaptive混合方法，动态调整mix_ratio
        if self.mix_method == "adaptive":
            # 可以基于当前时间步t动态调整mix_ratio
            # 较早的步骤更倾向于prompt1，较晚的步骤更倾向于prompt2
            step_percentage = t.float() / self.num_train_timesteps
            self.current_mix_ratio = self.mix_ratio * (1 - step_percentage.mean())
        else:
            self.current_mix_ratio = self.mix_ratio
        
        # 根据不同的混合策略进行噪声混合
        if self.mix_method == "weighted":
            # 加权混合策略
            noise_pred_text = self.current_mix_ratio * noise_pred_text_1 + (1 - self.current_mix_ratio) * noise_pred_text_2
            noise_pred_uncond = self.current_mix_ratio * noise_pred_uncond_1 + (1 - self.current_mix_ratio) * noise_pred_uncond_2
        
        elif self.mix_method == "alternate":
            # 交替混合策略 - 每个步骤使用不同的噪声预测
            if self.alternate_step % 2 == 0:
                noise_pred_text = noise_pred_text_1
                noise_pred_uncond = noise_pred_uncond_1
            else:
                noise_pred_text = noise_pred_text_2
                noise_pred_uncond = noise_pred_uncond_2
            self.alternate_step += 1
        
        elif self.mix_method == "adaptive":
            # 自适应混合策略 - 已在上面设置了current_mix_ratio
            noise_pred_text = self.current_mix_ratio * noise_pred_text_1 + (1 - self.current_mix_ratio) * noise_pred_text_2
            noise_pred_uncond = self.current_mix_ratio * noise_pred_uncond_1 + (1 - self.current_mix_ratio) * noise_pred_uncond_2
        
        elif self.mix_method == "factorized":
            # 因子化混合 - 在不同的特征通道上偏向不同的prompt
            channels = noise_pred_text_1.shape[1]
            split_idx = int(channels * self.mix_ratio)
            noise_pred_text = torch.cat([
                noise_pred_text_1[:, :split_idx], 
                noise_pred_text_2[:, split_idx:]
            ], dim=1)
            noise_pred_uncond = torch.cat([
                noise_pred_uncond_1[:, :split_idx], 
                noise_pred_uncond_2[:, split_idx:]
            ], dim=1)
        
        else:
            # 默认使用weighted
            noise_pred_text = self.mix_ratio * noise_pred_text_1 + (1 - self.mix_ratio) * noise_pred_text_2
            noise_pred_uncond = self.mix_ratio * noise_pred_uncond_1 + (1 - self.mix_ratio) * noise_pred_uncond_2

        # 计算最终的noise prediction
        if use_perp_neg:
            if neg_guidance_weights is None:
                # 获取perpendicular negative guidance的embeddings和权重
                text_embeddings_perp1, text_embeddings_perp2, neg_guidance_weights = prompt_embedding.get_text_embeddings_perp_neg(
                    elevation, azimuth, camera_distance, self.cfg.use_view_dependent_prompt
                )
                
                # 计算两个prompt的perpendicular components
                noise_pred_perp1 = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * (2 + 2)),  # 2 for pos/uncond, 2 for neg directions
                    encoder_hidden_states=text_embeddings_perp1,
                )
                
                noise_pred_perp2 = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * (2 + 2)),
                    encoder_hidden_states=text_embeddings_perp2,
                )
                
                # 混合两个prompt的perpendicular components
                if self.mix_method == "weighted":
                    e_pos = self.current_mix_ratio * noise_pred_perp1[:B] + (1 - self.current_mix_ratio) * noise_pred_perp2[:B]
                    e_neg = self.current_mix_ratio * noise_pred_perp1[2*B:] + (1 - self.current_mix_ratio) * noise_pred_perp2[2*B:]
                elif self.mix_method == "alternate":
                    if (self.alternate_step - 1) % 2 == 0:  # 使用上一步的alternate_step
                        e_pos = noise_pred_perp1[:B]
                        e_neg = noise_pred_perp1[2*B:]
                    else:
                        e_pos = noise_pred_perp2[:B]
                        e_neg = noise_pred_perp2[2*B:]
                elif self.mix_method == "adaptive":
                    adaptive_ratio = self.current_mix_ratio * (1 - t.float() / self.scheduler.config.num_train_timesteps)
                    e_pos = adaptive_ratio * noise_pred_perp1[:B] + (1 - adaptive_ratio) * noise_pred_perp2[:B]
                    e_neg = adaptive_ratio * noise_pred_perp1[2*B:] + (1 - adaptive_ratio) * noise_pred_perp2[2*B:]
                elif self.mix_method == "factorized":
                    e_pos = (noise_pred_perp1[:B] + noise_pred_perp2[:B]) / 2
                    e_neg = (noise_pred_perp1[2*B:] + noise_pred_perp2[2*B:]) / 2
                else:
                    e_pos = self.current_mix_ratio * noise_pred_perp1[:B] + (1 - self.current_mix_ratio) * noise_pred_perp2[:B]
                    e_neg = self.current_mix_ratio * noise_pred_perp1[2*B:] + (1 - self.current_mix_ratio) * noise_pred_perp2[2*B:]
                
                # 计算accumulated gradient
                accum_grad = torch.zeros_like(e_pos)
                for i in range(neg_guidance_weights.shape[1]):
                    accum_grad += neg_guidance_weights[:, i].view(-1, 1, 1, 1) * e_neg[i::2]
            
            noise_pred = noise_pred_uncond + self.guidance_scale * (e_pos + accum_grad)
        else:
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 计算梯度
        if self.cfg.weighting_strategy == "sds":
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.cfg.weighting_strategy}")

        grad = w * (noise_pred - noise)

        guidance_eval_utils = {
            "use_perp_neg": use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": torch.cat([text_emb1, text_emb2, uncond_text_emb]),
            "t_orig": t,
            "latents_noisy": latents,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils

    def forward(
        self,
        rgb,
        prompt_embedding,
        elevation,
        azimuth,
        camera_distance,
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        bs = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [bs],
            dtype=torch.long,
            device=self.device,
        )

        grad, guidance_eval_utils = self.compute_grad_sds(
            latents, t, prompt_embedding, elevation, azimuth, camera_distance
        )

        grad = torch.nan_to_num(grad)
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / bs
        loss_sds_each = 0.5 * F.mse_loss(latents, target, reduction="none").sum(
            dim=[1, 2, 3]
        )

        guidance_out = {
            "loss_sds": loss_sds,
            "loss_sds_each": loss_sds_each,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distance
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    def guidance_eval(
        self,
        use_perp_neg,
        neg_guidance_weights,
        text_embeddings,
        t_orig,
        latents_noisy,
        noise_pred,
    ):
        # get the predicted x0
        pred_x0 = self.scheduler.step(
            noise_pred, t_orig[0], latents_noisy, eta=0
        ).pred_original_sample
        # decode it to image
        pred_x0 = self.decode_latents(pred_x0)
        # move to cpu
        pred_x0 = pred_x0.detach().cpu()
        # scale it to 0-1
        pred_x0 = torch.clamp((pred_x0 + 1.0) / 2.0, min=0.0, max=1.0)
        # convert to image
        pred_x0 = pred_x0.permute(0, 2, 3, 1).numpy()
        # convert to uint8
        pred_x0 = (pred_x0 * 255).astype(np.uint8)

        return {
            "noise_levels": t_orig.detach().cpu().numpy() / self.scheduler.config.num_train_timesteps,
            "images": pred_x0,
        }

    def update(self, step):
        """更新guidance的状态"""
        self.step = step
        self.set_min_max_steps()
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, step, self._max_steps)

    def log(self, writer, step):
        pass

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def weights_dtype(self):
        return self._weights_dtype

    @property
    def num_train_timesteps(self):
        return self.scheduler.config.num_train_timesteps
