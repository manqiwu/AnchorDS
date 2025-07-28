import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from controlnet_aux import CannyDetector, NormalBaeDetector
from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-controlnet-guidance")
class ControlNetGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        # pretrained_model_name_or_path: str = "SG161222/Realistic_Vision_V2.0"
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        ddim_scheduler_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        control_type: str = "normal"  # normal/canny

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        fixed_size: int = -1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        diffusion_steps: int = 20
        view_dependent_prompting: bool = True
        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

        use_sds: bool = False

        max_steps:int = -1
        use_sdse: bool = True
        pure_seeking: bool = False
        tgt_with_cond: bool = True
        guess_mode: bool = False
        use_non_incr_timesteps: bool = False
        guidance_design: int = 0
        weighting_strategy: str = "sds"

        # Canny threshold
        canny_lower_bound: int = 50
        canny_upper_bound: int = 100
        
        stage_two_start_step: int = -1
        enough_small_t_threshold: int = -1 # if set, will use this threshold to determine if the t is small enough. -1 means not using this threshold

        filter_img_cond: Optional[str] = None # "mse"
        filter_img_cond_weight: float = 0.4 # 0.03

        finetune: bool = False
        finetune_layer_name: str = "" # "controlnet_mid_block" or "controlnet_down_blocks.10" or "controlnet_cond_embedding.conv_out"

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading ControlNet ...")

        controlnet_name_or_path: str
        if self.cfg.control_type == "normal":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_normalbae"
        elif self.cfg.control_type == "canny":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_canny"
        elif self.cfg.control_type == 'inpaint':
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_inpaint"
        elif self.cfg.control_type == 'tile':
            controlnet_name_or_path = "lllyasviel/control_v11f1e_sd15_tile"
        elif self.cfg.control_type == 'normal_sd21':
            controlnet_name_or_path = 'thibaud/controlnet-sd21-normalbae-diffusers'
            

        # check: use_sds and use_sdse: 1. cannot be both True at the same time 2. cannot be both False at the same time
        assert not (self.cfg.use_sds and self.cfg.use_sdse), "use_sds and use_sdse cannot be both True at the same time"
        assert self.cfg.use_sds or self.cfg.use_sdse, "use_sds and use_sdse cannot be both False at the same time"

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        controlnet = ControlNetModel.from_pretrained(
            controlnet_name_or_path,
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, controlnet=controlnet, **pipe_kwargs
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        # Set the controlnet mode based on finetuning flag.
        if self.cfg.finetune:
            # For adapter fine-tuning we want the controlnet modules to be in training mode
            # even though most of its parameters will remain frozen.
            self.controlnet = self.pipe.controlnet.train()
        else:
            self.controlnet = self.pipe.controlnet.eval()

        if self.cfg.control_type == "normal" or self.cfg.control_type == 'normal_sd21':
            self.preprocessor = NormalBaeDetector.from_pretrained(
                "lllyasviel/Annotators"
            )
            self.preprocessor.model.to(self.device)
        elif self.cfg.control_type == "canny":
            self.preprocessor = CannyDetector()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.controlnet.parameters():
            p.requires_grad_(False)

        # Only fine-tune modules in the specified ControlNet layer to reduce memory usage.
        if self.cfg.finetune:
            valid_layers = ["controlnet_mid_block", "controlnet_down_blocks.10", "controlnet_down_blocks.11", "controlnet_cond_embedding.conv_out"]
            assert self.cfg.finetune_layer_name in valid_layers, f"finetune_layer_name must be one of {valid_layers}, but got {self.cfg.finetune_layer_name}"
            
            unfrozen_params = 0
            for name, module in self.controlnet.named_modules():
                if self.cfg.finetune_layer_name in name:
                    module.train()
                    for param_name, param in module.named_parameters():
                        param.requires_grad_(True)
                        param.data = param.data.float()  # Ensure master copy is FP32
                        unfrozen_params += param.numel()
                        print(f"{name}.{param_name} dtype: {param.dtype}, shape: {param.shape}")
                    threestudio.info(f"Unfreezing ControlNet parameters in module: {name}")
            
            threestudio.info(f"Total unfrozen ControlNet parameters: {unfrozen_params:,}")
        else:
            threestudio.info("Finetuning disabled; all ControlNet parameters remain frozen.")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None
        self.phase_id = 1

        self.global_step: int = 0
        if self.cfg.use_non_incr_timesteps:
            assert self.cfg.max_steps > 0, "max_steps should be set if use_non_incr_timesteps is True"
            self.t_choice = self._init_non_incr_timesteps(self.cfg.max_steps, 0.02, 0.80) # !Fixme: 0.02, 0.80 hardcoded
            threestudio.info("Using non-increasing timesteps.")

        threestudio.info(f"Loaded ControlNet!")
    def _init_non_incr_timesteps(self, total_it, min_step_r=0.02, max_step_r=0.80):
        # --------- non-increasing t sampling ---------
        min_step = int(min_step_r * self.num_train_timesteps)
        max_step = int(max_step_r * self.num_train_timesteps)
        time_prior = [800, 500, 300, 100]
        r1, r2, s1, s2 = time_prior # r1, r2 for range, s1 s2 for exponents
        weights = torch.cat(
            (
                torch.exp( # 800-900
                    -(torch.arange(max_step, r1, -1) - r1)
                        / (2 * s1)
                    ),
                torch.ones(r1 - r2 + 1), # 500-800
                torch.exp( # 20-500
                        -(torch.arange(r2 - 1, min_step, -1) - r2) / (2 * s2)
                    ),
            )
        )
        weights = weights / torch.sum(weights)
        cumulative_density = torch.cumsum(weights, dim=0)
        t_choice = self._gen_nonlinear_t_choice(total_it, cumulative_density, min_step, max_step)
        return t_choice

    def _gen_nonlinear_t_choice(self, max_it: int, cumulative_density, min_step, max_step):
        total_num_steps = max_step - min_step
        t_choice = []
        for i in range(0, max_it + 1):
            current_it_ratio = i / (max_it + 1)
            time_index = torch.where(
                        (cumulative_density - current_it_ratio) > 0
                    )[0][0]
            if time_index == 0 or torch.abs(
                cumulative_density[time_index] - current_it_ratio
            ) < torch.abs(
                cumulative_density[time_index - 1] - current_it_ratio
            ):
                t = total_num_steps - time_index
            else:
                t = total_num_steps - time_index + 1
            t = torch.clip(t, min_step, max_step + 1)
            t = torch.full((1,), t, dtype=torch.long, device=self.device)
            t_choice.append(t)
        return t_choice

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    # @torch.cuda.amp.autocast(enabled=False)
    @torch.cuda.amp.autocast()
    def forward_controlnet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        image_cond: Float[Tensor, "..."],
        condition_scale: float,
        encoder_hidden_states: Float[Tensor, "..."],
        guess_mode: bool = False,
    ) -> Float[Tensor, "..."]:
        return self.controlnet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=image_cond.to(self.weights_dtype),
            conditioning_scale=condition_scale,
            guess_mode=guess_mode,
            return_dict=False,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_control_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        cross_attention_kwargs,
        down_block_additional_residuals,
        mid_block_additional_residual,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)


    def prepare_image_cond(self, cond_rgb: Float[Tensor, "B H W C"]):
        if self.cfg.control_type == "normal" or self.cfg.control_type == "normal_sd21":
            imgs = (cond_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
            detected_maps = [self.preprocessor(img) for img in imgs]
            controls = torch.from_numpy(np.array(detected_maps)).float().to(self.device) / 255.0
            controls = controls.permute(0, 3, 1, 2)
        elif self.cfg.control_type == "canny":
            imgs = (cond_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
            blurred_imgs = [cv2.blur(img, ksize=(5, 5)) for img in imgs]
            detected_maps = [self.preprocessor(img, self.cfg.canny_lower_bound, self.cfg.canny_upper_bound) for img in blurred_imgs]
            controls = torch.from_numpy(np.array(detected_maps)).float().to(self.device) / 255.0
            controls = controls.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
        elif self.cfg.control_type == 'inpaint':
            controls = cond_rgb.permute(0, 3, 1, 2)
        elif self.cfg.control_type == 'tile':
            controls = cond_rgb.permute(0, 3, 1, 2)
        else:
            raise NotImplementedError(f"How to prepare image cond for control type {self.cfg.control_type}? not implemented yet.")
        return controls

    def compute_grad_sds(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        img_orig: Float[Tensor, "B 3 H W"],
        image_cond: Float[Tensor, "B 3 H W"],
        t: Int[Tensor, "B"],
        sdse=False,
    ):
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            if sdse:
                pos_emb, neg_emb = text_embeddings.chunk(2)
                text_embeddings = torch.cat([neg_emb, pos_emb, neg_emb]) # [neg, pos, neg]
                controlnet_model_input = latents_noisy
                latent_model_input = torch.cat([latents_noisy] * 3) # BBB
                controlnet_t_input = t
                unet_t_input = torch.cat([t] * 3)
                controlnet_text_embeddings = neg_emb
                controlnet_image_cond = image_cond
                guess_mode = self.cfg.guess_mode
                if not self.cfg.tgt_with_cond:
                    # add no_image_condition
                    no_image_condition = torch.zeros_like(image_cond, device=self.device)
                    controlnet_image_cond = torch.cat([image_cond, no_image_condition])
                    controlnet_model_input = torch.cat([latents_noisy] * 2) # BB
                    controlnet_t_input = torch.cat([t] * 2)
                    controlnet_text_embeddings = torch.cat([neg_emb] * 2)
            else:
                controlnet_model_input = torch.cat([latents_noisy] * 2) # BB
                latent_model_input = controlnet_model_input
                controlnet_t_input = torch.cat([t] * 2)
                unet_t_input = controlnet_t_input
                controlnet_text_embeddings = text_embeddings # pos, neg
                controlnet_image_cond = torch.cat([image_cond] * 2)
                guess_mode = False

            if self.phase_id == 1:
                condition_scale = 0. # this means no condition, same as sds
            elif self.phase_id == 2:
                condition_scale = self.cfg.condition_scale

            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                controlnet_model_input,
                controlnet_t_input,
                encoder_hidden_states=controlnet_text_embeddings,
                image_cond=controlnet_image_cond,
                condition_scale=condition_scale,
                guess_mode=guess_mode,
            )

            if self.cfg.enough_small_t_threshold > 0:
                # t is a batch of numbers. create a mask for t < threshold
                cond_mask = t < self.cfg.enough_small_t_threshold
                down_block_res_samples = [sample * cond_mask.view(-1, 1, 1, 1) for sample in down_block_res_samples]
                mid_block_res_sample = mid_block_res_sample * cond_mask.view(-1, 1, 1, 1)

            if sdse:
                if self.cfg.tgt_with_cond:
                    down_block_res_samples = [torch.cat([ d, d, torch.zeros_like(d)]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([mid_block_res_sample, 
                                                    mid_block_res_sample, 
                                                    torch.zeros_like(mid_block_res_sample)])
                else:
                    down_block_res_samples = [
                        torch.cat(
                            [d, torch.zeros_like(d[: d.shape[0] // 2])]
                        )
                        for d in down_block_res_samples
                    ]

                    mid_block_res_sample = torch.cat(
                        [mid_block_res_sample, # res, res_zero_img
                         torch.zeros_like(mid_block_res_sample[mid_block_res_sample.shape[0] // 2:])] # 0
                    )


            noise_pred = self.forward_control_unet(
                latent_model_input,
                unet_t_input,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

        # perform classifier-free guidance
        if sdse:
            noise_pred_img, noise_pred_text, noise_pred_uncond = noise_pred.chunk(3)
            if self.cfg.weighting_strategy == "sds":
                # w(t), sigma_t^2
                w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            elif self.cfg.weighting_strategy == "uniform":
                w = 1
            elif self.cfg.weighting_strategy == "fantasia3d":
                w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
            else:
                raise ValueError(
                    f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
                )
            ###################
            if self.cfg.pure_seeking:
                # grad = w * (noise_pred_text - noise_pred_img)
                curr_term = noise_pred_text - noise_pred_img
                grad = w * curr_term #(noise_pred_text - noise_pred_img)
            else:
                if self.cfg.guidance_design == 0:
                    mode_seeking = self.cfg.guidance_scale * (noise_pred_text - noise_pred_img) + noise_pred_text
                    curr_term = mode_seeking
                    grad = w * (curr_term - noise)
                else:
                    raise ValueError(
                        f"Unknown guidance design: {self.cfg.guidance_design}"
                    )

            # Apply image condition filtering if enabled
            if self.cfg.filter_img_cond is not None:
                self.scheduler.set_timesteps(50)
                self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)

                bs = noise_pred_img.shape[0]
                large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t[:bs].unsqueeze(-1)
                idxs = torch.min(large_enough_idxs, dim=1)[1]
                t_validate = self.scheduler.timesteps_gpu[idxs]

                pred_1orig_image_cond = []
                for b in range(bs):
                    step_output = self.scheduler.step(
                        noise_pred_img[b : b + 1], t_validate[b], latents_noisy[b : b + 1], eta=1
                    )
                    pred_1orig_image_cond.append(step_output["pred_original_sample"])
                pred_1orig_image_cond = torch.cat(pred_1orig_image_cond)
                
                imgs_1orig_image_cond = self.decode_latents(pred_1orig_image_cond)
                
                img_src_loss = []
                for b in range(bs):
                    if self.cfg.filter_img_cond == 'mse':
                        img_src_loss.append(
                            torch.nn.functional.mse_loss(imgs_1orig_image_cond[b], img_orig[b]).unsqueeze(0)
                        )
                    else:
                        raise ValueError(f"filter_img_cond '{self.cfg.filter_img_cond}' not supported. Only 'mse' is implemented.")
                
                img_src_loss = torch.cat(img_src_loss)
                mask = img_src_loss < self.cfg.filter_img_cond_weight
                
                grad = grad * mask.view(-1, 1, 1, 1)
                
            ###################
            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            return grad, (noise_pred, latents_noisy, noise_pred_img, noise_pred_text, noise_pred_uncond, curr_term)
        else:
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            grad = w * (noise_pred - noise)
        return grad, (noise_pred, latents_noisy)

    def get_finetune_loss(self,
        latents: Float[Tensor, "B 4 DH DW"],
        img_orig: Float[Tensor, "B 3 H W"],
        image_cond: Float[Tensor, "B 3 H W"],
        t: Int[Tensor, "B"],
        text_embeddings: Float[Tensor, "BB 77 768"],
    ):
        latents = latents.detach()
        batch_size = text_embeddings.shape[0] // 2

        # --- Compute noise prediction via ControlNet + UNet ---
        # IMPORTANT: Don't use torch.no_grad() here since we need gradients to flow to ControlNet
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        
        _, neg_emb = text_embeddings.chunk(2)
        
        down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
            latents_noisy,
            t,
            encoder_hidden_states=neg_emb,
            image_cond=image_cond,
            condition_scale=self.cfg.condition_scale,
            guess_mode=False,
        )
        
        with torch.no_grad():
            noise_pred_image = self.forward_control_unet(
                latents_noisy,
                t,
                encoder_hidden_states=neg_emb,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

        self.scheduler.set_timesteps(10)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)

        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )
        large_enough_idxs = self.scheduler.timesteps_gpu.expand(bs, -1) > t[:bs].unsqueeze(-1)
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t_validate = self.scheduler.timesteps_gpu[idxs]  # shape [bs]

        pred_1orig_image_cond_list = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred_image[b : b + 1], t_validate[b], latents_noisy[b : b + 1], eta=1
            )
            pred_1orig_image_cond_list.append(step_output["pred_original_sample"])
        pred_1orig_image_cond = torch.cat(pred_1orig_image_cond_list, dim=0)

        loss_finetune = torch.nn.functional.mse_loss(pred_1orig_image_cond, latents, reduction="mean")

        return loss_finetune

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        cond_rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        guidance_eval=False,
        **kwargs,
    ):
        batch_size, H, W, _ = rgb.shape
        # assert batch_size == 1
        assert rgb.shape[:-1] == cond_rgb.shape[:-1]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 DH DW"]
        if self.cfg.fixed_size > 0:
            RH, RW = self.cfg.fixed_size, self.cfg.fixed_size
        else:
            RH, RW = H // 8 * 8, W // 8 * 8
        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )
        latents = self.encode_images(rgb_BCHW_HW8)

        image_cond = self.prepare_image_cond(cond_rgb)
        image_cond = F.interpolate(
            image_cond, (RH, RW), mode="bilinear", align_corners=False
        )

        text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
        # !important: threestudio returns [pos, neg], which is different from the other implementations!!

        if self.cfg.use_non_incr_timesteps:
            t = self.t_choice[self.global_step].repeat(batch_size)
        else:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )

        if self.cfg.use_sds:
            grad, eval_vars = self.compute_grad_sds(text_embeddings, latents, rgb_BCHW_HW8, image_cond, t)
        elif self.cfg.use_sdse:
            grad, eval_vars = self.compute_grad_sds(text_embeddings, latents, rgb_BCHW_HW8, image_cond, t, sdse=True)

        grad = torch.nan_to_num(grad)
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
            "t": t[0],
        }

        if self.cfg.finetune:
            loss_finetune = self.get_finetune_loss(
                latents, rgb_BCHW_HW8, image_cond, t, text_embeddings
            )
            guidance_out.update({"loss_finetune": loss_finetune})

        if not self.cfg.use_sds and not self.cfg.use_sdse:
            edit_latents = self.edit_latents(text_embeddings, latents, image_cond, t)
            edit_images = self.decode_latents(edit_latents)
            edit_images = F.interpolate(edit_images, (H, W), mode="bilinear")

            return {"edit_images": edit_images.permute(0, 2, 3, 1)}

        if guidance_eval:
            if self.cfg.use_sdse:
                noise_pred, latents_noisy, noise_pred_img, noise_pred_text, noise_pred_uncond, curr_term = eval_vars
            else:
                noise_pred, latents_noisy = eval_vars
                noise_pred_img, noise_pred_text, noise_pred_uncond, curr_term = None, None, None, None
            guidance_eval_utils = {
                "img_cond": image_cond,
                "img_orig": rgb_BCHW_HW8,
                "use_perp_neg": prompt_utils.use_perp_neg,
                "neg_guidance_weights": None, #neg_guidance_weights,
                "text_embeddings": text_embeddings,
                "t_orig": t,
                "latents_noisy": latents_noisy,
                "noise_pred": noise_pred,
                "noise_pred_img": noise_pred_img,
                "noise_pred_text": noise_pred_text,
                "noise_pred_uncond": noise_pred_uncond,
                "curr_term": curr_term,
            }
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        guidance_eval=False,
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size, H, W, _ = rgb.shape
        cond_rgb = rgb.clone()

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 DH DW"]
        if self.cfg.fixed_size > 0:
            RH, RW = self.cfg.fixed_size, self.cfg.fixed_size
        else:
            RH, RW = H // 8 * 8, W // 8 * 8
        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )
        latents = self.encode_images(rgb_BCHW_HW8)

        image_cond = self.prepare_image_cond(cond_rgb)
        image_cond = F.interpolate(
            image_cond, (RH, RW), mode="bilinear", align_corners=False
        )

        text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
        # !important: threestudio returns [pos, neg], which is different from the other implementations!!

        if self.cfg.use_non_incr_timesteps:
            t = self.t_choice[self.global_step].repeat(batch_size)
        else:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )

        if self.cfg.use_sds:
            grad, eval_vars = self.compute_grad_sds(text_embeddings, latents, rgb_BCHW_HW8, image_cond, t)
        elif self.cfg.use_sdse:
            grad, eval_vars = self.compute_grad_sds(text_embeddings, latents, rgb_BCHW_HW8, image_cond, t, sdse=True)

        grad = torch.nan_to_num(grad)
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
            "t": t[0],
        }

        if self.cfg.finetune:
            loss_finetune = self.get_finetune_loss(
                latents, rgb_BCHW_HW8, image_cond, t, text_embeddings
            )
            guidance_out.update({"loss_finetune": loss_finetune})


        if guidance_eval:
            if self.cfg.use_sdse:
                noise_pred, latents_noisy, noise_pred_img, noise_pred_text, noise_pred_uncond, curr_term = eval_vars
            else:
                noise_pred, latents_noisy = eval_vars
                noise_pred_img, noise_pred_text, noise_pred_uncond, curr_term = None, None, None, None
            guidance_eval_utils = {
                "img_cond": image_cond,
                "img_orig": rgb_BCHW_HW8,
                "use_perp_neg": prompt_utils.use_perp_neg,
                "neg_guidance_weights": None, #neg_guidance_weights,
                "text_embeddings": text_embeddings,
                "t_orig": t,
                "latents_noisy": latents_noisy,
                "noise_pred": noise_pred,
                "noise_pred_img": noise_pred_img,
                "noise_pred_text": noise_pred_text,
                "noise_pred_uncond": noise_pred_uncond,
                "curr_term": curr_term,
            }
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out


    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        img_cond,
        img_orig,
        latents_noisy,
        noise_pred,
        noise_pred_img,
        noise_pred_text,
        noise_pred_uncond,
        curr_term,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        """
        Evaluate and visualize various intermediate states during SDS/SDSE training.

        Args:
            t_orig: The original step indices used to partially noise latents.
            text_embeddings: The text embeddings used (pos/neg or triple-chunk for SDSE).
            img_cond: Control image, shape [B, 3, H, W].
            img_orig: Original image, shape [B, 3, H, W].
            latents_noisy: Partially noised latents, shape [B, 4, H//8, W//8].
            noise_pred: The final noise prediction from the combination (e.g. uncond+guidance_scale*(text-img), etc.).
            noise_pred_img, noise_pred_text, noise_pred_uncond: The separated noise predictions if you want to see them individually.
            use_perp_neg, neg_guidance_weights: Additional settings if you use multi-negative prompts or perpendicular guidance.
        """
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)

        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size

        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs] # shape [bs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())

        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_1step_image_cond = []
        pred_1orig_image_cond = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred_img[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step_image_cond.append(step_output["prev_sample"])
            pred_1orig_image_cond.append(step_output["pred_original_sample"])
        latents_1step_image_cond = torch.cat(latents_1step_image_cond)
        pred_1orig_image_cond = torch.cat(pred_1orig_image_cond)
        imgs_1step_image_cond = self.decode_latents(latents_1step_image_cond).permute(0, 2, 3, 1)
        imgs_1orig_image_cond = self.decode_latents(pred_1orig_image_cond)#.permute(0, 2, 3, 1)
        img_src_loss = {'mse':[]}
        for b in range(bs):
            img_src_loss['mse'].append(torch.nn.functional.mse_loss(imgs_1orig_image_cond[b], img_orig[b]).item())

        imgs_1orig_image_cond = imgs_1orig_image_cond.permute(0, 2, 3, 1)


        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1].clone()
            # standard prompt embeddings for pos/neg
            if use_perp_neg:
                text_emb = text_embeddings[[b, b + len(idxs), b + 2 * len(idxs)], ...]
            else:
                text_emb = text_embeddings[[b, b + len(idxs)], ...]
            img_cond_slice = img_cond[[b], ...]
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None

            # proceed from t[i]+1 down to 0
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False, desc=f'Evaluating guidance for step {i + 1}'):
                # pred noise
                ########### get noise pred ###

                if use_perp_neg:
                    NotImplementedError("use_perp_neg is not implemented yet")
                else:
                    latent_model_input = torch.cat([latents] * 2)
                    if self.phase_id == 1:
                        condition_scale = 0.
                    elif self.phase_id == 2:
                        condition_scale = self.cfg.condition_scale
                    down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_emb,
                        image_cond=img_cond_slice,
                        condition_scale=condition_scale,
                    )
                    noise_pred_2chunk = self.forward_control_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_emb,
                        cross_attention_kwargs=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    )

                    noise_pred_text_2, noise_pred_uncond_2 = noise_pred_2chunk.chunk(2)
                    noise_pred_2chunk = noise_pred_uncond_2 + self.cfg.guidance_scale * (noise_pred_text_2 - noise_pred_uncond_2)

                ###########

                # step to get next(prev) latents
                latents = self.scheduler.step(noise_pred_2chunk, t, latents).prev_sample
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        latents_final_unet = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1].clone()
            if use_perp_neg:
                text_emb = text_embeddings[[b, b + len(idxs), b + 2 * len(idxs)], ...]
            else:
                text_emb = text_embeddings[[b, b + len(idxs)], ...]

            for step_idx in self.scheduler.timesteps[i + 1 :]:
                latent_model_input = torch.cat([latents] * 2, dim=0)
                # purely unet: no controlnet
                noise_pred_unet_only = self.unet(
                    latent_model_input.to(self.weights_dtype),
                    step_idx.to(self.weights_dtype),
                    encoder_hidden_states=text_emb.to(self.weights_dtype),
                ).sample.to(latents.dtype)


                noise_pred_text_uo, noise_pred_uncond_uo = noise_pred_unet_only.chunk(2)
                noise_pred_unet_only = noise_pred_uncond_uo + self.cfg.guidance_scale * (
                    noise_pred_text_uo - noise_pred_uncond_uo
                )
                latents = self.scheduler.step(noise_pred_unet_only, step_idx, latents).prev_sample

            latents_final_unet.append(latents)

        latents_final_unet = torch.cat(latents_final_unet, dim=0)
        imgs_final_unet = self.decode_latents(latents_final_unet).permute(0, 2, 3, 1)


        latents_final_guess = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1].clone()
            if use_perp_neg:
                text_emb = text_embeddings[[b, b + len(idxs), b + 2 * len(idxs)], ...]
            else:
                text_emb = text_embeddings[[b, b + len(idxs)], ...]
            img_cond_slice = img_cond[[b], ...]

            for step_idx in self.scheduler.timesteps[i + 1 :]:
                pos_emb, neg_emb = text_emb.chunk(2)
                controlnet_model_input = latents
                latent_model_input = torch.cat([latents] * 2) # BB
                controlnet_t_input = step_idx
                unet_t_input = step_idx # torch.cat([step_idx] * 2)
                controlnet_text_embeddings = neg_emb

                down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                    controlnet_model_input,
                    controlnet_t_input,
                    encoder_hidden_states=controlnet_text_embeddings,
                    image_cond=img_cond_slice,
                    condition_scale=self.cfg.condition_scale,
                    guess_mode=False #True,
                )

                # concatenate res, 0
                down_block_res_samples = [torch.cat([ d, torch.zeros_like(d)]) for d in down_block_res_samples]
                mid_block_res_sample = torch.cat([mid_block_res_sample, 
                                                torch.zeros_like(mid_block_res_sample)])

                noise_pred_guess = self.forward_control_unet(
                    latent_model_input,
                    unet_t_input,
                    encoder_hidden_states=text_emb,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )

                # perform classifier-free guidance
                noise_pred_text_guess, noise_pred_uncond_guess = noise_pred_guess.chunk(2)
                noise_pred_guess = noise_pred_uncond_guess + self.cfg.guidance_scale * (
                    noise_pred_text_guess - noise_pred_uncond_guess
                )

                ##############################
                latents = self.scheduler.step(noise_pred_guess, step_idx, latents).prev_sample

            latents_final_guess.append(latents)

        latents_final_guess = torch.cat(latents_final_guess, dim=0)
        imgs_final_guess_mode = self.decode_latents(latents_final_guess).permute(0, 2, 3, 1)

        imgs_noise_pred_img = self.decode_latents(noise_pred_img[:bs]).permute(0, 2, 3, 1)
        imgs_noise_pred_text = self.decode_latents(noise_pred_text[:bs]).permute(0, 2, 3, 1)
        imgs_noise_pred_uncond = self.decode_latents(noise_pred_uncond[:bs]).permute(0, 2, 3, 1)

        imgs_text_minus_img = self.decode_latents(noise_pred_text[:bs] - noise_pred_img[:bs]).permute(0, 2, 3, 1)
        imgs_curr_term = self.decode_latents(curr_term[:bs]).permute(0, 2, 3, 1)

        img_cond_resized = F.interpolate(
            img_cond, size=(imgs_final.shape[1], imgs_final.shape[2]),
            mode='bilinear', align_corners=False
        ).permute(0, 2, 3, 1)

        formatted_str = (
    f"1ori_img_cond\n"
    f"mse: {', '.join(f'{x:.4f}' for x in img_src_loss['mse'])}"
)


        return {
            "bs": bs,
            "noise_levels": fracs,
            "texts": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "final_from_controlnet": imgs_final,       # Final image after normal CFG + ControlNet
            "final_only_unet": imgs_final_unet,        # Final image after skipping ControlNet
            "imgs_1step_image_cond": imgs_1step_image_cond,
            formatted_str: imgs_1orig_image_cond,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        self.global_step = global_step

        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if not self.cfg.use_non_incr_timesteps:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )


if __name__ == "__main__":
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.typing import Optional

    cfg = load_config("configs/debugging/controlnet-normal.yaml")
    guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
    prompt_processor = threestudio.find(cfg.system.prompt_processor_type)(
        cfg.system.prompt_processor
    )

    rgb_image = cv2.imread("assets/face.jpg")[:, :, ::-1].copy() / 255
    rgb_image = torch.FloatTensor(rgb_image).unsqueeze(0).to(guidance.device)
    prompt_utils = prompt_processor()
    guidance_out = guidance(rgb_image, rgb_image, prompt_utils)
    edit_image = (
        (guidance_out["edit_images"][0].detach().cpu().clip(0, 1).numpy() * 255)
        .astype(np.uint8)[:, :, ::-1]
        .copy()
    )
    os.makedirs(".threestudio_cache", exist_ok=True)
    cv2.imwrite(".threestudio_cache/edit_image.jpg", edit_image)