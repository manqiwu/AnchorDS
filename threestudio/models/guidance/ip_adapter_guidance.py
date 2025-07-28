from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.models import ImageProjection
from transformers import CLIPVisionModelWithProjection
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import (
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
)
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *
from gaussiansplatting.utils.loss_utils import ssim
from gaussiansplatting.lpipsPyTorch import lpips


@threestudio.register("ip-adapter-guidance")
class StableDiffusionIpAdapterGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        ip_adapter_name_or_path: str = "h94/IP-Adapter"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        condition_scale: float = 1.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

        tgt_with_cond: bool = False
        use_gen_tgt_img: int = 0 # 1: generate tgt img for each batch; 2: generate tgt img once and fix it
        generator_seed: int = 0

        guidance_design: int = 0
        filter_img_cond: Optional[str] = None
        filter_img_cond_weight: float = 0.03

        stage_two_start_step: int = -1
        enough_small_t_threshold: int = -1 # if set, will use this threshold to determine if the t is small enough. -1 means not using this threshold

        finetune: bool = False
        finetune_layer_name: str = "" # "mid_block" or "up_blocks.3"


    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        # for ip-adapter plus
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.cfg.ip_adapter_name_or_path,
            subfolder="models/image_encoder",
            torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            image_encoder=image_encoder,
            **pipe_kwargs,
        ).to(self.device)

        self.pipe.load_ip_adapter(self.cfg.ip_adapter_name_or_path, subfolder="models", weight_name="ip-adapter-plus_sd15.bin")
        self.pipe.set_ip_adapter_scale(self.cfg.condition_scale)
        
        if self.cfg.use_gen_tgt_img:
            self.generator = torch.Generator(device="cpu").manual_seed(self.cfg.generator_seed)

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

        del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        if self.cfg.finetune:
            self.unet = self.pipe.unet.train()
        else:
            self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        self.pipe.image_encoder.requires_grad_(False)

        # Only fine-tune modules in the 'mid_block' to reduce memory usage.
        if self.cfg.finetune:
            assert self.cfg.finetune_layer_name in ["up_blocks.3", "mid_block"], f"finetune_layer_name must be 'up_blocks.3' or 'mid_block', but got {self.cfg.finetune_layer_name}"
            for name, module in self.unet.attn_processors.items():
                if self.cfg.finetune_layer_name in name and isinstance(module, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)):
                    module.train()
                    for param_name, param in module.named_parameters():
                        param.requires_grad_(True)
                        param.data = param.data.float()
                        print(f"{name}.{param_name} dtype: {param.dtype}")
                    threestudio.info(f"Unfreezing (small part) IP-Adapter parameters in attn_processor: {name}")
        else:
            threestudio.info("Finetuning disabled; all UNet parameters remain frozen.")

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.phase_id = 1
        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Stable Diffusion!")


    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast()
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        added_cond_kwargs: Optional[dict] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            added_cond_kwargs=added_cond_kwargs,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, device, batch_size, t, tgt_img=None, tgt_img_embs=None
    ):
        image_embeds = []
        negative_image_embeds = []
        if not isinstance(ip_adapter_image, list):
            ip_adapter_image = [ip_adapter_image]

        if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
            raise ValueError(
                f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
            )

        for single_ip_adapter_image, image_proj_layer in zip(
            ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
        ):
            output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
            single_image_embeds, single_negative_image_embeds = self.pipe.encode_image(
                single_ip_adapter_image, device, 1, output_hidden_state
            )

            image_embeds.append(single_image_embeds[None, :])
            negative_image_embeds.append(single_negative_image_embeds[None, :])

        if tgt_img is not None:
            output_tgt_hidden_state = not isinstance(image_proj_layer, ImageProjection)
            single_tgt_image_embeds, _ = self.pipe.encode_image(
                tgt_img, device, 1, output_tgt_hidden_state
            )

            single_tgt_image_embeds = single_tgt_image_embeds[None, :]

        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            single_image_embeds = torch.cat([single_image_embeds] * batch_size, dim=0)
            single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * batch_size, dim=0)
            if self.cfg.tgt_with_cond:
                single_image_embeds = torch.cat([single_image_embeds, single_image_embeds], dim=0)
            else:
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
            # tgt_img condition
            if tgt_img is not None:
                single_image_embeds = torch.cat([single_tgt_image_embeds, single_image_embeds], dim=0)
            elif tgt_img_embs is not None:
                single_image_embeds = torch.cat([tgt_img_embs, single_image_embeds], dim=0)

            if self.cfg.enough_small_t_threshold > 0:
                mask = ( t < self.cfg.enough_small_t_threshold ).view(-1,1,1,1).repeat(2,1,1,1) # 8,1,1,1
                single_image_embeds = single_image_embeds * mask

                
            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds.to(self.weights_dtype))

        return ip_adapter_image_embeds

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        with torch.no_grad():
            image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        img_orig: Float[Tensor, "B 3 512 512"],
        image_cond_embeds: Float[Tensor, "..."],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = elevation.shape[0]
        added_cond_kwargs = (
            {"image_embeds": image_cond_embeds}
        )

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_image = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_image
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_image
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_image + accum_grad
            )
            curr_term = noise_pred - noise
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs,
                )

            noise_pred_text, noise_pred_image = noise_pred.chunk(2)
            if self.cfg.guidance_design == 0:
                # vanilla SDS decomposition
                # w * (e(y, I) - e(y_neg, I)) + e(y, I) - e
                noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_image
                )
                curr_term = noise_pred - noise
            else:
                raise NotImplementedError(f"Unknown guidance design: {self.cfg.guidance_design}")

        if self.cfg.weighting_strategy == "sds":
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)

        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size

        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t_validate = self.scheduler.timesteps_gpu[idxs] # shape [bs]

        pred_1orig_image_cond = []
        for b in range(noise_pred_image.shape[0]):
            step_output = self.scheduler.step(
                noise_pred_image[b : b + 1], t_validate[b], latents_noisy[b : b + 1], eta=1
            )
            pred_1orig_image_cond.append(step_output["pred_original_sample"])
        pred_1orig_image_cond = torch.cat(pred_1orig_image_cond)
        imgs_1orig_image_cond = self.decode_latents(pred_1orig_image_cond) #.permute(0, 2, 3, 1)
        img_src_loss = []
        if self.cfg.filter_img_cond != None:
            for b in range(noise_pred_image.shape[0]):
                if self.cfg.filter_img_cond == 'mse':
                    img_src_loss.append(torch.nn.functional.mse_loss(imgs_1orig_image_cond[b], img_orig[b]).unsqueeze(0))
                elif self.cfg.filter_img_cond == 'lpips':
                    img_src_loss.append(lpips(imgs_1orig_image_cond[b], img_orig[b], net_type='vgg'))
                elif self.cfg.filter_img_cond == 'ssim':
                    img_src_loss.append(ssim(imgs_1orig_image_cond[b], img_orig[b]))
                else:
                    raise ValueError(f"Unknown filter_img_cond: {self.cfg.filter_img_cond}")
            img_src_loss = torch.cat(img_src_loss)
            mask = img_src_loss < self.cfg.filter_img_cond_weight

            grad = w * curr_term * mask.view(-1, 1, 1, 1)
        else:
            grad = w * curr_term

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "img_orig": img_orig,
            "noise_pred": noise_pred,
            "noise_pred_text": noise_pred_text,
            "noise_pred_image": noise_pred_image,
            "added_cond_kwargs": added_cond_kwargs,
        }

        return grad, guidance_eval_utils
    
    def get_finetune_loss(self,
        latents: Float[Tensor, "B 4 64 64"],
        img_orig: Float[Tensor, "B 3 512 512"],
        image_cond_embeds: Float[Tensor, "..."],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        # Detach latents so that gradients do not backpropagate into the latent generator.
        latents = latents.detach()
        batch_size = elevation.shape[0]
        added_cond_kwargs = {"image_embeds": [image_cond_embeds[0][batch_size:]]}

        # --- Compute noise prediction via UNet (and thus IPAdapter) ---
        # Get text embeddings normally.
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        noise_pred_image = self.forward_unet(
            latents_noisy,
            t,
            encoder_hidden_states=text_embeddings[batch_size:],
            added_cond_kwargs=added_cond_kwargs,
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
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
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

        rgb_BCHW_224 = F.interpolate(
                rgb_BCHW, (224, 224), mode="bilinear", align_corners=False
            )

        cond_rgb = rgb_BCHW_224.clone()

        if self.cfg.use_gen_tgt_img == 1:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            tgt_img = self.pipe(
                prompt_embeds=text_embeddings[:batch_size],
                negative_prompt_embeds=text_embeddings[batch_size:],
                ip_adapter_image=cond_rgb,
                num_inference_steps=50,
                generator=self.generator,
            ).images  # tgt_img is a list of PIL.Image

            tgt_img = torch.stack([ToTensor()(img) for img in tgt_img])
            tgt_img_BCHW_224 = F.interpolate(
                tgt_img, (224, 224), mode="bilinear", align_corners=False
            )

            image_cond_embeds = self.prepare_ip_adapter_image_embeds(cond_rgb, self.device, batch_size, t, tgt_img=tgt_img_BCHW_224)
        elif self.cfg.use_gen_tgt_img == 2:
            if not hasattr(self, 'tgt_img_embs'):
                text_embeddings = prompt_utils.get_text_embeddings(
                    elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
                )
                tgt_img = self.pipe(
                    prompt_embeds=text_embeddings[:batch_size],
                    negative_prompt_embeds=text_embeddings[batch_size:],
                    ip_adapter_image=cond_rgb,
                    num_inference_steps=100,
                    generator=self.generator,
                ).images  # tgt_img is a list of PIL.Image
                tgt_img = ToTensor()(tgt_img[0]).repeat(batch_size, 1, 1, 1)
                tgt_img_BCHW_224 = F.interpolate(
                    tgt_img, (224, 224), mode="bilinear", align_corners=False
                )
                image_cond_embeds = self.prepare_ip_adapter_image_embeds(cond_rgb, self.device, batch_size, t, tgt_img=tgt_img_BCHW_224)
                self.tgt_img_embs = image_cond_embeds[0][:batch_size]
            else:
                image_cond_embeds = self.prepare_ip_adapter_image_embeds(cond_rgb, self.device, batch_size, t, tgt_img_embs=self.tgt_img_embs)
        else:
            image_cond_embeds = self.prepare_ip_adapter_image_embeds(cond_rgb, self.device, batch_size, t) # 1) #batch_size)
            


        grad, guidance_eval_utils = self.compute_grad_sds(
            latents, rgb_BCHW_512, image_cond_embeds, t, prompt_utils, elevation, azimuth, camera_distances
        )

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
        }

        if self.cfg.finetune:
            loss_finetune = self.get_finetune_loss(
                latents, rgb_BCHW_512, image_cond_embeds, t, prompt_utils, elevation, azimuth, camera_distances
            )
            guidance_out.update({"loss_finetune": loss_finetune})


        if self.cfg.use_gen_tgt_img == 1 or (self.cfg.use_gen_tgt_img == 2 and self.global_step == 0):
            guidance_eval_utils.update({"tgt_img": tgt_img})

        if guidance_eval:
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
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
        added_cond_kwargs=None,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
            )
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        img_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
        noise_pred_text=None,
        noise_pred_image=None,
        added_cond_kwargs=None,
        tgt_img=None,
    ):
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)

        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )

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
                noise_pred_image[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step_image_cond.append(step_output["prev_sample"])
            pred_1orig_image_cond.append(step_output["pred_original_sample"])
        latents_1step_image_cond = torch.cat(latents_1step_image_cond)
        pred_1orig_image_cond = torch.cat(pred_1orig_image_cond)
        imgs_1step_image_cond = self.decode_latents(latents_1step_image_cond).permute(0, 2, 3, 1)
        imgs_1orig_image_cond = self.decode_latents(pred_1orig_image_cond) #.permute(0, 2, 3, 1)

        img_src_loss = {'mse':[], 'lpips':[], 'ssim':[]}
        for b in range(bs):
            img_src_loss['mse'].append(torch.nn.functional.mse_loss(imgs_1orig_image_cond[b], img_orig[b]).item())
            img_src_loss['lpips'].append(lpips(imgs_1orig_image_cond[b], img_orig[b], net_type='vgg').item())
            img_src_loss['ssim'].append(ssim(imgs_1orig_image_cond[b], img_orig[b]).item())

        imgs_1orig_image_cond = imgs_1orig_image_cond.permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None

            # proceed from t[i]+1 down to 0
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False, desc=f'Evaluating guidance for step {i + 1}'):
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, use_perp_neg, neg_guid, added_cond_kwargs
                )
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        formatted_str = (
    f"1ori_img_cond\n"
    f"mse: {', '.join(f'{x:.4f}' for x in img_src_loss['mse'])}\n"
    f"lpips: {', '.join(f'{x:.4f}' for x in img_src_loss['lpips'])}\n"
    f"ssim: {', '.join(f'{x:.4f}' for x in img_src_loss['ssim'])}\n"
)


        out_dict = {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
            "imgs_1step_image_cond": imgs_1step_image_cond,
            formatted_str: imgs_1orig_image_cond,
        }

        if tgt_img is not None:
            out_dict.update({"tgt_img": tgt_img.permute(0, 2, 3, 1)})

        return out_dict

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        self.global_step = global_step

        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
