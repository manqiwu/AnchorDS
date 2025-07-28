import os
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.exporters.base import Exporter, ExporterOutput
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.base import Updateable, update_if_possible
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import C, cleanup, get_device, load_module_weights
from threestudio.utils.saving import SaverMixin
from threestudio.utils.typing import *


class BaseSystem(pl.LightningModule, Updateable, SaverMixin):
    @dataclass
    class Config:
        loggers: dict = field(default_factory=dict)
        loss: dict = field(default_factory=dict)
        optimizer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = None
        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None
        cleanup_after_validation_step: bool = False
        cleanup_after_test_step: bool = False

    cfg: Config

    def __init__(self, cfg, resumed=False) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self._save_dir: Optional[str] = None
        self._resumed: bool = resumed
        self._resumed_eval: bool = False
        self._resumed_eval_status: dict = {"global_step": 0, "current_epoch": 0}
        if "loggers" in cfg:
            self.create_loggers(cfg.loggers)

        self.configure()
        if self.cfg.weights is not None:
            self.load_weights(self.cfg.weights, self.cfg.weights_ignore_modules)
        self.post_configure()

    def load_weights(self, weights: str, ignore_modules: Optional[List[str]] = None):
        state_dict, epoch, global_step = load_module_weights(
            weights, ignore_modules=ignore_modules, map_location="cpu"
        )
        self.load_state_dict(state_dict, strict=False)
        # restore step-dependent states
        self.do_update_step(epoch, global_step, on_load_weights=True)

    def set_resume_status(self, current_epoch: int, global_step: int):
        # restore correct epoch and global step in eval
        self._resumed_eval = True
        self._resumed_eval_status["current_epoch"] = current_epoch
        self._resumed_eval_status["global_step"] = global_step

    @property
    def resumed(self):
        # whether from resumed checkpoint
        return self._resumed

    @property
    def true_global_step(self):
        if self._resumed_eval:
            return self._resumed_eval_status["global_step"]
        else:
            return self.global_step

    @property
    def true_current_epoch(self):
        if self._resumed_eval:
            return self._resumed_eval_status["current_epoch"]
        else:
            return self.current_epoch

    def configure(self) -> None:
        pass

    def post_configure(self) -> None:
        """
        executed after weights are loaded
        """
        pass

    def C(self, value: Any) -> float:
        return C(value, self.true_current_epoch, self.true_global_step)

    def configure_optimizers(self):
        optim = parse_optimizer(self.cfg.optimizer, self)
        ret = {
            "optimizer": optim,
        }
        if self.cfg.scheduler is not None:
            ret.update(
                {
                    "lr_scheduler": parse_scheduler(self.cfg.scheduler, optim),
                }
            )
        return ret

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        if self.cfg.cleanup_after_validation_step:
            # cleanup to save vram
            cleanup()

    def on_validation_epoch_end(self):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_test_batch_end(self, outputs, batch, batch_idx):
        if self.cfg.cleanup_after_test_step:
            # cleanup to save vram
            cleanup()

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_predict_batch_end(self, outputs, batch, batch_idx):
        if self.cfg.cleanup_after_test_step:
            # cleanup to save vram
            cleanup()

    def on_predict_epoch_end(self):
        pass

    def preprocess_data(self, batch, stage):
        pass

    """
    Implementing on_after_batch_transfer of DataModule does the same.
    But on_after_batch_transfer does not support DP.
    """

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        self.preprocess_data(batch, "train")
        self.dataset = self.trainer.train_dataloader.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "validation")
        self.dataset = self.trainer.val_dataloaders.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "test")
        self.dataset = self.trainer.test_dataloaders.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "predict")
        self.dataset = self.trainer.predict_dataloaders.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        pass

    def on_before_optimizer_step(self, optimizer):
        """
        # some gradient-related debugging goes here, example:
        from lightning.pytorch.utilities import grad_norm
        norms = grad_norm(self.geometry, norm_type=2)
        print(norms)
        """
        pass


class BaseLift3DSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        geometry_type: str = ""
        geometry: dict = field(default_factory=dict)
        geometry_convert_from: Optional[str] = None
        geometry_convert_inherit_texture: bool = False
        # used to override configurations of the previous geometry being converted from,
        # for example isosurface_threshold
        geometry_convert_override: dict = field(default_factory=dict)

        material_type: str = ""
        material: dict = field(default_factory=dict)

        background_type: str = ""
        background: dict = field(default_factory=dict)

        renderer_type: str = ""
        renderer: dict = field(default_factory=dict)

        guidance_type: str = ""
        guidance: dict = field(default_factory=dict)

        prompt_processor_type: str = ""
        prompt_processor: dict = field(default_factory=dict)

        # geometry export configurations, no need to specify in training
        exporter_type: str = "mesh-exporter"
        exporter: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        if (
            self.cfg.geometry_convert_from  # from_coarse must be specified
            and not self.cfg.weights  # not initialized from coarse when weights are specified
            and not self.resumed  # not initialized from coarse when resumed from checkpoints
        ):
            threestudio.info("Initializing geometry from a given checkpoint ...")
            from threestudio.utils.config import load_config, parse_structured

            prev_cfg = load_config(
                os.path.join(
                    os.path.dirname(self.cfg.geometry_convert_from),
                    "../configs/parsed.yaml",
                )
            )  # TODO: hard-coded relative path
            prev_system_cfg: BaseLift3DSystem.Config = parse_structured(
                self.Config, prev_cfg.system
            )
            prev_geometry_cfg = prev_system_cfg.geometry
            prev_geometry_cfg.update(self.cfg.geometry_convert_override)
            prev_geometry = threestudio.find(prev_system_cfg.geometry_type)(
                prev_geometry_cfg
            )
            state_dict, epoch, global_step = load_module_weights(
                self.cfg.geometry_convert_from,
                module_name="geometry",
                map_location="cpu",
            )
            prev_geometry.load_state_dict(state_dict, strict=False)
            # restore step-dependent states
            prev_geometry.do_update_step(epoch, global_step, on_load_weights=True)
            # convert from coarse stage geometry
            prev_geometry = prev_geometry.to(get_device())
            self.geometry = threestudio.find(self.cfg.geometry_type).create_from(
                prev_geometry,
                self.cfg.geometry,
                copy_net=self.cfg.geometry_convert_inherit_texture,
            )
            del prev_geometry
            cleanup()
        else:
            self.geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)

        self.material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        self.background = threestudio.find(self.cfg.background_type)(
            self.cfg.background
        )
        self.renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )

    def on_fit_start(self) -> None:
        if self._save_dir is not None:
            threestudio.info(f"Validation results will be saved to {self._save_dir}")
        else:
            threestudio.warn(
                f"Saving directory not set for the system, visualization results will not be saved"
            )

    def on_test_end(self) -> None:
        if self._save_dir is not None:
            threestudio.info(f"Test results saved to {self._save_dir}")

    def on_predict_start(self) -> None:
        self.exporter: Exporter = threestudio.find(self.cfg.exporter_type)(
            self.cfg.exporter,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )

    def predict_step(self, batch, batch_idx):
        if self.exporter.cfg.save_video:
            self.test_step(batch, batch_idx)

    def on_predict_epoch_end(self) -> None:
        if self.exporter.cfg.save_video:
            self.on_test_epoch_end()
        exporter_output: List[ExporterOutput] = self.exporter()
        for out in exporter_output:
            save_func_name = f"save_{out.save_type}"
            if not hasattr(self, save_func_name):
                raise ValueError(f"{save_func_name} not supported by the SaverMixin")
            save_func = getattr(self, save_func_name)
            save_func(f"it{self.true_global_step}-export/{out.save_name}", **out.params)

    def on_predict_end(self) -> None:
        if self._save_dir is not None:
            threestudio.info(f"Export assets saved to {self._save_dir}")

    def guidance_evaluation_save_depreciated(self, comp_rgb, guidance_eval_out):
        # B, size = comp_rgb.shape[:2]
        # resize = lambda x: F.interpolate(
        #     x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        # ).permute(0, 2, 3, 1)
        size = max(comp_rgb.size(1), comp_rgb.size(2))

        pad_ = lambda x: F.pad(
                    x.permute(0, 3, 1, 2),
                    pad=(
                        (size - x.size(2)) // 2,  # Padding for width (left)
                        (size - x.size(2) + 1) // 2,  # Padding for width (right)
                        (size - x.size(1)) // 2,  # Padding for height (top)
                        (size - x.size(1) + 1) // 2,  # Padding for height (bottom)
                    ),
                    mode="constant",
                    value=0
                ).permute(0, 2, 3, 1)
        filename = f"it{self.true_global_step}-train.png"

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])

        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb",
                    "img": merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(pad_(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(pad_(guidance_eval_out["imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(pad_(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(pad_(guidance_eval_out["imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ),
            name="train_step",
            step=self.true_global_step,
            texts=guidance_eval_out["texts"],
        )

    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        """
        - Automatically determines which keys in `guidance_eval_out` are image tensors of shape [B,H,W,C].
        - Adds a "title row" as row 0 in each batch (i.e. shape [B+1,H,W,C]) so that
        each column has a textual "header."
        - The rest of the code calls self.save_image_grid(...) so that each column is
        displayed as columns, and the batch dimension is displayed as rows.
        """

        filename = f"it{self.true_global_step}-train.png"

        # 1. Decide a base "size" from comp_rgb. We'll make the main images squared to [size x size].
        #    We'll also define the total "main image height" = `size`.
        #    Then inside that area, we can place a smaller row for the text at the top of row 0.
        size = max(comp_rgb.size(1), comp_rgb.size(2))  # comp_rgb is [B, H, W, C] shape

        # 2. A helper to pad or center an image [B,H,W,C] to [B,size,size,C]
        def pad_to_square(imgs: torch.Tensor, desired_size: int) -> torch.Tensor:
            B, H, W, C = imgs.shape
            assert C==3, "This function assumes the last dimension is 3 (RGB)."
            pad_left   = (desired_size - W) // 2
            pad_right  = (desired_size - W + 1) // 2
            pad_top    = (desired_size - H) // 2
            pad_bottom = (desired_size - H + 1) // 2
            imgs_padded = F.pad(
                imgs.permute(0, 3, 1, 2),
                (pad_left, pad_right, pad_top, pad_bottom),  # (left, right, top, bottom)
                mode="constant",
                value=0.0,
            ).permute(0, 2, 3, 1)
            return imgs_padded

        # 3. Identify all image-like tensors in `guidance_eval_out`. We'll store them in
        #    an ordered dict, so we can keep a stable column order. 
        #    We'll also show `comp_rgb` as the first column.
        from collections import OrderedDict
        images_dict = OrderedDict()
        images_dict["comp_rgb"] = comp_rgb  # user-labeled name for the composite image

        for k, v in guidance_eval_out.items():
            # We treat any 4D Tensor as an image. 
            if isinstance(v, torch.Tensor) and v.ndim == 4:
                images_dict[k] = v  # shape [B,H,W,C]

        # 4. Helper: Render text into a small image [1, text_height, text_width, 3] using PIL, then
        #    pad it to [1, main_height, main_width, 3], i.e. [1, size, size, 3].
        def create_title_row_for_key(key: str, width: int, main_height: int) -> torch.Tensor:
            """
            Returns shape [1, main_height, width, 3]. The top part contains a small text band,
            the rest is black. 
            """
            text_height = max(main_height // 10, 100)  # some fraction of main image height
            # We'll create a text image of [1, text_height, width, 3].
            small_title = text_to_image(key, width, text_height)  # -> [1,text_height,width,3]

            # Now we pad that to [1, main_height, width, 3].
            # i.e. fill the bottom portion with black.
            B, H, W, C = small_title.shape
            pad_top    = 0
            pad_bottom = main_height - H
            pad_left   = 0
            pad_right  = 0
            # pad: (left, right, top, bottom)
            padded = F.pad(
                small_title.permute(0, 3, 1, 2),
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0.0,
            ).permute(0, 2, 3, 1)
            return padded  # shape [1, main_height, width, 3]

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])  # [B,H,W,C] -> [B*H*W,C]

        def text_to_image(txt: str, width: int, height: int) -> torch.Tensor:
            """
            Simple PIL-based text rendering. Returns a float tensor [1,height,width,3].
            """
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np

            # create a blank white image
            img_pil = Image.new("RGB", (width, height), color=(255, 255, 255))
            draw = ImageDraw.Draw(img_pil)
            font_size = 15
            # !attention! font path is hard-coded here
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", size=font_size)
            draw.text((10, 10), txt, fill=(100, 100, 0), font=font)  # Draw black text
            arr = np.array(img_pil).astype(np.float32) / 255.0  # [H, W, 3]
            ten = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W,3]
            return ten

        # 5. Build a final list of columns, each a dict. We do the following:
        #    - For each key in images_dict, we get the "main images" of shape [B,H,W,C].
        #    - We pad the main images to [B,size,size,3].
        #    - We create a "title row" of shape [1,size,size,3].
        #    - Then cat along dim=0 to get [B+1,size,size,3].
        #    - That single tensor is given to self.save_image_grid(...) as one column.
        columns_list = []
        for key, val in images_dict.items():
            # val is shape [B,H,W,C]
            # pad to [B,size,size,3]
            val_padded = pad_to_square(val, size)

            # create title row [1,size,size,3]
            # we know the final width = size, final height = size
            title_row = create_title_row_for_key(key, width=size, main_height=size).to(val.device)

            # cat => [B+1,size,size,3]
            final_tensor = torch.cat([title_row, val_padded], dim=0)

            # build a dict for self.save_image_grid
            columns_list.append({
                "type": "rgb",
                "img": merge12(final_tensor),  # shape [B+1,size,size,3] -> [(B+1)*size*size,3]
                "kwargs": {"data_format": "HWC"}
            })

        # 6. We can retrieve any textual annotation from `guidance_eval_out["texts"]`, or none.
        texts = guidance_eval_out.get("texts", [])

        # 7. Finally call save_image_grid
        self.save_image_grid(
            filename=filename,
            imgs=columns_list,  # each item is a "column"
            name="train_step",
            step=self.true_global_step,
            texts=texts,
        )