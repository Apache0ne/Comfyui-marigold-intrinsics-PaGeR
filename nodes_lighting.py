from contextlib import nullcontext

import numpy as np
import torch

import comfy.model_management as mm
from comfy.utils import ProgressBar

from PIL import Image

from .nodes_shared import _mask_to_keep, _require_module


class MarigoldIIDLighting:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "iid_model": ("IIDMODEL",),
                "images": ("IMAGE",),
                "denoise_steps": ("INT", {"default": 4, "min": 0, "max": 50, "step": 1}),
                "processing_resolution": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 1}),
                "match_input_resolution": ("BOOLEAN", {"default": True}),
                "ensemble_size": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "shading", "residual")
    FUNCTION = "process"
    CATEGORY = "Marigold IID Lighting"

    DESCRIPTION = "Runs Marigold IID Lighting and returns albedo/shading/residual maps."

    def process(
        self,
        iid_model,
        images,
        denoise_steps: int = 4,
        processing_resolution: int = 768,
        match_input_resolution: bool = True,
        ensemble_size: int = 1,
        mask: torch.Tensor | None = None,
    ):
        _require_module("diffusers", "pip install -r custom_nodes/Comfyui-marigold-intrinsics/requirements.txt")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        pipe = iid_model.get("pipe", None) if isinstance(iid_model, dict) else None
        dtype = iid_model.get("dtype", torch.float16) if isinstance(iid_model, dict) else torch.float16
        autocast_dtype = iid_model.get("autocast_dtype", dtype) if isinstance(iid_model, dict) else dtype
        kind = iid_model.get("kind", None) if isinstance(iid_model, dict) else None
        keep_on_gpu = iid_model.get("keep_on_gpu", False) if isinstance(iid_model, dict) else False
        if pipe is None:
            raise RuntimeError("Invalid IIDMODEL input: missing 'pipe'. Re-run the loader node.")
        if kind is not None and kind != "lighting":
            raise RuntimeError(
                f"This node expects the *lighting/HyperSim* model, but got kind={kind!r}. Use `Load Marigold IID Lighting Model`."
            )

        pipe = pipe.to(device)

        B, _, _, _ = images.shape
        pbar = ProgressBar(B)

        steps = None if int(denoise_steps) <= 0 else int(denoise_steps)
        proc_res = int(processing_resolution)

        autocast_condition = (autocast_dtype != torch.float32) and not mm.is_device_mps(device)

        albedo_out = []
        shading_out = []
        residual_out = []

        try:
            with torch.inference_mode():
                with torch.autocast(mm.get_autocast_device(device), dtype=autocast_dtype) if autocast_condition else nullcontext():
                    for i in range(B):
                        img = images[i].detach().cpu().clamp(0, 1).numpy()
                        img_u8 = (img * 255.0).round().astype(np.uint8)
                        pil_img = Image.fromarray(img_u8)

                        intrinsics = pipe(
                            pil_img,
                            num_inference_steps=steps,
                            processing_resolution=proc_res,
                            match_input_resolution=bool(match_input_resolution),
                            ensemble_size=int(ensemble_size),
                        )

                        vis = pipe.image_processor.visualize_intrinsics(intrinsics.prediction, pipe.target_properties)[0]

                        def _pil_to_image_tensor(pil: Image.Image) -> torch.Tensor:
                            arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
                            return torch.from_numpy(arr).unsqueeze(0)

                        albedo_out.append(_pil_to_image_tensor(vis["albedo"]))
                        shading_key = "shading" if "shading" in vis else None
                        if shading_key is None:
                            shading_key = "diffuse_shading" if "diffuse_shading" in vis else None
                        if shading_key is None:
                            shading_key = next((k for k in vis.keys() if "shad" in k.lower()), None)

                        residual_key = "residual" if "residual" in vis else None
                        if residual_key is None:
                            residual_key = next((k for k in vis.keys() if "resid" in k.lower()), None)

                        if shading_key is None or residual_key is None:
                            raise RuntimeError(
                                "Lighting model outputs missing expected keys. "
                                f"Available visualize_intrinsics keys: {sorted(vis.keys())}. "
                                "Make sure you're using `Load Marigold IID Lighting Model`."
                            )

                        shading_out.append(_pil_to_image_tensor(vis[shading_key]))
                        residual_out.append(_pil_to_image_tensor(vis[residual_key]))

                        pbar.update(1)
        finally:
            if keep_on_gpu:
                try:
                    pipe.to(device)
                except Exception:
                    pass
            else:
                try:
                    offload_is_cpu = hasattr(offload_device, "type") and offload_device.type == "cpu"
                    half_dtype = dtype in (torch.float16, torch.bfloat16)
                    if offload_is_cpu and half_dtype:
                        pipe.to(device)
                    else:
                        pipe.to(offload_device)
                except Exception:
                    pass
            mm.soft_empty_cache()

        albedo = torch.cat(albedo_out, dim=0).clamp(0, 1).cpu().float()
        shading = torch.cat(shading_out, dim=0).clamp(0, 1).cpu().float()
        residual = torch.cat(residual_out, dim=0).clamp(0, 1).cpu().float()

        if mask is not None:
            keep = _mask_to_keep(mask.detach().cpu().float(), (albedo.shape[1], albedo.shape[2]), albedo.shape[0]).unsqueeze(-1)
            albedo = albedo * keep
            shading = shading * keep
            residual = residual * keep

        return (albedo, shading, residual)
