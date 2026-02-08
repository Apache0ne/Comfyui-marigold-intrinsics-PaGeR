import math
import os
import sys
import shutil
import time
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

from .nodes_shared import (
    STORAGE_DIRNAME,
    _mask_to_keep,
    _require_module,
    _select_torch_dtype,
)


PAGER_MODEL_IDS = [
    "prs-eth/PaGeR-depth",
    "prs-eth/PaGeR-metric-depth",
    "prs-eth/PaGeR-depth-indoor",
    "prs-eth/PaGeR-metric-depth-indoor",
    "prs-eth/PaGeR-normals-Structured3D",
]
PAGER_UNET_WEIGHT_PATH = "unet/diffusion_pytorch_model.safetensors"

DEFAULT_MIN_DEPTH = math.log(1e-2)
DEFAULT_DEPTH_RANGE = math.log(75.0)

_BASE_REQUIRED_FILES = [
    "scheduler/scheduler_config.json",
    "vae/config.json",
]

_VAE_WEIGHT_CANDIDATES = [
    "vae/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.bin",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.bin",
]

_DEPTH_CMAP = None
_PAGER_TAESD_PREFERRED = ["taesd", "taesdxl"]


def _pager_taesd_choices() -> list[str]:
    # Reuse split-loader discovery so choices match ComfyUI vae_approx availability.
    try:
        from .splitloader import _vae_list

        available = set(_vae_list([]))
        choices = [name for name in _PAGER_TAESD_PREFERRED if name in available]
        if choices:
            return choices
    except Exception:
        pass
    return ["taesd"]


def _comfy_vae_scaling_factor(vae) -> float:
    first_stage = getattr(vae, "first_stage_model", None)
    if first_stage is not None and hasattr(first_stage, "vae_scale"):
        try:
            scale = first_stage.vae_scale
            if isinstance(scale, torch.Tensor):
                return float(scale.detach().cpu().item())
            return float(scale)
        except Exception:
            pass
    return 0.18215


class _PaGeRComfyVAEAdapter(torch.nn.Module):
    def __init__(self, comfy_vae):
        super().__init__()
        self._vae = comfy_vae
        self.use_tiling = False
        self.use_slicing = False
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def encode(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        pixels = ((x / 2.0) + 0.5).clamp(0.0, 1.0).movedim(1, -1)
        if self.use_tiling and hasattr(self._vae, "encode_tiled"):
            latents = self._vae.encode_tiled(pixels)
        else:
            latents = self._vae.encode(pixels)
        if latents.device != x.device:
            latents = latents.to(x.device)
        return latents

    def decode(self, z: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        if self.use_tiling and hasattr(self._vae, "decode_tiled"):
            pixels = self._vae.decode_tiled(z)
        else:
            pixels = self._vae.decode(z)
        sample = pixels.movedim(-1, 1) * 2.0 - 1.0
        if sample.device != z.device:
            sample = sample.to(z.device)
        return sample

    def enable_tiling(self, enabled: bool = True):
        self.use_tiling = bool(enabled)
        return self

    def disable_tiling(self):
        self.use_tiling = False
        return self

    def enable_slicing(self):
        self.use_slicing = True
        return self

    def disable_slicing(self):
        self.use_slicing = False
        return self

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        del non_blocking
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def requires_grad_(self, requires_grad: bool):
        del requires_grad
        return self

    def eval(self):
        return self


def _get_depth_cmap():
    global _DEPTH_CMAP
    if _DEPTH_CMAP is None:
        _require_module("matplotlib", "pip install -r custom_nodes/Comfyui-marigold-intrinsics/requirements.txt")
        from matplotlib import pyplot as plt

        _DEPTH_CMAP = plt.get_cmap("Spectral")
    return _DEPTH_CMAP

def _ensure_pager_paths() -> Path:
    repo_root = Path(__file__).resolve().parent
    src_root = repo_root / "src"
    marigold_root = repo_root / "Marigold"
    if not src_root.exists():
        raise RuntimeError(f"PaGeR 'src' folder not found at '{src_root}'.")
    if not marigold_root.exists():
        raise RuntimeError(f"PaGeR 'Marigold' folder not found at '{marigold_root}'.")
    repo_root_s = str(repo_root)
    if repo_root_s not in sys.path:
        sys.path.insert(0, repo_root_s)

    src_mod = sys.modules.get("src")
    if src_mod is not None:
        src_file = getattr(src_mod, "__file__", None)
        src_paths = getattr(src_mod, "__path__", None)
        owns_src = False
        if isinstance(src_file, str) and str(src_root) in src_file:
            owns_src = True
        if src_paths is not None:
            try:
                owns_src = any(str(src_root) in str(p) for p in list(src_paths))
            except Exception:
                pass
        if not owns_src:
            for key in list(sys.modules.keys()):
                if key == "src" or key.startswith("src."):
                    del sys.modules[key]

    marigold_mod = sys.modules.get("Marigold")
    if marigold_mod is not None:
        marigold_file = getattr(marigold_mod, "__file__", None)
        marigold_paths = getattr(marigold_mod, "__path__", None)
        owns_marigold = False
        if isinstance(marigold_file, str) and str(marigold_root) in marigold_file:
            owns_marigold = True
        if marigold_paths is not None:
            try:
                owns_marigold = any(str(marigold_root) in str(p) for p in list(marigold_paths))
            except Exception:
                pass
        if not owns_marigold:
            for key in list(sys.modules.keys()):
                if key == "Marigold" or key.startswith("Marigold."):
                    del sys.modules[key]
    return repo_root


def _pager_storage_root() -> str:
    return os.path.join(folder_paths.models_dir, STORAGE_DIRNAME, "pager")


def _pager_repo_dir(repo_id: str) -> str:
    return os.path.join(_pager_storage_root(), repo_id.replace("/", "__"))


def _shared_base_dir(variant: str) -> str:
    return os.path.join(folder_paths.models_dir, STORAGE_DIRNAME, "base", variant)


def _link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _ensure_pager_file(repo_id: str, remote_path: str, local_path: str) -> None:
    if os.path.exists(local_path):
        print(f"[PaGeR Loader] using local: {local_path}")
        return
    from huggingface_hub import hf_hub_download

    print(f"[PaGeR Loader] downloading {repo_id}/{remote_path}")
    src_path = hf_hub_download(repo_id=repo_id, filename=remote_path)
    _link_or_copy(src_path, local_path)


def _has_any_file(root: str, candidates: list[str]) -> bool:
    for rel in candidates:
        if os.path.exists(os.path.join(root, rel)):
            return True
    return False


def _is_valid_shared_base(root: str) -> bool:
    for rel in _BASE_REQUIRED_FILES:
        if not os.path.exists(os.path.join(root, rel)):
            return False
    if not _has_any_file(root, _VAE_WEIGHT_CANDIDATES):
        return False
    return True


def _resolve_shared_base(precision: str) -> str | None:
    pref = (precision or "auto").lower()
    if pref == "fp32":
        order = ["fp32", "fp16"]
    else:
        order = ["fp16", "fp32"]

    for variant in order:
        base_dir = _shared_base_dir(variant)
        if _is_valid_shared_base(base_dir):
            print(f"[PaGeR Loader] using shared base variant '{variant}': {base_dir}")
            return base_dir
    return None


def _preferred_base_variant(precision: str, dtype: torch.dtype) -> str:
    pref = (precision or "auto").lower()
    if pref == "fp32":
        return "fp32"
    if pref in ("fp16", "bf16"):
        return "fp16"
    return "fp32" if dtype == torch.float32 else "fp16"


def _ensure_pager_base_files(model_variant: str, repo_id_source: str) -> str:
    if model_variant not in ("fp16", "fp32"):
        raise ValueError(f"Invalid model_variant={model_variant!r}")

    base = _shared_base_dir(model_variant)
    files = [
        ("scheduler/scheduler_config.json", "scheduler/scheduler_config.json"),
        ("vae/config.json", "vae/config.json"),
    ]
    if model_variant == "fp16":
        files.append(("vae/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.safetensors"))
    else:
        files.append(("vae/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"))

    for remote_path, rel_dst in files:
        _ensure_pager_file(repo_id_source, remote_path, os.path.join(base, rel_dst))
    return base


def _ensure_pager_model_files(model_repo: str) -> str:
    model_dir = _pager_repo_dir(model_repo)
    files = [
        ("config.yaml", "config.yaml"),
        ("unet/config.json", "unet/config.json"),
        (PAGER_UNET_WEIGHT_PATH, PAGER_UNET_WEIGHT_PATH),
    ]

    for remote_path, rel_dst in files:
        _ensure_pager_file(model_repo, remote_path, os.path.join(model_dir, rel_dst))

    print(f"[PaGeR Loader] model files ready: {model_dir}")
    return model_dir


def _load_pager_config(local_repo_dir: str):
    cfg_path = os.path.join(local_repo_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise RuntimeError(f"Missing config.yaml in '{local_repo_dir}'.")
    from omegaconf import OmegaConf

    return OmegaConf.load(cfg_path)


def _infer_modality(cfg, repo_id: str) -> str:
    modality = ""
    try:
        modality = str(cfg.model.modality).lower()
    except Exception:
        modality = ""
    if modality.startswith("depth"):
        return "depth"
    if "normal" in modality:
        return "normal"
    if "normal" in repo_id.lower():
        return "normal"
    if "depth" in repo_id.lower():
        return "depth"
    raise RuntimeError(f"Could not infer modality from config or repo_id='{repo_id}'.")


def _pager_to(pager, device: torch.device, dtype: torch.dtype) -> None:
    pager.device = device
    pager.weight_dtype = dtype
    if hasattr(pager, "vae"):
        pager.vae.to(device=device, dtype=dtype)
    if hasattr(pager, "unet"):
        for unet in pager.unet.values():
            unet.to(device=device, dtype=dtype)
    if hasattr(pager, "empty_encoding"):
        pager.empty_encoding = pager.empty_encoding.to(device=device, dtype=dtype)
    if hasattr(pager, "alpha_prod"):
        pager.alpha_prod = pager.alpha_prod.to(device=device, dtype=dtype)
    if hasattr(pager, "beta_prod"):
        pager.beta_prod = pager.beta_prod.to(device=device, dtype=dtype)
    if hasattr(pager, "PE_cubemap"):
        pager.PE_cubemap = pager.PE_cubemap.to(device=device, dtype=dtype)


def _resize_hwc(image: torch.Tensor, target_hw: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if image.shape[1:3] == target_hw:
        return image
    image_nchw = image.permute(0, 3, 1, 2)
    resized = F.interpolate(
        image_nchw,
        size=target_hw,
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
    )
    return resized.permute(0, 2, 3, 1)


def _resize_mask_bhw(mask: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError(f"mask must be [B,H,W], got shape={tuple(mask.shape)}")
    if mask.shape[1:3] == target_hw:
        return mask
    mask_nchw = mask.unsqueeze(1)
    resized = F.interpolate(mask_nchw, size=target_hw, mode="nearest")
    return resized.squeeze(1)


def _broadcast_bhw(mask: torch.Tensor, target_b: int) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError(f"mask must be [B,H,W], got shape={tuple(mask.shape)}")
    b = int(mask.shape[0])
    if b == target_b:
        return mask
    if b == 1 and target_b > 1:
        return mask.repeat(target_b, 1, 1)
    raise ValueError(f"Mask batch mismatch: got {b}, expected {target_b}")


def _prepare_erp_image(image_hwc: torch.Tensor, target_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    image_nchw = image_hwc.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    image_nchw = F.interpolate(image_nchw, size=target_hw, mode="bilinear", align_corners=False)
    image_chw = image_nchw.squeeze(0).clamp(0.0, 1.0)
    image_chw = image_chw * 2.0 - 1.0
    return image_chw


def _depth_to_hwc(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 4:
        if depth.shape[1] == 1:
            depth_hw = depth[:, 0]
        elif depth.shape[-1] == 1:
            return depth
        else:
            raise ValueError(f"Unexpected depth shape={tuple(depth.shape)}")
    elif depth.ndim == 3:
        depth_hw = depth
    elif depth.ndim == 2:
        depth_hw = depth.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected depth shape={tuple(depth.shape)}")
    return depth_hw.unsqueeze(-1)


def _normal_to_hwc(normal: torch.Tensor) -> torch.Tensor:
    if normal.ndim == 4:
        if normal.shape[1] == 3:
            return normal.permute(0, 2, 3, 1)
        if normal.shape[-1] == 3:
            return normal
    if normal.ndim == 3:
        if normal.shape[0] == 3:
            return normal.permute(1, 2, 0).unsqueeze(0)
        if normal.shape[-1] == 3:
            return normal.unsqueeze(0)
    raise ValueError(f"Unexpected normal shape={tuple(normal.shape)}")


def _compute_edge_mask_torch(
    depth: torch.Tensor,
    abs_thresh: float = 0.1,
    rel_thresh: float = 0.1,
) -> torch.Tensor:
    """
    Torch/GPU equivalent of src.utils.geometry_utils.compute_edge_mask.
    depth: [H,W] or [1,H,W] metric depth
    returns keep mask [1,H,W,1] float32 in [0,1]
    """
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"depth must be [H,W] (or [1,H,W]), got {tuple(depth.shape)}")

    depth = depth.to(dtype=torch.float32)
    valid = depth > 0
    eps = 1e-6
    edge = torch.zeros_like(valid, dtype=torch.bool)

    d1 = depth[:, :-1]
    d2 = depth[:, 1:]
    v_pair = valid[:, :-1] & valid[:, 1:]
    diff = (d1 - d2).abs()
    rel = diff / (torch.minimum(d1, d2) + eps)
    edge_pair = v_pair & (diff > float(abs_thresh)) & (rel > float(rel_thresh))
    edge[:, :-1] |= edge_pair
    edge[:, 1:] |= edge_pair

    d1 = depth[:-1, :]
    d2 = depth[1:, :]
    v_pair = valid[:-1, :] & valid[1:, :]
    diff = (d1 - d2).abs()
    rel = diff / (torch.minimum(d1, d2) + eps)
    edge_pair = v_pair & (diff > float(abs_thresh)) & (rel > float(rel_thresh))
    edge[:-1, :] |= edge_pair
    edge[1:, :] |= edge_pair

    keep = (valid & (~edge)).to(dtype=torch.float32)
    return keep.unsqueeze(0).unsqueeze(-1)


def _prepare_image_for_logging_np(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = (image * 255.0).astype(np.uint8)
    return image


def _depth_viz_to_color_hwc(pred_viz: torch.Tensor) -> torch.Tensor:
    """
    Match HF Space visualization:
    1) clip to [min, q99]
    2) min-max normalize to uint8
    3) apply matplotlib Spectral colormap
    Returns IMAGE-shaped tensor [1,H,W,3] float32 in [0,1].
    """
    cmap = _get_depth_cmap()
    arr = pred_viz.detach().float().cpu().numpy()
    arr = np.clip(arr, arr.min(), np.quantile(arr, 0.99))
    arr_u8 = _prepare_image_for_logging_np(arr)

    if arr_u8.ndim == 4:
        arr_u8 = arr_u8[0]
    if arr_u8.ndim == 3 and arr_u8.shape[0] == 1:
        arr_u8 = arr_u8[0]
    elif arr_u8.ndim == 3 and arr_u8.shape[-1] == 1:
        arr_u8 = arr_u8[..., 0]
    if arr_u8.ndim != 2:
        raise RuntimeError(f"Unexpected depth viz shape for colormap: {arr_u8.shape}")

    colored = cmap(arr_u8.astype(np.float32) / 255.0)[..., :3].astype(np.float32)
    return torch.from_numpy(colored).unsqueeze(0)


class DownloadAndLoadPaGeRModel:
    @classmethod
    def INPUT_TYPES(cls):
        taesd_choices = _pager_taesd_choices()
        default_taesd = "taesd" if "taesd" in taesd_choices else taesd_choices[0]
        return {
            "required": {
                "model_id": (PAGER_MODEL_IDS, {"default": "prs-eth/PaGeR-depth"}),
                "precision": (["auto", "fp16", "bf16", "fp32"], {"default": "auto"}),
                "keep_on_gpu": ("BOOLEAN", {"default": True}),
                "vae_tiling": ("BOOLEAN", {"default": False, "advanced": True}),
                "use_comfy_taesd_vae": ("BOOLEAN", {"default": False}),
                "taesd_vae_name": (taesd_choices, {"default": default_taesd, "advanced": True}),
                "force_gpu": ("BOOLEAN", {"default": True, "advanced": True}),
            },
        }

    RETURN_TYPES = ("PAGERMODEL",)
    RETURN_NAMES = ("pager_model",)
    FUNCTION = "load"
    CATEGORY = "PaGeR"

    DESCRIPTION = """
Downloads & loads PaGeR models (depth or normals) from HuggingFace.
Weights are FP32 on disk but can be cast to FP16/BF16 at load time.
"""

    def load(
        self,
        model_id: str,
        precision: str,
        keep_on_gpu: bool = True,
        vae_tiling: bool = False,
        use_comfy_taesd_vae: bool = False,
        taesd_vae_name: str = "taesd",
        force_gpu: bool = True,
    ):
        _require_module("diffusers", "pip install -r custom_nodes/Comfyui-marigold-intrinsics/requirements.txt")
        _require_module("omegaconf", "pip install -r custom_nodes/Comfyui-marigold-intrinsics/requirements.txt")
        _require_module("pytorch360convert", "pip install -r custom_nodes/Comfyui-marigold-intrinsics/requirements.txt")
        _require_module("huggingface_hub", "pip install -r custom_nodes/Comfyui-marigold-intrinsics/requirements.txt")

        _ensure_pager_paths()
        from src.pager import Pager

        device = mm.get_torch_device()
        if force_gpu and hasattr(torch, "cuda") and torch.cuda.is_available():
            device = torch.device("cuda")

        dtype = _select_torch_dtype(precision, device)

        config = {
            "model_id": model_id,
            "precision": precision,
            "dtype": str(dtype),
            "keep_on_gpu": bool(keep_on_gpu),
            "vae_tiling": bool(vae_tiling),
            "use_comfy_taesd_vae": bool(use_comfy_taesd_vae),
            "taesd_vae_name": str(taesd_vae_name),
            "force_gpu": bool(force_gpu),
            "device": str(device),
        }
        if getattr(self, "_cache", None) is not None and self._cache.get("config") == config:
            return self._cache["value"]

        # Free/offload previous cached model when switching configs to avoid accumulating VRAM.
        if getattr(self, "_cache", None) is not None:
            old_value = self._cache.get("value", None)
            try:
                if isinstance(old_value, tuple) and len(old_value) > 0 and isinstance(old_value[0], dict):
                    old_pager_model = old_value[0]
                    old_pipe = old_pager_model.get("pipe", None)
                    old_dtype = old_pager_model.get("dtype", torch.float32)
                    if old_pipe is not None:
                        _pager_to(old_pipe, mm.unet_offload_device(), old_dtype)
            except Exception as e:
                print(f"[PaGeR Loader] Warning: failed to offload previous cached model: {e}")
            self._cache = None
            gc.collect()
            mm.soft_empty_cache()

        local_model_dir = _ensure_pager_model_files(model_id)
        cfg = _load_pager_config(local_model_dir)

        pretrained_repo = str(cfg.model.pretrained_path)
        shared_base_dir = _resolve_shared_base(precision)
        if shared_base_dir is not None:
            local_pretrained_dir = shared_base_dir
        else:
            target_variant = _preferred_base_variant(precision, dtype)
            print(
                f"[PaGeR Loader] shared base missing; downloading into shared base/{target_variant} from {pretrained_repo}"
            )
            try:
                local_pretrained_dir = _ensure_pager_base_files(target_variant, pretrained_repo)
            except Exception as e:
                if target_variant == "fp16":
                    # Some repos only publish fp32 weights; still keep shared-base layout.
                    print(
                        f"[PaGeR Loader] fp16 base download failed ({e}); retrying shared base/fp32 from {pretrained_repo}"
                    )
                    local_pretrained_dir = _ensure_pager_base_files("fp32", pretrained_repo)
                else:
                    raise

        modality = _infer_modality(cfg, model_id)
        model_configs = {
            modality: {
                "path": local_model_dir,
                "mode": "trained",
                "config": cfg.model,
            }
        }

        pager = Pager(
            model_configs=model_configs,
            pretrained_path=local_pretrained_dir,
            device=device,
            weight_dtype=dtype,
        )
        if use_comfy_taesd_vae:
            try:
                from .splitloader import _load_vae_by_name

                comfy_vae = _load_vae_by_name(str(taesd_vae_name), [])
                latent_channels = int(getattr(comfy_vae, "latent_channels", 4))
                if latent_channels != 4:
                    raise RuntimeError(
                        f"Selected TAESD '{taesd_vae_name}' has latent_channels={latent_channels}, expected 4."
                    )
                pager.vae = _PaGeRComfyVAEAdapter(comfy_vae)
                vae_scale = _comfy_vae_scaling_factor(comfy_vae)
                pager.rgb_latent_scale_factor = float(vae_scale)
                pager.depth_latent_scale_factor = float(vae_scale)
                print(
                    f"[PaGeR Loader] Using ComfyUI TAESD VAE '{taesd_vae_name}' "
                    f"(latent_scale={vae_scale:.5f})."
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load ComfyUI TAESD VAE '{taesd_vae_name}': {e}") from e
        _pager_to(pager, device, dtype)
        if hasattr(pager, "vae"):
            try:
                # PaGeR uses custom cubemap padding that requires face-batch=6.
                # VAE slicing splits batch into 1-face chunks and breaks that assumption.
                pager.vae.disable_slicing()
                print("[PaGeR Loader] VAE slicing: disabled (required by PaGeR)")
            except Exception as e:
                print(f"[PaGeR Loader] Warning: could not disable VAE slicing: {e}")
            try:
                if vae_tiling:
                    pager.vae.enable_tiling(True)
                else:
                    pager.vae.disable_tiling()
                print(f"[PaGeR Loader] VAE tiling: {'enabled' if vae_tiling else 'disabled'}")
            except Exception as e:
                print(f"[PaGeR Loader] Warning: could not set VAE tiling={vae_tiling}: {e}")
        if modality in pager.unet:
            pager.unet[modality].eval()

        pager_model = {
            "pipe": pager,
            "dtype": dtype,
            "autocast_dtype": dtype,
            "modality": modality,
            "log_scale": bool(getattr(cfg.model, "log_scale", False)),
            "metric_depth": bool(getattr(cfg.model, "metric_depth", False)),
            "repo_id": model_id,
            "pretrained_path": pretrained_repo,
            "keep_on_gpu": bool(keep_on_gpu),
            "vae_tiling": bool(vae_tiling),
            "use_comfy_taesd_vae": bool(use_comfy_taesd_vae),
            "taesd_vae_name": str(taesd_vae_name),
            "force_gpu": bool(force_gpu),
            "min_depth": DEFAULT_MIN_DEPTH,
            "depth_range": DEFAULT_DEPTH_RANGE,
        }

        result = (pager_model,)
        self._cache = {"config": config, "value": result}
        return result


class PaGeRInferCubemap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pager_model": ("PAGERMODEL",),
                "images": ("IMAGE",),
                "erp_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 1}),
                "face_w": ("INT", {"default": 768, "min": 32, "max": 2048, "step": 1}),
            }
        }

    RETURN_TYPES = ("PAGERCUBEMAP",)
    RETURN_NAMES = ("pager_cubemap",)
    FUNCTION = "process"
    CATEGORY = "PaGeR"

    DESCRIPTION = "Node A: Runs PaGeR encode+UNet+decode and returns decoded cubemap predictions for postprocess nodes."

    def process(
        self,
        pager_model,
        images: torch.Tensor,
        erp_height: int = 1024,
        face_w: int = 768,
    ):
        print("[PaGeR Infer Cubemap] Starting.")
        _require_module("pytorch360convert", "pip install -r custom_nodes/Comfyui-marigold-intrinsics/requirements.txt")
        _ensure_pager_paths()
        from src.utils.geometry_utils import erp_to_cubemap

        pipe = pager_model.get("pipe", None) if isinstance(pager_model, dict) else None
        if pipe is None:
            raise RuntimeError("Invalid PAGERMODEL input: missing 'pipe'. Re-run the loader node.")

        modality = str(pager_model.get("modality", ""))
        if modality not in ("depth", "normal"):
            raise RuntimeError(f"Unsupported PaGeR modality: {modality!r}")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        if pager_model.get("force_gpu", False) and hasattr(torch, "cuda") and torch.cuda.is_available():
            device = torch.device("cuda")

        dtype = pager_model.get("dtype", torch.float32)
        autocast_dtype = pager_model.get("autocast_dtype", dtype)
        autocast_condition = (autocast_dtype != torch.float32) and not mm.is_device_mps(device)
        vae_backend = "comfy_taesd" if bool(pager_model.get("use_comfy_taesd_vae", False)) else "pager_base"

        print(
            f"[PaGeR Infer Cubemap] modality={modality} device={device} dtype={dtype} "
            f"autocast={'on' if autocast_condition else 'off'} erp_height={erp_height} face_w={face_w} "
            f"vae_backend={vae_backend}"
        )
        if vae_backend == "comfy_taesd":
            print(f"[PaGeR Infer Cubemap] taesd_vae_name={pager_model.get('taesd_vae_name', 'taesd')}")
        if hasattr(pipe, "vae") and bool(getattr(pipe.vae, "use_slicing", False)):
            # Safety: slicing breaks custom cubemap padding path in PaGeR.
            try:
                pipe.vae.disable_slicing()
                print("[PaGeR Infer Cubemap] VAE slicing was on; forced off for PaGeR compatibility.")
            except Exception as e:
                print(f"[PaGeR Infer Cubemap] Warning: failed to disable VAE slicing: {e}")
        if hasattr(pipe, "vae"):
            print(
                f"[PaGeR Infer Cubemap] vae_tiling={bool(getattr(pipe.vae, 'use_tiling', False))}"
            )

        _pager_to(pipe, device, dtype)

        images = images.to(dtype=torch.float32)
        if images.ndim != 4 or images.shape[-1] not in (1, 3):
            raise ValueError(f"`images` must be IMAGE [B,H,W,1|3], got shape={tuple(images.shape)}")
        if images.shape[-1] == 1:
            images = images.repeat(1, 1, 1, 3)

        B, H, W, _ = images.shape
        pbar = ProgressBar(B)
        erp_w = int(erp_height) * 2
        target_hw = (int(erp_height), int(erp_w))

        pred_out = []

        try:
            with torch.inference_mode():
                with torch.autocast(mm.get_autocast_device(device), dtype=autocast_dtype) if autocast_condition else torch.no_grad():
                    for i in range(B):
                        print(f"[PaGeR Infer Cubemap] Processing image {i + 1}/{B}")
                        img = images[i]
                        t0 = time.perf_counter()
                        img_erp = _prepare_erp_image(img, target_hw, device)
                        t1 = time.perf_counter()
                        cubemap = erp_to_cubemap(img_erp, face_w=int(face_w)).unsqueeze(0).to(device)
                        t2 = time.perf_counter()
                        try:
                            pred_cubemap = pipe({"rgb_cubemap": cubemap}, modality)
                        except torch.OutOfMemoryError as oom:
                            print(f"[PaGeR Infer Cubemap] OOM during forward: {oom}")
                            mm.soft_empty_cache()
                            retried = False
                            if hasattr(pipe, "vae"):
                                try:
                                    if not bool(getattr(pipe.vae, "use_tiling", False)):
                                        pipe.vae.enable_tiling(True)
                                        retried = True
                                        print("[PaGeR Infer Cubemap] Retry with VAE tiling enabled.")
                                except Exception as e:
                                    print(f"[PaGeR Infer Cubemap] Failed to enable retry memory mode: {e}")
                            if not retried:
                                raise
                            pred_cubemap = pipe({"rgb_cubemap": cubemap}, modality)
                        t3 = time.perf_counter()
                        pred_out.append(pred_cubemap.detach())
                        pbar.update(1)
                        print(
                            f"[PaGeR Infer Cubemap] timings: prep={t1 - t0:.3f}s cubemap={t2 - t1:.3f}s "
                            f"unet+decode={t3 - t2:.3f}s total={t3 - t0:.3f}s"
                        )
        finally:
            if pager_model.get("keep_on_gpu", True):
                _pager_to(pipe, device, dtype)
            else:
                _pager_to(pipe, offload_device, dtype)
            mm.soft_empty_cache()

        pred_batch = torch.stack(pred_out, dim=0).contiguous()

        print(f"[PaGeR Infer Cubemap] Done. cubemap_shape={tuple(pred_batch.shape)}")
        return (
            {
                "pred_cubemap": pred_batch,
                "modality": modality,
                "input_hw": (int(H), int(W)),
                "target_hw": target_hw,
                "erp_height": int(erp_height),
                "face_w": int(face_w),
            },
        )


class PaGeRDepthPostprocess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pager_model": ("PAGERMODEL",),
                "pager_cubemap": ("PAGERCUBEMAP",),
                "match_input_resolution": ("BOOLEAN", {"default": True}),
                "min_depth": ("FLOAT", {"default": DEFAULT_MIN_DEPTH, "min": -100.0, "max": 100.0, "step": 0.001, "advanced": True}),
                "depth_range": ("FLOAT", {"default": DEFAULT_DEPTH_RANGE, "min": 1e-6, "max": 1000.0, "step": 0.001, "advanced": True}),
                "edge_abs_thresh": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "edge_rel_thresh": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 1.0, "step": 0.0001}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("depth", "edge_mask", "depth_color")
    FUNCTION = "process"
    CATEGORY = "PaGeR"

    DESCRIPTION = "Node B (depth): GPU postprocess from decoded cubemap to normalized depth + edge mask + colored depth."

    def process(
        self,
        pager_model,
        pager_cubemap,
        match_input_resolution: bool = True,
        min_depth: float = DEFAULT_MIN_DEPTH,
        depth_range: float = DEFAULT_DEPTH_RANGE,
        edge_abs_thresh: float = 0.002,
        edge_rel_thresh: float = 0.002,
        mask: torch.Tensor | None = None,
    ):
        print("[PaGeR Depth Postprocess] Starting.")
        _ensure_pager_paths()

        pipe = pager_model.get("pipe", None) if isinstance(pager_model, dict) else None
        if pipe is None:
            raise RuntimeError("Invalid PAGERMODEL input: missing 'pipe'. Re-run the loader node.")
        if pager_model.get("modality") != "depth":
            raise RuntimeError("This node expects a *depth* PaGeR model.")

        if not isinstance(pager_cubemap, dict):
            raise RuntimeError("Invalid PAGERCUBEMAP input.")
        if pager_cubemap.get("modality") != "depth":
            raise RuntimeError("PAGERCUBEMAP modality mismatch: expected depth.")

        pred_batch = pager_cubemap.get("pred_cubemap", None)
        if not isinstance(pred_batch, torch.Tensor):
            raise RuntimeError("Invalid PAGERCUBEMAP: missing tensor field 'pred_cubemap'.")
        if pred_batch.ndim != 5:
            raise RuntimeError(f"Invalid cubemap tensor shape {tuple(pred_batch.shape)}; expected [B,6,C,H,W].")

        input_hw = tuple(pager_cubemap.get("input_hw", (pred_batch.shape[-2], pred_batch.shape[-1])))
        target_hw = tuple(pager_cubemap.get("target_hw", input_hw))
        B = int(pred_batch.shape[0])
        device = pred_batch.device
        offload_device = mm.unet_offload_device()
        if device.type == "cpu":
            pref = mm.get_torch_device()
            if pager_model.get("force_gpu", False) and hasattr(torch, "cuda") and torch.cuda.is_available():
                pref = torch.device("cuda")
            if pref.type != "cpu":
                pred_batch = pred_batch.to(pref)
                device = pref

        dtype = pager_model.get("dtype", torch.float32)
        _pager_to(pipe, device, dtype)

        log_scale = bool(pager_model.get("log_scale", False))
        print(
            f"[PaGeR Depth Postprocess] device={device} cubemap_shape={tuple(pred_batch.shape)} "
            f"target_hw={target_hw} input_hw={input_hw} match_input_resolution={match_input_resolution}"
        )

        depth_out = []
        edge_out = []
        depth_color_out = []
        pbar = ProgressBar(B)

        try:
            with torch.inference_mode():
                for i in range(B):
                    print(f"[PaGeR Depth Postprocess] Processing cubemap {i + 1}/{B}")
                    t0 = time.perf_counter()
                    pred_metric, pred_viz = pipe.process_depth_output(
                        pred_batch[i].float(),
                        orig_size=target_hw,
                        min_depth=float(min_depth),
                        depth_range=float(depth_range),
                        log_scale=log_scale,
                    )
                    t1 = time.perf_counter()

                    if log_scale:
                        depth_norm = (pred_viz - float(min_depth)) / max(float(depth_range), 1e-8)
                    else:
                        depth_norm = (pred_metric - float(min_depth)) / max(float(depth_range), 1e-8)
                    depth_norm = depth_norm.clamp(0.0, 1.0)
                    depth_img = _depth_to_hwc(depth_norm)
                    depth_color_img = _depth_viz_to_color_hwc(pred_viz)

                    depth_metric = pred_metric
                    if depth_metric.ndim == 4 and depth_metric.shape[1] == 1:
                        depth_metric = depth_metric[0, 0]
                    elif depth_metric.ndim == 3:
                        depth_metric = depth_metric[0]
                    edge_img = _compute_edge_mask_torch(
                        depth_metric,
                        abs_thresh=float(edge_abs_thresh),
                        rel_thresh=float(edge_rel_thresh),
                    )

                    if match_input_resolution and target_hw != input_hw:
                        depth_img = _resize_hwc(depth_img, input_hw, mode="bilinear")
                        edge_img = _resize_hwc(edge_img, input_hw, mode="nearest")
                        depth_color_img = _resize_hwc(depth_color_img, input_hw, mode="bilinear")

                    depth_out.append(depth_img)
                    edge_out.append(edge_img)
                    depth_color_out.append(depth_color_img)
                    pbar.update(1)
                    t2 = time.perf_counter()
                    print(
                        f"[PaGeR Depth Postprocess] timings: erp={t1 - t0:.3f}s tail={t2 - t1:.3f}s total={t2 - t0:.3f}s"
                    )
        finally:
            if pager_model.get("keep_on_gpu", True):
                _pager_to(pipe, device, dtype)
            else:
                _pager_to(pipe, offload_device, dtype)
            mm.soft_empty_cache()

        depth = torch.cat(depth_out, dim=0).clamp(0.0, 1.0).float()
        edge = torch.cat(edge_out, dim=0).clamp(0.0, 1.0).float()
        depth_color = torch.cat(depth_color_out, dim=0).clamp(0.0, 1.0).float()

        if mask is not None:
            keep = _mask_to_keep(mask.detach().float(), (depth.shape[1], depth.shape[2]), depth.shape[0]).to(depth.device)
            keep = keep.unsqueeze(-1)
            depth = depth * keep
            edge = edge * keep
            depth_color = depth_color * keep

        depth = depth.repeat(1, 1, 1, 3)
        edge = edge.repeat(1, 1, 1, 3)

        print(
            f"[PaGeR Depth Postprocess] Done. depth_shape={tuple(depth.shape)} "
            f"edge_shape={tuple(edge.shape)} depth_color_shape={tuple(depth_color.shape)}"
        )
        return (depth, edge, depth_color)


class PaGeRNormalPostprocess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pager_model": ("PAGERMODEL",),
                "pager_cubemap": ("PAGERCUBEMAP",),
                "match_input_resolution": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal",)
    FUNCTION = "process"
    CATEGORY = "PaGeR"

    DESCRIPTION = "Node B (normal): GPU postprocess from decoded cubemap to normal visualization."

    def process(
        self,
        pager_model,
        pager_cubemap,
        match_input_resolution: bool = True,
        mask: torch.Tensor | None = None,
    ):
        print("[PaGeR Normal Postprocess] Starting.")
        _ensure_pager_paths()

        pipe = pager_model.get("pipe", None) if isinstance(pager_model, dict) else None
        if pipe is None:
            raise RuntimeError("Invalid PAGERMODEL input: missing 'pipe'. Re-run the loader node.")
        if pager_model.get("modality") != "normal":
            raise RuntimeError("This node expects a *normal* PaGeR model.")

        if not isinstance(pager_cubemap, dict):
            raise RuntimeError("Invalid PAGERCUBEMAP input.")
        if pager_cubemap.get("modality") != "normal":
            raise RuntimeError("PAGERCUBEMAP modality mismatch: expected normal.")

        pred_batch = pager_cubemap.get("pred_cubemap", None)
        if not isinstance(pred_batch, torch.Tensor):
            raise RuntimeError("Invalid PAGERCUBEMAP: missing tensor field 'pred_cubemap'.")
        if pred_batch.ndim != 5:
            raise RuntimeError(f"Invalid cubemap tensor shape {tuple(pred_batch.shape)}; expected [B,6,C,H,W].")

        input_hw = tuple(pager_cubemap.get("input_hw", (pred_batch.shape[-2], pred_batch.shape[-1])))
        target_hw = tuple(pager_cubemap.get("target_hw", input_hw))
        B = int(pred_batch.shape[0])
        device = pred_batch.device
        offload_device = mm.unet_offload_device()
        if device.type == "cpu":
            pref = mm.get_torch_device()
            if pager_model.get("force_gpu", False) and hasattr(torch, "cuda") and torch.cuda.is_available():
                pref = torch.device("cuda")
            if pref.type != "cpu":
                pred_batch = pred_batch.to(pref)
                device = pref

        dtype = pager_model.get("dtype", torch.float32)
        _pager_to(pipe, device, dtype)

        print(
            f"[PaGeR Normal Postprocess] device={device} cubemap_shape={tuple(pred_batch.shape)} "
            f"target_hw={target_hw} input_hw={input_hw} match_input_resolution={match_input_resolution}"
        )

        normal_out = []
        pbar = ProgressBar(B)

        try:
            with torch.inference_mode():
                for i in range(B):
                    print(f"[PaGeR Normal Postprocess] Processing cubemap {i + 1}/{B}")
                    t0 = time.perf_counter()
                    pred_normal = pipe.process_normal_output(pred_batch[i].float(), orig_size=target_hw)
                    t1 = time.perf_counter()

                    pred_normal = pred_normal.float()
                    n_min = pred_normal.amin()
                    n_max = pred_normal.amax()
                    pred_normal_vis = (pred_normal - n_min) / (n_max - n_min + 1e-8)
                    normal_img = _normal_to_hwc(pred_normal_vis).clamp(0.0, 1.0)

                    if match_input_resolution and target_hw != input_hw:
                        normal_img = _resize_hwc(normal_img, input_hw, mode="bilinear")

                    normal_out.append(normal_img)
                    pbar.update(1)
                    t2 = time.perf_counter()
                    print(
                        f"[PaGeR Normal Postprocess] timings: erp={t1 - t0:.3f}s tail={t2 - t1:.3f}s total={t2 - t0:.3f}s"
                    )
        finally:
            if pager_model.get("keep_on_gpu", True):
                _pager_to(pipe, device, dtype)
            else:
                _pager_to(pipe, offload_device, dtype)
            mm.soft_empty_cache()

        normal = torch.cat(normal_out, dim=0).clamp(0.0, 1.0).float()

        if mask is not None:
            keep = _mask_to_keep(mask.detach().float(), (normal.shape[1], normal.shape[2]), normal.shape[0]).to(normal.device)
            keep = keep.unsqueeze(-1)
            normal = normal * keep

        print(f"[PaGeR Normal Postprocess] Done. normal_shape={tuple(normal.shape)}")
        return (normal,)


class PaGeRDepth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pager_model": ("PAGERMODEL",),
                "images": ("IMAGE",),
                "erp_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 1}),
                "face_w": ("INT", {"default": 768, "min": 32, "max": 2048, "step": 1}),
                "match_input_resolution": ("BOOLEAN", {"default": True}),
                "min_depth": ("FLOAT", {"default": DEFAULT_MIN_DEPTH, "min": -100.0, "max": 100.0, "step": 0.001, "advanced": True}),
                "depth_range": ("FLOAT", {"default": DEFAULT_DEPTH_RANGE, "min": 1e-6, "max": 1000.0, "step": 0.001, "advanced": True}),
                "edge_abs_thresh": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "edge_rel_thresh": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 1.0, "step": 0.0001}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("depth", "edge_mask", "depth_color")
    FUNCTION = "process"
    CATEGORY = "PaGeR"

    DESCRIPTION = (
        "Runs PaGeR depth on ERP panoramas and returns normalized depth + edge mask + "
        "HF-style Spectral colorized depth visualization."
    )

    def process(
        self,
        pager_model,
        images: torch.Tensor,
        erp_height: int = 1024,
        face_w: int = 768,
        match_input_resolution: bool = True,
        min_depth: float = DEFAULT_MIN_DEPTH,
        depth_range: float = DEFAULT_DEPTH_RANGE,
        edge_abs_thresh: float = 0.002,
        edge_rel_thresh: float = 0.002,
        mask: torch.Tensor | None = None,
    ):
        print("[PaGeR Depth] Starting.")
        _require_module("pytorch360convert", "pip install -r custom_nodes/Comfyui-marigold-intrinsics/requirements.txt")
        _ensure_pager_paths()
        from src.utils.geometry_utils import compute_edge_mask, erp_to_cubemap

        pipe = pager_model.get("pipe", None) if isinstance(pager_model, dict) else None
        if pipe is None:
            raise RuntimeError("Invalid PAGERMODEL input: missing 'pipe'. Re-run the loader node.")
        if pager_model.get("modality") != "depth":
            raise RuntimeError("This node expects a *depth* PaGeR model. Use `Load PaGeR Model` with a depth repo.")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        if pager_model.get("force_gpu", False) and hasattr(torch, "cuda") and torch.cuda.is_available():
            device = torch.device("cuda")

        dtype = pager_model.get("dtype", torch.float32)
        autocast_dtype = pager_model.get("autocast_dtype", dtype)
        autocast_condition = (autocast_dtype != torch.float32) and not mm.is_device_mps(device)

        print(
            f"[PaGeR Depth] device={device} dtype={dtype} autocast={'on' if autocast_condition else 'off'} "
            f"erp_height={erp_height} face_w={face_w}"
        )
        print(
            f"[PaGeR Depth] log_scale={pager_model.get('log_scale', False)} min_depth={float(min_depth):.6f} "
            f"depth_range={float(depth_range):.6f} edge_abs={float(edge_abs_thresh):.4f} edge_rel={float(edge_rel_thresh):.4f}"
        )

        _pager_to(pipe, device, dtype)

        images = images.to(dtype=torch.float32)
        if images.ndim != 4 or images.shape[-1] not in (1, 3):
            raise ValueError(f"`images` must be IMAGE [B,H,W,1|3], got shape={tuple(images.shape)}")
        if images.shape[-1] == 1:
            images = images.repeat(1, 1, 1, 3)

        B, H, W, _ = images.shape
        pbar = ProgressBar(B)
        erp_w = int(erp_height) * 2
        target_hw = (int(erp_height), int(erp_w))

        log_scale = bool(pager_model.get("log_scale", False))

        print(f"[PaGeR Depth] batch={B} input_hw={H}x{W} target_hw={target_hw[0]}x{target_hw[1]}")

        depth_out = []
        edge_out = []
        depth_color_out = []

        try:
            with torch.inference_mode():
                with torch.autocast(mm.get_autocast_device(device), dtype=autocast_dtype) if autocast_condition else torch.no_grad():
                    for i in range(B):
                        print(f"[PaGeR Depth] Processing image {i + 1}/{B}")
                        img = images[i]
                        t0 = time.perf_counter()
                        img_erp = _prepare_erp_image(img, target_hw, device)
                        t1 = time.perf_counter()
                        cubemap = erp_to_cubemap(img_erp, face_w=int(face_w)).unsqueeze(0).to(device)
                        t2 = time.perf_counter()
                        batch = {"rgb_cubemap": cubemap}

                        pred_cubemap = pipe(batch, "depth")
                        t3 = time.perf_counter()
                        pred_cubemap_f32 = pred_cubemap.float()
                        pred_metric, pred_viz = pipe.process_depth_output(
                            pred_cubemap_f32,
                            orig_size=target_hw,
                            min_depth=float(min_depth),
                            depth_range=float(depth_range),
                            log_scale=log_scale,
                        )
                        t4 = time.perf_counter()

                        if log_scale:
                            depth_norm = (pred_viz - float(min_depth)) / max(float(depth_range), 1e-8)
                        else:
                            depth_norm = (pred_metric - float(min_depth)) / max(float(depth_range), 1e-8)
                        depth_norm = depth_norm.clamp(0.0, 1.0)

                        depth_img = _depth_to_hwc(depth_norm)
                        depth_color_img = _depth_viz_to_color_hwc(pred_viz)
                        if match_input_resolution and target_hw != (H, W):
                            depth_img = _resize_hwc(depth_img, (H, W), mode="bilinear")
                            depth_color_img = _resize_hwc(depth_color_img, (H, W), mode="bilinear")

                        depth_metric_cpu = pred_metric.detach().cpu()
                        if depth_metric_cpu.ndim == 4 and depth_metric_cpu.shape[1] == 1:
                            depth_metric_cpu = depth_metric_cpu[0, 0]
                        elif depth_metric_cpu.ndim == 3:
                            depth_metric_cpu = depth_metric_cpu[0]
                        edge_mask_np = compute_edge_mask(
                            depth_metric_cpu.numpy(),
                            abs_thresh=float(edge_abs_thresh),
                            rel_thresh=float(edge_rel_thresh),
                        ).astype(np.float32)
                        edge_img = torch.from_numpy(edge_mask_np).unsqueeze(0).unsqueeze(-1)
                        if match_input_resolution and target_hw != (H, W):
                            edge_img = _resize_hwc(edge_img, (H, W), mode="nearest")

                        depth_out.append(depth_img.detach().cpu())
                        edge_out.append(edge_img.detach().cpu())
                        depth_color_out.append(depth_color_img.detach().cpu())
                        pbar.update(1)
                        t5 = time.perf_counter()
                        print(f"[PaGeR Depth] timings: prep={t1 - t0:.3f}s cubemap={t2 - t1:.3f}s unet={t3 - t2:.3f}s "
                              f"erp_post={t4 - t3:.3f}s tail={t5 - t4:.3f}s total={t5 - t0:.3f}s")
        finally:
            if pager_model.get("keep_on_gpu", True):
                _pager_to(pipe, device, dtype)
            else:
                _pager_to(pipe, offload_device, dtype)
            mm.soft_empty_cache()

        depth = torch.cat(depth_out, dim=0).clamp(0.0, 1.0).cpu().float()
        edge = torch.cat(edge_out, dim=0).clamp(0.0, 1.0).cpu().float()
        depth_color = torch.cat(depth_color_out, dim=0).clamp(0.0, 1.0).cpu().float()

        if mask is not None:
            keep = _mask_to_keep(mask.detach().cpu().float(), (depth.shape[1], depth.shape[2]), depth.shape[0]).unsqueeze(-1)
            depth = depth * keep
            edge = edge * keep
            depth_color = depth_color * keep

        depth = depth.repeat(1, 1, 1, 3)
        edge = edge.repeat(1, 1, 1, 3)

        print(
            f"[PaGeR Depth] Done. depth_shape={tuple(depth.shape)} edge_shape={tuple(edge.shape)} "
            f"depth_color_shape={tuple(depth_color.shape)}"
        )
        return (depth, edge, depth_color)


class PaGeRSavePointCloud:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": ("IMAGE",),
                "depth": ("IMAGE",),
                "file_format": (["glb", "ply"], {"default": "glb"}),
                "filename_prefix": ("STRING", {"default": "PaGeR/pointcloud"}),
                "depth_mode": (
                    ["auto_from_pager_model", "auto_guess", "normalized_log", "normalized_linear", "metric"],
                    {"default": "auto_from_pager_model"},
                ),
                "min_depth": (
                    "FLOAT",
                    {"default": DEFAULT_MIN_DEPTH, "min": -100.0, "max": 100.0, "step": 0.001, "advanced": True},
                ),
                "depth_range": (
                    "FLOAT",
                    {"default": DEFAULT_DEPTH_RANGE, "min": 1e-6, "max": 1000.0, "step": 0.001, "advanced": True},
                ),
                "downsample_factor": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "max_points": ("INT", {"default": 200000, "min": 1000, "max": 5000000, "step": 1000}),
                "edge_filter": ("BOOLEAN", {"default": False}),
                "edge_abs_thresh": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "edge_rel_thresh": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 1.0, "step": 0.0001}),
            },
            "optional": {
                "pager_model": ("PAGERMODEL",),
                "mask": ("MASK",),
                "edge_mask": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_paths",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "PaGeR"

    DESCRIPTION = (
        "Export ERP depth + color to point cloud files (.glb or .ply). "
        "Depth can be auto-resolved from a PaGeR model config; normals can be used as point colors."
    )

    def save(
        self,
        color: torch.Tensor,
        depth: torch.Tensor,
        file_format: str = "glb",
        filename_prefix: str = "PaGeR/pointcloud",
        depth_mode: str = "auto_from_pager_model",
        min_depth: float = DEFAULT_MIN_DEPTH,
        depth_range: float = DEFAULT_DEPTH_RANGE,
        downsample_factor: int = 2,
        max_points: int = 200000,
        edge_filter: bool = False,
        edge_abs_thresh: float = 0.002,
        edge_rel_thresh: float = 0.002,
        pager_model=None,
        mask: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ):
        _ensure_pager_paths()
        _require_module("trimesh", "pip install -r custom_nodes/Comfyui-marigold-intrinsics/requirements.txt")
        from src.utils.geometry_utils import compute_edge_mask, erp_to_pointcloud
        import trimesh

        if color.ndim != 4 or color.shape[-1] not in (1, 3):
            raise ValueError(f"`color` must be IMAGE [B,H,W,1|3], got shape={tuple(color.shape)}")
        if depth.ndim != 4 or depth.shape[-1] not in (1, 3):
            raise ValueError(f"`depth` must be IMAGE [B,H,W,1|3], got shape={tuple(depth.shape)}")

        color = color.to(dtype=torch.float32)
        depth = depth.to(dtype=torch.float32)
        if color.shape[-1] == 1:
            color = color.repeat(1, 1, 1, 3)

        b_color = int(color.shape[0])
        b_depth = int(depth.shape[0])
        if b_color != b_depth:
            if b_color == 1 and b_depth > 1:
                color = color.repeat(b_depth, 1, 1, 1)
            elif b_depth == 1 and b_color > 1:
                depth = depth.repeat(b_color, 1, 1, 1)
            else:
                raise ValueError(f"Batch mismatch for color/depth: color={b_color}, depth={b_depth}")

        target_hw = (int(color.shape[1]), int(color.shape[2]))
        if depth.shape[1:3] != target_hw:
            depth = _resize_hwc(depth, target_hw, mode="bilinear")

        resolved_depth_mode = str(depth_mode)
        if resolved_depth_mode == "auto_from_pager_model":
            if isinstance(pager_model, dict) and ("log_scale" in pager_model):
                resolved_depth_mode = "normalized_log" if bool(pager_model.get("log_scale", False)) else "normalized_linear"
                print(
                    f"[PaGeR PointCloud] depth_mode auto from pager_model: {resolved_depth_mode} "
                    f"(log_scale={bool(pager_model.get('log_scale', False))}, "
                    f"metric_depth={bool(pager_model.get('metric_depth', False))})"
                )
            else:
                resolved_depth_mode = "auto_guess"
                print("[PaGeR PointCloud] depth_mode auto_from_pager_model requested, but pager_model not connected; using auto_guess.")

        warned_metric_normalized = False

        mask_bhw = None
        if mask is not None:
            mask_bhw = mask.detach().float()
            mask_bhw = _broadcast_bhw(mask_bhw, int(color.shape[0]))
            if mask_bhw.shape[1:3] != target_hw:
                mask_bhw = _resize_mask_bhw(mask_bhw, target_hw)

        edge_mask_bhw = None
        if edge_mask is not None:
            if edge_mask.ndim != 4 or edge_mask.shape[-1] not in (1, 3):
                raise ValueError(f"`edge_mask` must be IMAGE [B,H,W,1|3], got shape={tuple(edge_mask.shape)}")
            edge_mask = edge_mask.detach().float()
            if int(edge_mask.shape[0]) != int(color.shape[0]):
                if int(edge_mask.shape[0]) == 1 and int(color.shape[0]) > 1:
                    edge_mask = edge_mask.repeat(int(color.shape[0]), 1, 1, 1)
                else:
                    raise ValueError(
                        f"Batch mismatch for edge_mask/color: edge_mask={int(edge_mask.shape[0])}, "
                        f"color={int(color.shape[0])}"
                    )
            if edge_mask.shape[1:3] != target_hw:
                edge_mask = _resize_hwc(edge_mask, target_hw, mode="nearest")
            edge_mask_bhw = edge_mask[..., 0]

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, _subfolder, _resolved_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            output_dir,
            target_hw[1],
            target_hw[0],
        )

        saved_paths: list[str] = []
        for i in range(int(color.shape[0])):
            color_np = color[i].detach().cpu().numpy()
            depth_np = depth[i].detach().cpu().numpy()
            if depth_np.ndim == 3 and depth_np.shape[-1] == 3:
                c0 = depth_np[..., 0]
                channel_delta = max(
                    float(np.max(np.abs(c0 - depth_np[..., 1]))),
                    float(np.max(np.abs(c0 - depth_np[..., 2]))),
                )
                if channel_delta > 1e-3:
                    raise RuntimeError(
                        "Depth input appears colorized (RGB channels differ). "
                        "Use the grayscale `depth` output, not `depth_color`."
                    )
                depth_np = c0
            elif depth_np.ndim == 3:
                depth_np = depth_np[..., 0]

            mode_i = resolved_depth_mode
            if mode_i == "auto_guess":
                dmin = float(np.nanmin(depth_np))
                dmax = float(np.nanmax(depth_np))
                if -1e-5 <= dmin and dmax <= 1.0 + 1e-5:
                    if isinstance(pager_model, dict):
                        mode_i = "normalized_log" if bool(pager_model.get("log_scale", False)) else "normalized_linear"
                    else:
                        mode_i = "normalized_log"
                else:
                    mode_i = "metric"
                print(f"[PaGeR PointCloud] depth_mode auto_guess image {i + 1}: {mode_i} (min={dmin:.6f}, max={dmax:.6f})")

            if mode_i == "normalized_log":
                depth_norm = np.clip(depth_np, 0.0, 1.0)
                depth_metric = np.exp(depth_norm * float(depth_range) + float(min_depth))
            elif mode_i == "normalized_linear":
                depth_norm = np.clip(depth_np, 0.0, 1.0)
                depth_metric = depth_norm * float(depth_range) + float(min_depth)
            elif mode_i == "metric":
                if not warned_metric_normalized:
                    dmin = float(np.nanmin(depth_np))
                    dmax = float(np.nanmax(depth_np))
                    if -1e-5 <= dmin and dmax <= 1.1:
                        print(
                            "[PaGeR PointCloud] Warning: depth_mode='metric' but depth looks normalized in [0,1]. "
                            "If output looks spherical, use auto_from_pager_model or normalized_* mode."
                        )
                        warned_metric_normalized = True
                depth_metric = depth_np
            else:
                raise ValueError(f"Unsupported depth_mode={mode_i!r}")
            depth_metric = depth_metric.astype(np.float32, copy=False)

            extra_mask = None
            if mask_bhw is not None:
                extra_mask = (mask_bhw[i].detach().cpu().numpy() > 0.5)
            if edge_mask_bhw is not None:
                edge_keep = (edge_mask_bhw[i].detach().cpu().numpy() > 0.5)
                extra_mask = edge_keep if extra_mask is None else (extra_mask & edge_keep)

            keep = None
            if bool(edge_filter):
                keep = compute_edge_mask(
                    depth_metric,
                    abs_thresh=float(edge_abs_thresh),
                    rel_thresh=float(edge_rel_thresh),
                )
            if extra_mask is not None:
                keep = extra_mask if keep is None else (keep & extra_mask)

            ds = max(1, int(downsample_factor))
            if ds > 1:
                color_np = color_np[::ds, ::ds, :]
                depth_metric = depth_metric[::ds, ::ds]
                if keep is not None:
                    keep = keep[::ds, ::ds]

            # Match HF-space convention: geometry helper expects rgb in [-1, 1].
            color_signed = np.clip(color_np * 2.0 - 1.0, -1.0, 1.0).astype(np.float32, copy=False)
            points, colors = erp_to_pointcloud(color_signed, depth_metric, keep)
            if points.shape[0] == 0:
                raise RuntimeError(
                    "Point cloud export produced zero valid points. "
                    "Check depth values, mask, and edge-filter settings."
                )

            limit = max(1, int(max_points))
            if points.shape[0] > limit:
                idx = np.linspace(0, points.shape[0] - 1, num=limit, dtype=np.int64)
                points = points[idx]
                colors = colors[idx]

            pointcloud = trimesh.PointCloud(vertices=points, colors=colors)
            filename_with_batch_num = filename.replace("%batch_num%", str(i))
            out_name = f"{filename_with_batch_num}_{counter:05}_.{file_format}"
            out_path = os.path.join(full_output_folder, out_name)

            if file_format == "glb":
                scene = trimesh.Scene()
                scene.add_geometry(pointcloud)
                scene.export(out_path)
            else:
                pointcloud.export(out_path)

            saved_paths.append(out_path)
            counter += 1

        print(f"[PaGeR PointCloud] Saved {len(saved_paths)} file(s) as {file_format.upper()}.")
        return ("\n".join(saved_paths),)


class PaGeRNormal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pager_model": ("PAGERMODEL",),
                "images": ("IMAGE",),
                "erp_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 1}),
                "face_w": ("INT", {"default": 768, "min": 32, "max": 2048, "step": 1}),
                "match_input_resolution": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal",)
    FUNCTION = "process"
    CATEGORY = "PaGeR"

    DESCRIPTION = "Runs PaGeR surface normals on ERP panoramas."

    def process(
        self,
        pager_model,
        images: torch.Tensor,
        erp_height: int = 1024,
        face_w: int = 768,
        match_input_resolution: bool = True,
        mask: torch.Tensor | None = None,
    ):
        print("[PaGeR Normal] Starting.")
        _require_module("pytorch360convert", "pip install -r custom_nodes/Comfyui-marigold-intrinsics/requirements.txt")
        _ensure_pager_paths()
        from src.utils.geometry_utils import erp_to_cubemap

        pipe = pager_model.get("pipe", None) if isinstance(pager_model, dict) else None
        if pipe is None:
            raise RuntimeError("Invalid PAGERMODEL input: missing 'pipe'. Re-run the loader node.")
        if pager_model.get("modality") != "normal":
            raise RuntimeError("This node expects a *normal* PaGeR model. Use `Load PaGeR Model` with a normals repo.")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        if pager_model.get("force_gpu", False) and hasattr(torch, "cuda") and torch.cuda.is_available():
            device = torch.device("cuda")

        dtype = pager_model.get("dtype", torch.float32)
        autocast_dtype = pager_model.get("autocast_dtype", dtype)
        autocast_condition = (autocast_dtype != torch.float32) and not mm.is_device_mps(device)

        print(
            f"[PaGeR Normal] device={device} dtype={dtype} autocast={'on' if autocast_condition else 'off'} "
            f"erp_height={erp_height} face_w={face_w}"
        )

        _pager_to(pipe, device, dtype)

        images = images.to(dtype=torch.float32)
        if images.ndim != 4 or images.shape[-1] not in (1, 3):
            raise ValueError(f"`images` must be IMAGE [B,H,W,1|3], got shape={tuple(images.shape)}")
        if images.shape[-1] == 1:
            images = images.repeat(1, 1, 1, 3)

        B, H, W, _ = images.shape
        pbar = ProgressBar(B)
        erp_w = int(erp_height) * 2
        target_hw = (int(erp_height), int(erp_w))

        print(f"[PaGeR Normal] batch={B} input_hw={H}x{W} target_hw={target_hw[0]}x{target_hw[1]}")

        normal_out = []

        try:
            with torch.inference_mode():
                with torch.autocast(mm.get_autocast_device(device), dtype=autocast_dtype) if autocast_condition else torch.no_grad():
                    for i in range(B):
                        print(f"[PaGeR Normal] Processing image {i + 1}/{B}")
                        img = images[i]
                        t0 = time.perf_counter()
                        img_erp = _prepare_erp_image(img, target_hw, device)
                        t1 = time.perf_counter()
                        cubemap = erp_to_cubemap(img_erp, face_w=int(face_w)).unsqueeze(0).to(device)
                        t2 = time.perf_counter()
                        batch = {"rgb_cubemap": cubemap}

                        pred_cubemap = pipe(batch, "normal")
                        t3 = time.perf_counter()
                        pred_cubemap_f32 = pred_cubemap.float()
                        pred_normal = pipe.process_normal_output(pred_cubemap_f32, orig_size=target_hw)
                        t4 = time.perf_counter()
                        # HF Space uses prepare_image_for_logging: global min/max normalization.
                        pred_normal = pred_normal.float()
                        n_min = pred_normal.amin()
                        n_max = pred_normal.amax()
                        pred_normal_vis = (pred_normal - n_min) / (n_max - n_min + 1e-8)
                        normal_img = _normal_to_hwc(pred_normal_vis).clamp(0.0, 1.0)

                        if match_input_resolution and target_hw != (H, W):
                            normal_img = _resize_hwc(normal_img, (H, W), mode="bilinear")

                        normal_out.append(normal_img.detach().cpu())
                        pbar.update(1)
                        t5 = time.perf_counter()
                        print(f"[PaGeR Normal] timings: prep={t1 - t0:.3f}s cubemap={t2 - t1:.3f}s unet={t3 - t2:.3f}s "
                              f"erp_post={t4 - t3:.3f}s tail={t5 - t4:.3f}s total={t5 - t0:.3f}s")
        finally:
            if pager_model.get("keep_on_gpu", True):
                _pager_to(pipe, device, dtype)
            else:
                _pager_to(pipe, offload_device, dtype)
            mm.soft_empty_cache()

        normal = torch.cat(normal_out, dim=0).clamp(0.0, 1.0).cpu().float()

        if mask is not None:
            keep = _mask_to_keep(mask.detach().cpu().float(), (normal.shape[1], normal.shape[2]), normal.shape[0]).unsqueeze(-1)
            normal = normal * keep

        print(f"[PaGeR Normal] Done. normal_shape={tuple(normal.shape)}")
        return (normal,)
