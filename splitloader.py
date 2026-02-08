import json
import math
import os
from types import SimpleNamespace

import torch

import comfy.model_management as mm
import comfy.sd
import comfy.utils
import folder_paths

from .nodes_shared import (
    MODEL_REPO_ID_APPEARANCE,
    MODEL_REPO_ID_LIGHTING,
    _cleanup_hf_cache,
    _ensure_base_runtime_files,
    _ensure_base_text_files,
    _ensure_model_files,
    _marigold_zero_empty_conditioning,
    _require_module,
    _select_torch_dtype,
)


_DIFFUSERS_WEIGHTS = [
    "diffusion_pytorch_model.fp16.safetensors",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.fp16.bin",
    "diffusion_pytorch_model.bin",
]

_TEXT_ENCODER_WEIGHTS = [
    "model.fp16.safetensors",
    "model.safetensors",
    "pytorch_model.fp16.bin",
    "pytorch_model.bin",
]

def _first_file(path: str, filenames: list[str]) -> str | None:
    for name in filenames:
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            return candidate
    return None


def _vae_list(video_taes: list[str]) -> list[str]:
    vaes = folder_paths.get_filename_list("vae")
    approx_vaes = folder_paths.get_filename_list("vae_approx")

    sdxl_taesd_enc = False
    sdxl_taesd_dec = False
    sd1_taesd_enc = False
    sd1_taesd_dec = False
    sd3_taesd_enc = False
    sd3_taesd_dec = False
    f1_taesd_enc = False
    f1_taesd_dec = False

    f2_has_safetensors = False
    f2_has_enc = False
    f2_has_dec = False

    for v in approx_vaes:
        if v.startswith("taesd_decoder."):
            sd1_taesd_dec = True
        elif v.startswith("taesd_encoder."):
            sd1_taesd_enc = True
        elif v.startswith("taesdxl_decoder."):
            sdxl_taesd_dec = True
        elif v.startswith("taesdxl_encoder."):
            sdxl_taesd_enc = True
        elif v.startswith("taesd3_decoder."):
            sd3_taesd_dec = True
        elif v.startswith("taesd3_encoder."):
            sd3_taesd_enc = True
        elif v.startswith("taef1_encoder."):
            f1_taesd_enc = True
        elif v.startswith("taef1_decoder."):
            f1_taesd_dec = True
        elif v == "taef2.safetensors":
            f2_has_safetensors = True
        elif v.startswith("taef2_encoder."):
            f2_has_enc = True
        elif v.startswith("taef2_decoder."):
            f2_has_dec = True
        else:
            for tae in video_taes:
                if v.startswith(tae):
                    vaes.append(v)

    if sd1_taesd_dec and sd1_taesd_enc:
        vaes.append("taesd")
    if sdxl_taesd_dec and sdxl_taesd_enc:
        vaes.append("taesdxl")
    if sd3_taesd_dec and sd3_taesd_enc:
        vaes.append("taesd3")
    if f1_taesd_dec and f1_taesd_enc:
        vaes.append("taef1")

    if f2_has_safetensors or (f2_has_enc and f2_has_dec):
        vaes.append("taef2")

    vaes.append("pixel_space")
    return vaes


def _load_taesd(name: str) -> dict:
    sd: dict = {}
    approx_vaes = folder_paths.get_filename_list("vae_approx")

    try:
        encoder = next(filter(lambda a: a.startswith(f"{name}_encoder."), approx_vaes))
        decoder = next(filter(lambda a: a.startswith(f"{name}_decoder."), approx_vaes))
    except StopIteration as e:
        raise RuntimeError(f"Could not find TAESD encoder/decoder for '{name}' in models/vae_approx") from e

    enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
    for k in enc:
        sd[f"taesd_encoder.{k}"] = enc[k]

    dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
    for k in dec:
        sd[f"taesd_decoder.{k}"] = dec[k]

    if name == "taesd":
        sd["vae_scale"] = torch.tensor(0.18215)
        sd["vae_shift"] = torch.tensor(0.0)
    elif name == "taesdxl":
        sd["vae_scale"] = torch.tensor(0.13025)
        sd["vae_shift"] = torch.tensor(0.0)
    elif name == "taesd3":
        sd["vae_scale"] = torch.tensor(1.5305)
        sd["vae_shift"] = torch.tensor(0.0609)
    elif name == "taef1":
        sd["vae_scale"] = torch.tensor(0.3611)
        sd["vae_shift"] = torch.tensor(0.1159)

    return sd


def _load_vae_by_name(vae_name: str, video_taes: list[str]) -> comfy.sd.VAE:
    metadata = None

    if vae_name == "pixel_space":
        sd = {"pixel_space_vae": torch.tensor(1.0)}
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    if vae_name == "taef2":
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        if "taef2.safetensors" in approx_vaes:
            from comfy.taesd.taef2 import ComfyTAEF2VAE

            path = folder_paths.get_full_path_or_raise("vae_approx", "taef2.safetensors")
            return ComfyTAEF2VAE(path)

        sd = _load_taesd("taef2")
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
        sd = _load_taesd(vae_name)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    if os.path.splitext(vae_name)[0] in video_taes:
        vae_path = folder_paths.get_full_path_or_raise("vae_approx", vae_name)
    else:
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)

    sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    return vae


def _attach_marigold_metadata(target, data: dict) -> None:
    if target is None:
        return
    attachments = getattr(target, "attachments", None)
    if isinstance(attachments, dict):
        attachments["marigold_iid"] = data


def _is_taesd_vae(vae: comfy.sd.VAE) -> bool:
    first_stage = getattr(vae, "first_stage_model", None)
    return hasattr(first_stage, "vae_scale") and hasattr(first_stage, "vae_shift")


def _vae_scaling_factor(vae: comfy.sd.VAE) -> float:
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


class _LatentDist:
    def __init__(self, latents: torch.Tensor):
        self._latents = latents

    def sample(self, generator=None):
        return self._latents

    def mode(self):
        return self._latents


class _VaeEncodeOutput:
    def __init__(self, latents: torch.Tensor):
        self.latent_dist = _LatentDist(latents)


class _VaeDecodeOutput:
    def __init__(self, sample: torch.Tensor):
        self.sample = sample


def _vae_downscale_ratio(vae: comfy.sd.VAE) -> int:
    ratio = getattr(vae, "downscale_ratio", 8)
    if isinstance(ratio, (tuple, list)):
        ratio_int = None
        for item in reversed(ratio):
            if isinstance(item, int):
                ratio_int = item
                break
        ratio = ratio_int if ratio_int is not None else 8
    if not isinstance(ratio, int):
        ratio = 8
    return max(1, ratio)


def _block_out_channels_from_ratio(ratio: int) -> list[int]:
    length = 1
    while 2 ** (length - 1) < ratio:
        length += 1
    return [1] * length


class _ComfyVAEAdapter(torch.nn.Module):
    def __init__(self, vae: comfy.sd.VAE, scaling_factor: float):
        super().__init__()
        self._vae = vae
        ratio = _vae_downscale_ratio(vae)
        latent_channels = getattr(vae, "latent_channels", 4)
        self.config = SimpleNamespace(
            scaling_factor=float(scaling_factor),
            block_out_channels=_block_out_channels_from_ratio(ratio),
            latent_channels=int(latent_channels),
        )
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        # diffusers VAE expects BCHW in [-1,1]; Comfy VAE expects BHWC in [0,1].
        pixels = ((x / 2.0) + 0.5).clamp(0.0, 1.0).movedim(1, -1)
        latents = self._vae.encode(pixels)
        if latents.device != x.device:
            latents = latents.to(x.device)
        if return_dict:
            return _VaeEncodeOutput(latents)
        return (latents,)

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        # Comfy VAE returns BHWC in [0,1]; convert back to BCHW in [-1,1].
        pixels = self._vae.decode(z)
        sample = pixels.movedim(-1, 1) * 2.0 - 1.0
        if sample.device != z.device:
            sample = sample.to(z.device)
        if return_dict:
            return _VaeDecodeOutput(sample)
        return (sample,)

    def to(self, device=None, dtype=None, non_blocking=False):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self


def _normalize_precision(value: str | None) -> str:
    if value is None:
        return "auto"
    if value in ("auto", "fp16", "bf16", "fp32"):
        return value
    if isinstance(value, str):
        v = value.lower()
        if "float16" in v:
            return "fp16"
        if "bfloat16" in v:
            return "bf16"
        if "float32" in v:
            return "fp32"
    return "auto"


class MarigoldIIDSplitLoader:
    video_taes = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5"]

    @classmethod
    def INPUT_TYPES(cls):
        vae_choices = ["marigold_base"] + _vae_list(cls.video_taes)
        tiny_choices = _vae_list(cls.video_taes)
        default_tiny = "taesd" if "taesd" in tiny_choices else tiny_choices[0]
        return {
            "required": {
                "model_kind": (["lighting", "appearance"], {"default": "lighting"}),
                "unet_variant": (["fp16", "fp32"], {"default": "fp16"}),
                "clip_variant": (["fp16", "fp32"], {"default": "fp16"}),
                "unet_precision": (["auto", "fp16", "bf16", "fp32"], {"default": "auto"}),
                "clip_precision": (["auto", "fp16", "bf16", "fp32"], {"default": "auto"}),
                "vae_name": (vae_choices, {"default": "marigold_base"}),
                "load_comfy_model_clip": ("BOOLEAN", {"default": False, "advanced": True}),
                "use_full_precision": ("BOOLEAN", {"default": False, "advanced": True}),
                "use_tiny_vae": ("BOOLEAN", {"default": False}),
                "tiny_vae_name": (tiny_choices, {"default": default_tiny}),
                "compile_unet": ("BOOLEAN", {"default": False, "advanced": True}),
                "compile_vae": ("BOOLEAN", {"default": False, "advanced": True}),
                "keep_on_gpu": ("BOOLEAN", {"default": True}),
                "force_gpu": ("BOOLEAN", {"default": True, "advanced": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"
    CATEGORY = "Marigold IID Loaders"

    DESCRIPTION = """
Loads the Marigold IID UNet as ComfyUI MODEL and optionally loads CLIP when enabled.
The VAE can be the Marigold base VAE or any TAESD/vae_approx selection from ComfyUI.
"""

    def load(
        self,
        model_kind: str,
        unet_variant: str,
        clip_variant: str,
        unet_precision: str,
        clip_precision: str,
        vae_name: str,
        load_comfy_model_clip: bool = False,
        use_full_precision: bool = False,
        use_tiny_vae: bool = False,
        tiny_vae_name: str = "taesd",
        compile_unet: bool = False,
        compile_vae: bool = False,
        keep_on_gpu: bool = True,
        force_gpu: bool = True,
    ):
        device = mm.get_torch_device()
        unet_dtype = _select_torch_dtype(unet_precision, device)
        clip_dtype = _select_torch_dtype(clip_precision, device)

        config = {
            "model_kind": model_kind,
            "unet_variant": unet_variant,
            "clip_variant": clip_variant,
            "unet_precision": str(unet_dtype),
            "clip_precision": str(clip_dtype),
            "vae_name": vae_name,
            "use_full_precision": bool(use_full_precision),
            "use_tiny_vae": bool(use_tiny_vae),
            "tiny_vae_name": tiny_vae_name,
            "compile_unet": bool(compile_unet),
            "compile_vae": bool(compile_vae),
            "keep_on_gpu": bool(keep_on_gpu),
            "force_gpu": bool(force_gpu),
        }
        if getattr(self, "_cache", None) is not None and self._cache.get("config") == config:
            return self._cache["value"]

        if model_kind == "lighting":
            repo_id = MODEL_REPO_ID_LIGHTING
        elif model_kind == "appearance":
            repo_id = MODEL_REPO_ID_APPEARANCE
        else:
            raise ValueError(f"Invalid model_kind={model_kind!r}")

        base_dir = _ensure_base_runtime_files(clip_variant, repo_id)
        model_dir = _ensure_model_files(repo_id, model_kind, unet_variant)

        unet_path = _first_file(os.path.join(model_dir, "unet"), _DIFFUSERS_WEIGHTS)
        if unet_path is None:
            raise RuntimeError(f"Could not find UNet weights under {os.path.join(model_dir, 'unet')}")

        if load_comfy_model_clip:
            _ensure_base_text_files(clip_variant, repo_id)
            text_encoder_path = _first_file(os.path.join(base_dir, "text_encoder"), _TEXT_ENCODER_WEIGHTS)
            if text_encoder_path is None:
                raise RuntimeError(f"Could not find text encoder weights under {os.path.join(base_dir, 'text_encoder')}")
            model_options = {"dtype": unet_dtype}
            try:
                model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
            except Exception as e:
                print(f"[Marigold IID] Warning: ComfyUI could not load UNet as MODEL ({e}). Using a stub for translation.")
                model = SimpleNamespace(attachments={})

            try:
                clip = comfy.sd.load_clip(
                    ckpt_paths=[text_encoder_path],
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    model_options={"dtype": clip_dtype},
                )
            except Exception as e:
                print(f"[Marigold IID] Warning: ComfyUI could not load CLIP as CLIP ({e}). Using a stub for translation.")
                clip = SimpleNamespace(attachments={})
        else:
            model = SimpleNamespace(attachments={})
            clip = SimpleNamespace(attachments={})

        effective_vae_name = vae_name
        if use_tiny_vae:
            effective_vae_name = tiny_vae_name

        if effective_vae_name == "marigold_base":
            vae_path = _first_file(os.path.join(base_dir, "vae"), _DIFFUSERS_WEIGHTS)
            if vae_path is None:
                raise RuntimeError(f"Could not find VAE weights under {os.path.join(base_dir, 'vae')}")
            sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
            vae = comfy.sd.VAE(sd=sd, metadata=metadata)
            vae.throw_exception_if_invalid()
        else:
            vae = _load_vae_by_name(effective_vae_name, self.video_taes)

        metadata = {
            "repo_id": repo_id,
            "model_kind": model_kind,
            "unet_variant": unet_variant,
            "clip_variant": clip_variant,
            "unet_precision": unet_precision,
            "clip_precision": clip_precision,
            "unet_dtype": str(unet_dtype),
            "clip_dtype": str(clip_dtype),
            "vae_name": effective_vae_name,
            "use_full_precision": bool(use_full_precision),
            "use_tiny_vae": bool(use_tiny_vae),
            "tiny_vae_name": tiny_vae_name,
            "compile_unet": bool(compile_unet),
            "compile_vae": bool(compile_vae),
            "keep_on_gpu": bool(keep_on_gpu),
            "force_gpu": bool(force_gpu),
        }
        _attach_marigold_metadata(model, metadata)
        if hasattr(clip, "patcher"):
            _attach_marigold_metadata(clip.patcher, metadata)

        _cleanup_hf_cache()

        result = (model, clip, vae)
        self._cache = {"config": config, "value": result}
        return result


class MarigoldIIDSplitToIIDModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("IIDMODEL",)
    RETURN_NAMES = ("iid_model",)
    FUNCTION = "translate"
    CATEGORY = "Marigold IID Loaders"

    DESCRIPTION = """
Translate split-loaded MODEL/CLIP/VAE into an IIDMODEL for Marigold IID inference nodes.
Supports TAESD/vae_approx VAEs via a lightweight adapter.
"""

    def translate(
        self,
        model,
        clip,
        vae,
    ):
        _require_module("diffusers", "pip install -r custom_nodes/Comfyui-marigold-intrinsics/requirements.txt")

        cfg = None
        if hasattr(model, "attachments"):
            cfg = model.attachments.get("marigold_iid")
        if cfg is None and hasattr(clip, "patcher") and hasattr(clip.patcher, "attachments"):
            cfg = clip.patcher.attachments.get("marigold_iid")
        if cfg is None:
            raise RuntimeError("Missing Marigold metadata. Use `Load Marigold IID Split (UNet/CLIP/VAE)` before this node.")

        use_full_precision = bool(cfg.get("use_full_precision", False))
        use_tiny_vae = bool(cfg.get("use_tiny_vae", False))
        tiny_vae_name = cfg.get("tiny_vae_name", "taesd")
        compile_unet = bool(cfg.get("compile_unet", False))
        compile_vae = bool(cfg.get("compile_vae", False))
        keep_on_gpu = bool(cfg.get("keep_on_gpu", True))
        force_gpu = bool(cfg.get("force_gpu", True))

        device = mm.get_torch_device()
        if force_gpu and hasattr(torch, "cuda") and torch.cuda.is_available():
            device = torch.device("cuda")

        model_kind = cfg.get("model_kind")
        repo_id = cfg.get("repo_id")
        unet_variant = cfg.get("unet_variant", "fp16")
        clip_variant = cfg.get("clip_variant", "fp16")

        unet_precision = _normalize_precision(cfg.get("unet_precision", "auto"))
        clip_precision = _normalize_precision(cfg.get("clip_precision", "auto"))
        unet_dtype = _select_torch_dtype(unet_precision, device)
        clip_dtype = _select_torch_dtype(clip_precision, device)

        if use_full_precision:
            unet_dtype = torch.float32
            clip_dtype = torch.float32

        if hasattr(device, "type") and device.type == "cpu":
            if unet_dtype in (torch.float16, torch.bfloat16):
                unet_dtype = torch.float32
            if clip_dtype in (torch.float16, torch.bfloat16):
                clip_dtype = torch.float32

        if model_kind == "lighting":
            repo_id = repo_id or MODEL_REPO_ID_LIGHTING
        elif model_kind == "appearance":
            repo_id = repo_id or MODEL_REPO_ID_APPEARANCE
        else:
            raise ValueError(f"Invalid model_kind={model_kind!r}")

        from diffusers import DDIMScheduler, MarigoldIntrinsicsPipeline, UNet2DConditionModel

        base_dir = _ensure_base_runtime_files(clip_variant, repo_id)
        model_dir = _ensure_model_files(repo_id, model_kind, unet_variant)

        with open(os.path.join(model_dir, "model_index.json"), "r", encoding="utf-8") as f:
            cfg_file = json.load(f)

        unet = UNet2DConditionModel.from_pretrained(
            model_dir,
            subfolder="unet",
            torch_dtype=unet_dtype,
            use_safetensors=True,
            local_files_only=True,
        )
        scheduler = DDIMScheduler.from_pretrained(
            base_dir,
            subfolder="scheduler",
            local_files_only=True,
        )

        if use_tiny_vae:
            vae = _load_vae_by_name(tiny_vae_name, MarigoldIIDSplitLoader.video_taes)

        scaling_factor = _vae_scaling_factor(vae)
        vae_adapter = _ComfyVAEAdapter(vae, scaling_factor=scaling_factor)
        vae_adapter.to(device=device, dtype=unet_dtype)

        pipe = MarigoldIntrinsicsPipeline(
            unet=unet,
            vae=vae_adapter,
            scheduler=scheduler,
            text_encoder=None,
            tokenizer=None,
            prediction_type=cfg_file.get("prediction_type"),
            target_properties=cfg_file.get("target_properties"),
            default_denoising_steps=cfg_file.get("default_denoising_steps"),
            default_processing_resolution=cfg_file.get("default_processing_resolution"),
        )
        pipe.empty_text_embedding = _marigold_zero_empty_conditioning(unet=unet, dtype=unet_dtype, device=device)

        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[Marigold IID] xFormers not enabled: {e}")

        pipe = pipe.to(device)

        if compile_vae:
            try:
                pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
            except Exception as e:
                print(f"[Marigold IID] Warning: torch.compile failed for VAE: {e}")
        if compile_unet:
            try:
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            except Exception as e:
                print(f"[Marigold IID] Warning: torch.compile failed for UNet: {e}")

        _cleanup_hf_cache()

        return (
            {
                "pipe": pipe,
                "dtype": unet_dtype,
                "autocast_dtype": unet_dtype,
                "clip_dtype": clip_dtype,
                "kind": model_kind,
                "repo_id": repo_id,
                "vae_is_taesd": _is_taesd_vae(vae),
                "keep_on_gpu": bool(keep_on_gpu),
            },
        )
