import os
import shutil
import glob

import torch
import torch.nn.functional as F

import folder_paths

MODEL_REPO_ID_APPEARANCE = "prs-eth/marigold-iid-appearance-v1-1"
MODEL_REPO_ID_LIGHTING = "prs-eth/marigold-iid-lighting-v1-1"
STORAGE_DIRNAME = "marigold_intrinsics"
EMPTY_PROMPT_FILENAME = "empty_prompt.safetensors"
EMPTY_PROMPT_FILENAME_MARIGOLD = "empty_prompt_marigold.safetensors"
EMPTY_PROMPT_FILENAME_PAGER = "empty_prompt_pager.safetensors"
AUTO_CONDITIONING_FILENAME = "auto_by_mode"


def _select_torch_dtype(precision: str, device: torch.device) -> torch.dtype:
    precision = (precision or "auto").lower()
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp32":
        return torch.float32

    # auto
    if hasattr(device, "type") and device.type == "cpu":
        return torch.float32
    if hasattr(device, "type") and device.type == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    if hasattr(device, "type") and device.type == "mps":
        return torch.float16
    return torch.float16


def _requirements_install_hint() -> str:
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    return f'pip install -r "{req_path}"'


def _require_module(module_name: str, install_hint: str | None = None) -> None:
    try:
        __import__(module_name)
    except Exception as e:
        hint = (install_hint or "").strip()
        if not hint:
            hint = _requirements_install_hint()
        # Normalize any requirements-file hint to this extension's local path.
        if "pip install -r" in hint and "requirements.txt" in hint:
            hint = _requirements_install_hint()
        raise RuntimeError(f"Missing dependency '{module_name}'. Install with: {hint}") from e


def _storage_root() -> str:
    return os.path.join(folder_paths.models_dir, STORAGE_DIRNAME)


def _hf_cache_dir() -> str:
    # We use hf_hub_download for individual files; keep its cache under ComfyUI/models too.
    return os.path.join(_storage_root(), "_hf_cache")


def _cleanup_hf_cache() -> None:
    cache_dir = _hf_cache_dir()
    if not os.path.isdir(cache_dir):
        return
    try:
        shutil.rmtree(cache_dir)
    except Exception as e:
        print(f"[Marigold IID] Warning: could not remove HF cache dir '{cache_dir}': {e}")


def _link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _legacy_repo_dir(repo_id: str) -> str | None:
    # Support migration from the older folder layout used by earlier versions of this node.
    if repo_id == MODEL_REPO_ID_APPEARANCE:
        return os.path.join(folder_paths.models_dir, "marigold_iid_appearance", "marigold-iid-appearance-v1-1")
    if repo_id == MODEL_REPO_ID_LIGHTING:
        return os.path.join(folder_paths.models_dir, "marigold_iid_lighting", "marigold-iid-lighting-v1-1")
    return None


def _ensure_file(repo_id: str, remote_path: str, local_path: str) -> None:
    if os.path.exists(local_path):
        return

    legacy_dir = _legacy_repo_dir(repo_id)
    if legacy_dir is not None:
        legacy_path = os.path.join(legacy_dir, remote_path)
        if os.path.exists(legacy_path):
            _link_or_copy(legacy_path, local_path)
            return

    _require_module("huggingface_hub", "pip install -r custom_nodes/Comfyui-marigold-intrinsics/requirements.txt")
    from huggingface_hub import hf_hub_download

    src_path = hf_hub_download(
        repo_id=repo_id,
        filename=remote_path,
        cache_dir=_hf_cache_dir(),
    )
    _link_or_copy(src_path, local_path)


def _base_dir(model_variant: str) -> str:
    return os.path.join(_storage_root(), "base", model_variant)


def _default_conditioning_filename_for_mode(mode: str) -> str:
    if mode == "pager_max_length":
        return EMPTY_PROMPT_FILENAME_PAGER
    if mode == "marigold_pipeline":
        return EMPTY_PROMPT_FILENAME_MARIGOLD
    return EMPTY_PROMPT_FILENAME


def _conditioning_filename_aliases(filename: str) -> list[str]:
    names = []

    def _add(v: str) -> None:
        if v and v not in names:
            names.append(v)

    _add(filename)
    if filename.endswith(".safetensor"):
        _add(f"{filename}s")
    elif filename.endswith(".safetensors"):
        _add(filename[:-1])
    elif "." not in os.path.basename(filename):
        _add(f"{filename}.safetensors")
        _add(f"{filename}.safetensor")

    return names


def _conditioning_filename_choices(
    preferred: list[str] | None = None,
    include_auto_by_mode: bool = False,
) -> list[str]:
    choices: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        name = (name or "").strip()
        if not name or name in seen:
            return
        seen.add(name)
        choices.append(name)

    if include_auto_by_mode:
        _add(AUTO_CONDITIONING_FILENAME)

    for name in preferred or []:
        _add(name)
        for alias in _conditioning_filename_aliases(name):
            _add(alias)

    for variant in ("fp16", "fp32"):
        base = _base_dir(variant)
        if not os.path.isdir(base):
            continue
        for pattern in ("*.safetensor", "*.safetensors"):
            for path in sorted(glob.glob(os.path.join(base, pattern))):
                _add(os.path.basename(path))

    if not choices:
        if include_auto_by_mode:
            _add(AUTO_CONDITIONING_FILENAME)
        else:
            for name in (
                EMPTY_PROMPT_FILENAME_MARIGOLD,
                EMPTY_PROMPT_FILENAME_PAGER,
                EMPTY_PROMPT_FILENAME,
            ):
                _add(name)

    return choices


def _model_dir(kind: str, model_variant: str) -> str:
    # kind: "appearance" | "lighting"
    return os.path.join(_storage_root(), kind, model_variant)


def _empty_prompt_path(base_dir: str, filename: str = EMPTY_PROMPT_FILENAME) -> str:
    return os.path.join(base_dir, filename)


def _ensure_base_files(model_variant: str, repo_id_source: str) -> str:
    if model_variant not in ("fp16", "fp32"):
        raise ValueError(f"Invalid model_variant={model_variant!r}")

    base = _base_dir(model_variant)

    common = [
        ("scheduler/scheduler_config.json", "scheduler/scheduler_config.json"),
        ("tokenizer/merges.txt", "tokenizer/merges.txt"),
        ("tokenizer/special_tokens_map.json", "tokenizer/special_tokens_map.json"),
        ("tokenizer/tokenizer_config.json", "tokenizer/tokenizer_config.json"),
        ("tokenizer/vocab.json", "tokenizer/vocab.json"),
        ("text_encoder/config.json", "text_encoder/config.json"),
        ("vae/config.json", "vae/config.json"),
    ]

    if model_variant == "fp16":
        weights = [
            ("text_encoder/model.fp16.safetensors", "text_encoder/model.safetensors"),
            ("vae/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.safetensors"),
        ]
    else:
        weights = [
            ("text_encoder/model.safetensors", "text_encoder/model.safetensors"),
            ("vae/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"),
        ]

    for remote_path, rel_dst in common + weights:
        _ensure_file(repo_id_source, remote_path, os.path.join(base, rel_dst))

    return base


def _ensure_model_files(repo_id: str, kind: str, model_variant: str) -> str:
    if model_variant not in ("fp16", "fp32"):
        raise ValueError(f"Invalid model_variant={model_variant!r}")

    root = _model_dir(kind, model_variant)

    files = [
        ("model_index.json", "model_index.json"),
        ("unet/config.json", "unet/config.json"),
    ]

    if model_variant == "fp16":
        files.append(("unet/diffusion_pytorch_model.fp16.safetensors", "unet/diffusion_pytorch_model.safetensors"))
    else:
        files.append(("unet/diffusion_pytorch_model.safetensors", "unet/diffusion_pytorch_model.safetensors"))

    for remote_path, rel_dst in files:
        _ensure_file(repo_id, remote_path, os.path.join(root, rel_dst))

    return root


def _load_precomputed_conditioning(
    base_dir: str,
    dtype: torch.dtype,
    device: torch.device,
    filename: str = EMPTY_PROMPT_FILENAME,
) -> torch.Tensor | None:
    path = None
    for candidate in _conditioning_filename_aliases(filename):
        candidate_path = _empty_prompt_path(base_dir, candidate)
        if os.path.exists(candidate_path):
            path = candidate_path
            break
    if path is None:
        return None

    _require_module("safetensors", "pip install safetensors")
    from safetensors.torch import load_file

    data = load_file(path, device="cpu")
    if not data:
        raise RuntimeError(f"Conditioning file exists but contains no tensors: {path}")

    tensor = data.get("empty_prompt_embedding", None)
    if tensor is None:
        # Backward compatibility with unknown key names.
        tensor = next(iter(data.values()))
    if tensor.ndim != 3:
        raise RuntimeError(f"Invalid conditioning tensor shape {tuple(tensor.shape)} in {path}; expected [1,T,C].")

    return tensor.to(device=device, dtype=dtype)


def _save_precomputed_conditioning(
    base_dir: str,
    embedding: torch.Tensor,
    filename: str = EMPTY_PROMPT_FILENAME,
    overwrite: bool = True,
    metadata: dict[str, str] | None = None,
) -> str:
    _require_module("safetensors", "pip install safetensors")
    from safetensors.torch import save_file

    path = _empty_prompt_path(base_dir, filename)
    os.makedirs(base_dir, exist_ok=True)
    if os.path.exists(path) and not overwrite:
        raise RuntimeError(f"Conditioning file already exists and overwrite=False: {path}")

    emb = embedding.detach().cpu().contiguous()
    if emb.ndim != 3:
        raise RuntimeError(f"Invalid conditioning tensor shape {tuple(emb.shape)}; expected [1,T,C].")

    save_file({"empty_prompt_embedding": emb}, path, metadata=metadata or {})
    return path


def _prepare_mask(mask: torch.Tensor, target_hw: tuple[int, int], target_b: int) -> torch.Tensor:
    """
    Prepare a MASK tensor to match [B,H,W] with values in [0,1].
    - Supports [H,W], [B,H,W], or [B,H,W,1]
    - Broadcasts batch if mask B=1
    - Resizes to target_hw using bilinear interpolation
    """
    if mask is None:
        raise ValueError("mask is None")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 3:
        raise ValueError(f"`mask` must be [B,H,W] (or [H,W]/[B,H,W,1]), got shape={tuple(mask.shape)}")

    b = int(mask.shape[0])
    if b != target_b:
        if b == 1 and target_b > 1:
            mask = mask.repeat(target_b, 1, 1)
        else:
            raise ValueError(f"Mask batch mismatch: mask B={b} vs target B={target_b}")

    if mask.shape[1:3] != target_hw:
        mask_nchw = mask.unsqueeze(1)
        mask_nchw = F.interpolate(mask_nchw, size=target_hw, mode="bilinear", align_corners=False)
        mask = mask_nchw.squeeze(1)

    return mask.clamp(0.0, 1.0)


def _mask_to_keep(mask: torch.Tensor, target_hw: tuple[int, int], target_b: int) -> torch.Tensor:
    """
    Convert a MASK to a keep-map in [0,1] where 1 keeps pixels.

    Heuristic:
    - If the mask is mostly >0.5, treat it as ComfyUI-style mask (1 = transparent) and invert.
    - Otherwise treat it as alpha (1 = opaque) and keep as-is.
    """
    mask_p = _prepare_mask(mask, target_hw, target_b)
    # Heuristic inversion to handle different mask conventions.
    if mask_p.mean().item() > 0.5:
        keep = 1.0 - mask_p
    else:
        keep = mask_p
    return keep.clamp(0.0, 1.0)
