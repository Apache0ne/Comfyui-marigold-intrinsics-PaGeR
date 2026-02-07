import os

import torch

import comfy.model_management as mm
import folder_paths

from .nodes_shared import (
    AUTO_CONDITIONING_FILENAME,
    EMPTY_PROMPT_FILENAME,
    EMPTY_PROMPT_FILENAME_MARIGOLD,
    EMPTY_PROMPT_FILENAME_PAGER,
    STORAGE_DIRNAME,
    _conditioning_filename_choices,
    _default_conditioning_filename_for_mode,
    _load_precomputed_conditioning,
    _require_module,
    _save_precomputed_conditioning,
    _select_torch_dtype,
)


def _base_root() -> str:
    return os.path.join(folder_paths.models_dir, STORAGE_DIRNAME, "base")


def _base_dir(model_variant: str) -> str:
    return os.path.join(_base_root(), model_variant)


def _available_base_variants() -> list[str]:
    root = _base_root()
    variants = []
    for v in ("fp16", "fp32"):
        if os.path.isdir(_base_dir(v)):
            variants.append(v)
    if not variants:
        variants = ["fp16", "fp32"]
    return variants


def _require_local_base_variant(model_variant: str) -> str:
    base_dir = _base_dir(model_variant)
    if not os.path.isdir(base_dir):
        raise RuntimeError(
            f"Shared base folder does not exist: {base_dir}. "
            "Download/load a Marigold model first to populate shared base files."
        )
    return base_dir


class LoadMarigoldBaseTextEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        variants = _available_base_variants()
        default_variant = variants[0]
        return {
            "required": {
                "model_variant": (variants, {"default": default_variant}),
                "precision": (["auto", "fp16", "bf16", "fp32"], {"default": "auto"}),
                "force_gpu": ("BOOLEAN", {"default": False, "advanced": True}),
            }
        }

    RETURN_TYPES = ("BASE_TEXT_ENCODER",)
    RETURN_NAMES = ("base_text_encoder",)
    FUNCTION = "load"
    CATEGORY = "Marigold Conditioning"

    DESCRIPTION = "Loads tokenizer + CLIP text encoder from shared base/<variant> for generating empty-prompt conditioning."

    def load(self, model_variant: str, precision: str, force_gpu: bool = False):
        _require_module("transformers", "pip install -r custom_nodes/Comfyui-marigold-intrinsics/requirements.txt")
        from transformers import CLIPTextModel, CLIPTokenizer

        device = mm.get_torch_device()
        if force_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        dtype = _select_torch_dtype(precision, device)

        base_dir = _require_local_base_variant(model_variant)
        text_encoder_weight = os.path.join(base_dir, "text_encoder", "model.safetensors")
        if not os.path.exists(text_encoder_weight):
            text_encoder_weight = os.path.join(base_dir, "text_encoder", "pytorch_model.bin")
        print(f"[Marigold Conditioning] Loading tokenizer/text encoder from: {base_dir}")
        print(f"[Marigold Conditioning] Text encoder weights: {text_encoder_weight}")
        tokenizer = CLIPTokenizer.from_pretrained(base_dir, subfolder="tokenizer", local_files_only=True)
        text_encoder = CLIPTextModel.from_pretrained(
            base_dir,
            subfolder="text_encoder",
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=True,
        )
        text_encoder.to(device=device, dtype=dtype)
        text_encoder.eval()

        return (
            {
                "tokenizer": tokenizer,
                "text_encoder": text_encoder,
                "base_dir": base_dir,
                "dtype": dtype,
                "device": device,
                "model_variant": model_variant,
            },
        )


class SaveEmptyPromptConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        filename_choices = _conditioning_filename_choices(
            preferred=[EMPTY_PROMPT_FILENAME_MARIGOLD, EMPTY_PROMPT_FILENAME_PAGER, EMPTY_PROMPT_FILENAME],
            include_auto_by_mode=True,
        )
        return {
            "required": {
                "base_text_encoder": ("BASE_TEXT_ENCODER",),
                "conditioning_mode": (["marigold_pipeline", "pager_max_length"], {"default": "marigold_pipeline"}),
                "filename": (filename_choices, {"default": AUTO_CONDITIONING_FILENAME}),
                "overwrite": ("BOOLEAN", {"default": True}),
                "save_dtype": (["fp16", "fp32"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "Marigold Conditioning"

    DESCRIPTION = "Generates empty-prompt embedding from base text encoder and saves it as a safetensor in the base folder."

    def save(
        self,
        base_text_encoder,
        conditioning_mode: str = "marigold_pipeline",
        filename: str = AUTO_CONDITIONING_FILENAME,
        overwrite: bool = True,
        save_dtype: str = "fp16",
    ):
        tokenizer = base_text_encoder.get("tokenizer", None)
        text_encoder = base_text_encoder.get("text_encoder", None)
        base_dir = base_text_encoder.get("base_dir", None)
        if tokenizer is None or text_encoder is None or base_dir is None:
            raise RuntimeError("Invalid BASE_TEXT_ENCODER input. Re-run `Load Marigold Base Text Encoder`.")

        filename = (filename or "").strip()
        if not filename or filename == AUTO_CONDITIONING_FILENAME:
            filename = _default_conditioning_filename_for_mode(conditioning_mode)

        device = next(text_encoder.parameters()).device
        with torch.inference_mode():
            if conditioning_mode == "marigold_pipeline":
                text_inputs = tokenizer(
                    "",
                    padding="do_not_pad",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
            elif conditioning_mode == "pager_max_length":
                text_inputs = tokenizer(
                    [""],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
            else:
                raise ValueError(f"Invalid conditioning_mode={conditioning_mode!r}")

            text_input_ids = text_inputs.input_ids.to(device)
            embedding = text_encoder(text_input_ids)[0]

        if save_dtype == "fp16":
            embedding = embedding.to(dtype=torch.float16)
        elif save_dtype == "fp32":
            embedding = embedding.to(dtype=torch.float32)
        else:
            raise ValueError(f"Invalid save_dtype={save_dtype!r}")

        path = _save_precomputed_conditioning(
            base_dir=base_dir,
            embedding=embedding,
            filename=filename,
            overwrite=bool(overwrite),
            metadata={
                "conditioning_mode": conditioning_mode,
                "save_dtype": save_dtype,
            },
        )
        return (path,)


class LoadEmptyPromptConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        variants = _available_base_variants()
        default_variant = variants[0]
        filename_choices = _conditioning_filename_choices(
            preferred=[EMPTY_PROMPT_FILENAME_MARIGOLD, EMPTY_PROMPT_FILENAME_PAGER, EMPTY_PROMPT_FILENAME],
        )
        return {
            "required": {
                "model_variant": (variants, {"default": default_variant}),
                "precision": (["auto", "fp16", "bf16", "fp32"], {"default": "auto"}),
                "filename": (filename_choices, {"default": EMPTY_PROMPT_FILENAME_MARIGOLD}),
                "force_gpu": ("BOOLEAN", {"default": False, "advanced": True}),
            }
        }

    RETURN_TYPES = ("EMPTY_CONDITIONING",)
    RETURN_NAMES = ("empty_conditioning",)
    FUNCTION = "load"
    CATEGORY = "Marigold Conditioning"

    DESCRIPTION = "Loads precomputed empty-prompt conditioning tensor from shared base/<variant>."

    def load(
        self,
        model_variant: str,
        precision: str,
        filename: str = EMPTY_PROMPT_FILENAME_MARIGOLD,
        force_gpu: bool = False,
    ):
        device = mm.get_torch_device()
        if force_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        dtype = _select_torch_dtype(precision, device)

        base_dir = _require_local_base_variant(model_variant)
        tensor = _load_precomputed_conditioning(base_dir, dtype=dtype, device=device, filename=filename)
        if tensor is None:
            raise RuntimeError(f"Precomputed conditioning file not found: {base_dir}/{filename}")
        return ({"tensor": tensor, "base_dir": base_dir, "dtype": dtype},)
