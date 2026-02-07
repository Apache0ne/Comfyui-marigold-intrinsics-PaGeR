import json
import os

import torch

import comfy.model_management as mm

from .nodes_shared import (
    EMPTY_PROMPT_FILENAME_MARIGOLD,
    MODEL_REPO_ID_APPEARANCE,
    MODEL_REPO_ID_LIGHTING,
    STORAGE_DIRNAME,
    _cleanup_hf_cache,
    _conditioning_filename_choices,
    _ensure_base_files,
    _ensure_model_files,
    _load_precomputed_conditioning,
    _require_module,
    _select_torch_dtype,
)


class DownloadAndLoadMarigoldIIDAppearanceModel:
    @classmethod
    def INPUT_TYPES(cls):
        filename_choices = _conditioning_filename_choices(preferred=[EMPTY_PROMPT_FILENAME_MARIGOLD])
        return {
            "required": {
                "model_variant": (["fp16", "fp32"], {"default": "fp16"}),
                "precision": (["auto", "fp16", "bf16", "fp32"], {"default": "auto"}),
                "use_precomputed_conditioning": ("BOOLEAN", {"default": False}),
                "conditioning_filename": (filename_choices, {"default": EMPTY_PROMPT_FILENAME_MARIGOLD}),
            }
        }

    RETURN_TYPES = ("IIDMODEL",)
    RETURN_NAMES = ("iid_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Marigold IID Appearance"

    DESCRIPTION = f"""
Downloads & loads `{MODEL_REPO_ID_APPEARANCE}` via 🤗 diffusers.

Storage (deduplicated):
- Common SD2 components (tokenizer/text_encoder/vae/scheduler) are stored once under `models/{STORAGE_DIRNAME}/base/<variant>/`.
- Model-specific weights (UNet + model_index.json) are stored under `models/{STORAGE_DIRNAME}/appearance/<variant>/`.
"""

    def loadmodel(
        self,
        model_variant: str,
        precision: str,
        use_precomputed_conditioning: bool = False,
        conditioning_filename: str = EMPTY_PROMPT_FILENAME_MARIGOLD,
    ):
        device = mm.get_torch_device()
        dtype = _select_torch_dtype(precision, device)

        config = {
            "model_variant": model_variant,
            "dtype": str(dtype),
            "use_precomputed_conditioning": bool(use_precomputed_conditioning),
            "conditioning_filename": str(conditioning_filename),
        }
        if not hasattr(self, "_pipe") or self._pipe is None or getattr(self, "_config", None) != config:
            _require_module("diffusers", "pip install -r custom_nodes/Comfyui-marigold-intrinsics/requirements.txt")
            if model_variant not in ("fp16", "fp32"):
                raise ValueError(f"Invalid model_variant={model_variant!r}")

            _require_module("transformers", "pip install -r requirements.txt")
            from diffusers import AutoencoderKL, DDIMScheduler, MarigoldIntrinsicsPipeline, UNet2DConditionModel
            from transformers import CLIPTextModel, CLIPTokenizer

            base_dir = _ensure_base_files(model_variant, MODEL_REPO_ID_APPEARANCE)
            model_dir = _ensure_model_files(MODEL_REPO_ID_APPEARANCE, "appearance", model_variant)

            with open(os.path.join(model_dir, "model_index.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)

            unet = UNet2DConditionModel.from_pretrained(
                model_dir,
                subfolder="unet",
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
            )
            vae = AutoencoderKL.from_pretrained(
                base_dir,
                subfolder="vae",
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
            )
            scheduler = DDIMScheduler.from_pretrained(
                base_dir,
                subfolder="scheduler",
                local_files_only=True,
            )
            tokenizer = CLIPTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer"), local_files_only=True)
            text_encoder = CLIPTextModel.from_pretrained(
                os.path.join(base_dir, "text_encoder"),
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
            )

            pipe = MarigoldIntrinsicsPipeline(
                unet=unet,
                vae=vae,
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prediction_type=cfg.get("prediction_type"),
                target_properties=cfg.get("target_properties"),
                default_denoising_steps=cfg.get("default_denoising_steps"),
                default_processing_resolution=cfg.get("default_processing_resolution"),
            )

            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass

            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"[Marigold IID] xFormers not enabled: {e}")

            if use_precomputed_conditioning:
                try:
                    cond = _load_precomputed_conditioning(
                        base_dir=base_dir,
                        dtype=dtype,
                        device=device,
                        filename=str(conditioning_filename),
                    )
                    if cond is not None:
                        expected_tokens = int(
                            tokenizer(
                                "",
                                padding="do_not_pad",
                                max_length=tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt",
                            ).input_ids.shape[1]
                        )
                        if cond.ndim != 3 or int(cond.shape[1]) != expected_tokens:
                            print(
                                "[Marigold IID] Precomputed conditioning shape mismatch; "
                                f"expected [1,{expected_tokens},C], got {tuple(cond.shape)}. "
                                "Using runtime-generated conditioning."
                            )
                        else:
                            pipe.empty_text_embedding = cond
                            print(f"[Marigold IID] Using precomputed conditioning: {base_dir}/{conditioning_filename}")
                    else:
                        print(
                            f"[Marigold IID] Precomputed conditioning not found: {base_dir}/{conditioning_filename}"
                        )
                except Exception as e:
                    print(f"[Marigold IID] Failed to load precomputed conditioning: {e}")

            pipe = pipe.to(device)

            self._pipe = pipe
            self._config = config
            _cleanup_hf_cache()

        return (
            {
                "pipe": self._pipe,
                "dtype": dtype,
                "kind": "appearance",
                "repo_id": MODEL_REPO_ID_APPEARANCE,
            },
        )


class DownloadAndLoadMarigoldIIDLightingModel:
    @classmethod
    def INPUT_TYPES(cls):
        filename_choices = _conditioning_filename_choices(preferred=[EMPTY_PROMPT_FILENAME_MARIGOLD])
        return {
            "required": {
                "model_variant": (["fp16", "fp32"], {"default": "fp16"}),
                "precision": (["auto", "fp16", "bf16", "fp32"], {"default": "auto"}),
                "use_precomputed_conditioning": ("BOOLEAN", {"default": False}),
                "conditioning_filename": (filename_choices, {"default": EMPTY_PROMPT_FILENAME_MARIGOLD}),
            }
        }

    RETURN_TYPES = ("IIDMODEL",)
    RETURN_NAMES = ("iid_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Marigold IID Lighting"

    DESCRIPTION = f"""
Downloads & loads `{MODEL_REPO_ID_LIGHTING}` via 🤗 diffusers.

This model decomposes an image into:
- Albedo
- Diffuse shading
- Non-diffuse residual
"""

    def loadmodel(
        self,
        model_variant: str,
        precision: str,
        use_precomputed_conditioning: bool = False,
        conditioning_filename: str = EMPTY_PROMPT_FILENAME_MARIGOLD,
    ):
        device = mm.get_torch_device()
        dtype = _select_torch_dtype(precision, device)

        config = {
            "model_variant": model_variant,
            "dtype": str(dtype),
            "use_precomputed_conditioning": bool(use_precomputed_conditioning),
            "conditioning_filename": str(conditioning_filename),
        }
        if not hasattr(self, "_pipe") or self._pipe is None or getattr(self, "_config", None) != config:
            _require_module("diffusers", "pip install -r custom_nodes/Comfyui-marigold-intrinsics/requirements.txt")
            if model_variant not in ("fp16", "fp32"):
                raise ValueError(f"Invalid model_variant={model_variant!r}")

            _require_module("transformers", "pip install -r requirements.txt")
            from diffusers import AutoencoderKL, DDIMScheduler, MarigoldIntrinsicsPipeline, UNet2DConditionModel
            from transformers import CLIPTextModel, CLIPTokenizer

            base_dir = _ensure_base_files(model_variant, MODEL_REPO_ID_LIGHTING)
            model_dir = _ensure_model_files(MODEL_REPO_ID_LIGHTING, "lighting", model_variant)

            with open(os.path.join(model_dir, "model_index.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)

            unet = UNet2DConditionModel.from_pretrained(
                model_dir,
                subfolder="unet",
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
            )
            vae = AutoencoderKL.from_pretrained(
                base_dir,
                subfolder="vae",
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
            )
            scheduler = DDIMScheduler.from_pretrained(
                base_dir,
                subfolder="scheduler",
                local_files_only=True,
            )
            tokenizer = CLIPTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer"), local_files_only=True)
            text_encoder = CLIPTextModel.from_pretrained(
                os.path.join(base_dir, "text_encoder"),
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
            )

            pipe = MarigoldIntrinsicsPipeline(
                unet=unet,
                vae=vae,
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prediction_type=cfg.get("prediction_type"),
                target_properties=cfg.get("target_properties"),
                default_denoising_steps=cfg.get("default_denoising_steps"),
                default_processing_resolution=cfg.get("default_processing_resolution"),
            )

            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass

            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"[Marigold IID] xFormers not enabled: {e}")

            if use_precomputed_conditioning:
                try:
                    cond = _load_precomputed_conditioning(
                        base_dir=base_dir,
                        dtype=dtype,
                        device=device,
                        filename=str(conditioning_filename),
                    )
                    if cond is not None:
                        expected_tokens = int(
                            tokenizer(
                                "",
                                padding="do_not_pad",
                                max_length=tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt",
                            ).input_ids.shape[1]
                        )
                        if cond.ndim != 3 or int(cond.shape[1]) != expected_tokens:
                            print(
                                "[Marigold IID] Precomputed conditioning shape mismatch; "
                                f"expected [1,{expected_tokens},C], got {tuple(cond.shape)}. "
                                "Using runtime-generated conditioning."
                            )
                        else:
                            pipe.empty_text_embedding = cond
                            print(f"[Marigold IID] Using precomputed conditioning: {base_dir}/{conditioning_filename}")
                    else:
                        print(
                            f"[Marigold IID] Precomputed conditioning not found: {base_dir}/{conditioning_filename}"
                        )
                except Exception as e:
                    print(f"[Marigold IID] Failed to load precomputed conditioning: {e}")

            pipe = pipe.to(device)

            self._pipe = pipe
            self._config = config
            _cleanup_hf_cache()

        return (
            {
                "pipe": self._pipe,
                "dtype": dtype,
                "kind": "lighting",
                "repo_id": MODEL_REPO_ID_LIGHTING,
            },
        )
