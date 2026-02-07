# ComfyUI Marigold Intrinsics PaGeR

This custom node pack provides Marigold IID nodes and PaGeR ERP geometry nodes.

## Included Nodes

### Marigold IID Loaders

1. `Load Marigold IID Appearance Model` (`DownloadAndLoadMarigoldIIDAppearanceModel`)
- Downloads and loads the Marigold appearance pipeline as `IIDMODEL`.
- Supports `use_precomputed_conditioning`.

2. `Load Marigold IID Lighting Model` (`DownloadAndLoadMarigoldIIDLightingModel`)
- Downloads and loads the Marigold lighting pipeline as `IIDMODEL`.
- Supports `use_precomputed_conditioning`.

3. `Load Marigold IID Split (UNet/CLIP/VAE)` (`MarigoldIIDSplitLoader`)
- Loads split components as Comfy `MODEL`, `CLIP`, `VAE`.
- Supports tiny VAEs, including TAESD-style VAE choices.

4. `Translate Marigold Split -> IIDMODEL` (`MarigoldIIDSplitToIIDModel`)
- Converts split-loaded `MODEL` + `CLIP` + `VAE` into `IIDMODEL`.
- Supports precomputed conditioning and TAESD-compatible VAE adapters.

### Marigold Conditioning

1. `Load Marigold Base Text Encoder` (`LoadMarigoldBaseTextEncoder`)
- Loads tokenizer and CLIP text encoder from shared base files.

2. `Save Empty Prompt Conditioning` (`SaveEmptyPromptConditioning`)
- Computes and saves empty-prompt text embedding to a `.safetensors` file.

3. `Load Empty Prompt Conditioning` (`LoadEmptyPromptConditioning`)
- Loads a saved empty-prompt embedding for reuse.

### Marigold IID Inference

1. `Marigold IID Appearance` (`MarigoldIIDAppearance`)
- Runs appearance inference and returns `albedo`, `roughness`, `metallicity`.

2. `Marigold IID Appearance (with Material)` (`MarigoldIIDAppearanceExtended`)
- Same as above, plus `material`.

3. `Marigold IID Lighting (HyperSim)` (`MarigoldIIDLighting`)
- Runs lighting inference and returns `albedo`, `shading`, `residual`.

### PaGeR

1. `Load PaGeR Model` (`DownloadAndLoadPaGeRModel`)
- Loads PaGeR depth or normal model.
- Supports precomputed conditioning.
- Includes optional Comfy TAESD VAE backend (`use_comfy_taesd_vae`).

2. `PaGeR Infer Cubemap (ERP)` (`PaGeRInferCubemap`)
- Node A in the PaGeR pipeline.
- Runs encode + UNet + decode and returns decoded cubemap predictions.

3. `PaGeR Depth Postprocess (ERP)` (`PaGeRDepthPostprocess`)
- Node B depth postprocess.
- Converts cubemap prediction to ERP outputs:
  - `depth`
  - `edge_mask`
  - `depth_color`

4. `PaGeR Normal Postprocess (ERP)` (`PaGeRNormalPostprocess`)
- Node B normal postprocess.
- Converts cubemap prediction to ERP normal visualization.

5. `PaGeR Save Point Cloud (GLB/PLY)` (`PaGeRSavePointCloud`)
- Exports point cloud from `color` + `depth` as `.glb` or `.ply`.
- Supports `depth_mode` auto logic, optional masks, and optional edge filtering.

## Recommended PaGeR Pipeline

For depth:
1. `Load PaGeR Model` with a depth checkpoint.
2. `PaGeR Infer Cubemap (ERP)`.
3. `PaGeR Depth Postprocess (ERP)`.
4. Optional: `PaGeR Save Point Cloud (GLB/PLY)`.

For normals:
1. `Load PaGeR Model` with a normals checkpoint.
2. `PaGeR Infer Cubemap (ERP)`.
3. `PaGeR Normal Postprocess (ERP)`.

## Precomputed Text Conditioning

Precomputed conditioning means reusing a saved empty-prompt embedding instead of generating it every run.

How it works:
1. Load tokenizer/text encoder once with `Load Marigold Base Text Encoder`.
2. Save the empty-prompt embedding using `Save Empty Prompt Conditioning`.
3. Enable `use_precomputed_conditioning` in model loader nodes, or load it with `Load Empty Prompt Conditioning`.

Why it helps:
1. Reduces repeated text-encoding work during inference setup.
2. Lowers overhead and can improve throughput consistency in repeated runs.
3. Can reduce transient VRAM pressure from extra text-encoding passes.

## TAESD VAE Option

`Load PaGeR Model` supports an optional Comfy TAESD VAE backend.

Why use it:
1. Lower VRAM usage.
2. Faster VAE encode/decode on constrained GPUs.
3. Useful when fitting PaGeR workflows on smaller cards.

Tradeoff:
1. TAESD is a tiny/approximate VAE.
2. Output quality can degrade in fine details, smooth gradients, and sharp depth edges.
3. In many scenes it is still usable, especially when speed and memory are priority.

## 8GB VRAM Note

This setup has been tested on an 8GB GPU with practical settings:
1. Prefer `fp16` precision.
2. Use precomputed conditioning.
3. Enable TAESD VAE when memory is tight.
4. Keep PaGeR as `Infer Cubemap -> Postprocess` (current default workflow).

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

## 🎓 Citation

Please cite our paper:  (Waiting for Citation)

```bibtex
Put citations here
```

## 🎫 License

This code of this work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE)).

The models are licensed under RAIL++-M License (as defined in the [LICENSE-MODEL](LICENSE-MODEL))

By downloading and using the code and model you agree to the terms in [LICENSE](LICENSE) and [LICENSE-MODEL](LICENSE-MODEL) respectively.

## Acknowledgements

This project builds upon and is inspired by the following repositories and works:

- [Marigold-e2e-ft](https://github.com/VisualComputingInstitute/diffusion-e2e-ft), based on paper [Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think](https://arxiv.org/abs/2409.11355).
- [Marigold](https://github.com/prs-eth/Marigold/tree/main), based on paper [Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation](https://arxiv.org/abs/2312.02145).

We thank the authors and maintainers for making their code publicly available.
