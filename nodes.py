from .nodes_appearance import MarigoldIIDAppearance, MarigoldIIDAppearanceExtended
from .nodes_lighting import MarigoldIIDLighting
from .nodes_loaders import DownloadAndLoadMarigoldIIDAppearanceModel, DownloadAndLoadMarigoldIIDLightingModel
from .nodes_pager import (
    DownloadAndLoadPaGeRModel,
    PaGeRDepthPostprocess,
    PaGeRInferCubemap,
    PaGeRNormalPostprocess,
    PaGeRSavePointCloud,
)
from .splitloader import MarigoldIIDSplitLoader, MarigoldIIDSplitToIIDModel


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadMarigoldIIDAppearanceModel": DownloadAndLoadMarigoldIIDAppearanceModel,
    "MarigoldIIDAppearance": MarigoldIIDAppearance,
    "MarigoldIIDAppearanceExtended": MarigoldIIDAppearanceExtended,
    "DownloadAndLoadMarigoldIIDLightingModel": DownloadAndLoadMarigoldIIDLightingModel,
    "MarigoldIIDLighting": MarigoldIIDLighting,
    "DownloadAndLoadPaGeRModel": DownloadAndLoadPaGeRModel,
    "PaGeRInferCubemap": PaGeRInferCubemap,
    "PaGeRDepthPostprocess": PaGeRDepthPostprocess,
    "PaGeRNormalPostprocess": PaGeRNormalPostprocess,
    "PaGeRSavePointCloud": PaGeRSavePointCloud,
    "MarigoldIIDSplitLoader": MarigoldIIDSplitLoader,
    "MarigoldIIDSplitToIIDModel": MarigoldIIDSplitToIIDModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadMarigoldIIDAppearanceModel": "Load Marigold IID Appearance Model",
    "MarigoldIIDAppearance": "Marigold IID Appearance",
    "MarigoldIIDAppearanceExtended": "Marigold IID Appearance (with Material)",
    "DownloadAndLoadMarigoldIIDLightingModel": "Load Marigold IID Lighting Model",
    "MarigoldIIDLighting": "Marigold IID Lighting (HyperSim)",
    "DownloadAndLoadPaGeRModel": "Load PaGeR Model",
    "PaGeRInferCubemap": "PaGeR Infer Cubemap (ERP)",
    "PaGeRDepthPostprocess": "PaGeR Depth Postprocess (ERP)",
    "PaGeRNormalPostprocess": "PaGeR Normal Postprocess (ERP)",
    "PaGeRSavePointCloud": "PaGeR Save Point Cloud (GLB/PLY)",
    "MarigoldIIDSplitLoader": "Load Marigold IID Split (UNet/CLIP/VAE)",
    "MarigoldIIDSplitToIIDModel": "Translate Marigold Split -> IIDMODEL",
}
