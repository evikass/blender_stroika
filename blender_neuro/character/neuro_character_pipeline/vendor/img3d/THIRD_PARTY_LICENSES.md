# Image to 3D: licences

## The add-on

Image to 3D (the Blender add-on, its card UI, its engine scripts and its
rasterizer) is © 3DRedbox Studio Kft. and is licensed under the
GNU General Public License v3.0 or later (GPL-3.0-or-later).

Image to 3D is a front end. It does not contain any AI model, and it does
not contain the code of Tencent's Hunyuan3D. Each engine is optional and is
downloaded only when you choose it in the Downloads section, directly from
its makers' official pages (Hugging Face and GitHub). Every engine comes
with its own licence, which governs your use of that engine and of what you
make with it. Please read the licence of each engine you download. You can
use the add-on with any one engine and skip the others.

## Engines and models (downloaded on request, not included)

| Engine / model | Used for | Licence | Notes |
|---|---|---|---|
| Microsoft TRELLIS.2-4B, TRELLIS-image-large | Default 3D engine | MIT | |
| TencentARC Pixal3D | Pixal3D engine | MIT | Its third-party parts keep their own licences (see its NOTICE). |
| Tencent Hunyuan3D 2.1 (code and weights) | Hunyuan engine | Tencent Hunyuan 3D 2.1 Community License | **Does not apply in the European Union, the United Kingdom and South Korea.** Downloaded straight from Tencent's GitHub and Hugging Face pages, with Tencent's LICENSE and Notice.txt. |
| Tencent Hunyuan3D-2mv | Hunyuan with extra views | Tencent Hunyuan 3D 2.0 Community License | Same territory limits as above. |
| Hi3D v1.0 | Enhance Detail / Enhance 3D Model | Hi3D Open License Agreement 1.0 | Commercial use allowed. Products must show "Powered by Hi3D 1.0" (the add-on shows it next to these features). Above 200,000 monthly active users or USD 1,000,000 monthly revenue a separate licence from Hi3D is required. |
| Black Forest Labs FLUX.2 [klein] 4B | Enhance Texture | Apache-2.0 | |
| Meta DINOv3 ViT-L/16 | Image features for TRELLIS.2 | DINOv3 License | No use under ITAR or for military or weapons purposes. |
| Meta DINOv2 (large, giant) | Image features | Apache-2.0 | |
| OpenAI CLIP ViT-L/14 | Image features | MIT | |
| BiRefNet | Background removal | MIT | |
| MoGe-2 ViT-L | Depth for Pixal3D | MIT | |
| Marigold IID Lighting v1.1 | Material maps | CreativeML Open RAIL++-M | Its use restrictions pass on to you. |
| Real-ESRGAN x4 | Texture upscaling | BSD-3-Clause | |
| U-2-Net | Background removal | Apache-2.0 | |

Required notice for Tencent Hunyuan 3D 2.1, as it appears in Tencent's own
download:
"Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1
Community License Agreement, Copyright © 2025 Tencent. All Rights Reserved.
The trademark rights of "Tencent Hunyuan" are owned by Tencent or its
affiliate."

## Software libraries the engine installer downloads

PyTorch and torchvision (BSD-3-Clause), Hugging Face diffusers, transformers,
accelerate, safetensors and tokenizers (Apache-2.0), spconv (Apache-2.0),
FlashAttention (BSD-3-Clause), Triton for Windows (MIT), NATTEN (MIT),
CuMesh, FlexGEMM and O-Voxel from TRELLIS.2 (MIT), utils3d (MIT),
xatlas (MIT), trimesh (MIT), PyMeshLab (GPL-3.0), fast-simplification (MIT),
pyfqmr (MIT), rembg (MIT), ONNX Runtime (MIT), OpenCV (Apache-2.0),
kornia (Apache-2.0), timm (Apache-2.0), bitsandbytes (MIT), spandrel (MIT),
ModernGL (MIT), NumPy, SciPy and scikit-image (BSD-3-Clause), Pillow
(MIT-CMU). Each keeps its own licence, installed alongside it.

The add-on does not use or install NVIDIA nvdiffrast, NVIDIA nvdiffrec or
MeshLib (versions before 1.1.6 did; they are removed from the engine folder
automatically on update).
