"""custom_rasterizer on top of the add-on's own rasterizer (_raster.py).

Hunyuan3D-2.1 ships its own CUDA rasterizer, which has to be compiled for the
exact PyTorch on the machine. _raster.py does the same job in plain
PyTorch, so this gives Hunyuan the two calls it uses with the same outputs: face index + 1 per pixel (0 = empty) and per-vertex barycentrics.
"""
import torch
from . import _raster as dr

_CTX = {}


def _ctx(device):
    key = str(device)
    if key not in _CTX:
        _CTX[key] = dr.RasterizeCudaContext(device=device)
    return _CTX[key]


def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
    if pos.dim() == 2:
        pos = pos[None]
    pos = pos.float().contiguous()
    tri = tri.int().contiguous()
    h, w = int(resolution[0]), int(resolution[1])
    rast, _ = dr.rasterize(_ctx(pos.device), pos, tri, resolution=[h, w])
    rast = rast[0]
    u, v = rast[..., 0], rast[..., 1]
    findices = rast[..., 3].round().int()
    bary = torch.stack([u, v, 1.0 - u - v], dim=-1)
    bary = bary * (findices > 0)[..., None]
    return findices, bary


def interpolate(col, findices, barycentric, tri):
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, axis=-2)
    return result.view(1, *result.shape)
