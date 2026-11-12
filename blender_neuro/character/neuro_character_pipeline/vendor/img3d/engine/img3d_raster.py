"""Triangle rasterizer in plain PyTorch, for Image to 3D.

Written for this add-on (3DRedbox Studio, GPL-3.0-or-later) to replace
NVIDIA's nvdiffrast, whose licence does not allow commercial use. It offers
the few calls the engines use, with the same inputs and outputs:

    rast, _ = rasterize(ctx, pos, tri, resolution)
        pos  [B, V, 4] or [V, 4] clip-space positions
        tri  [F, 3] int triangle vertex indices
        rast [B, H, W, 4]: (u, v, z/w, triangle index + 1); 0 = empty.
              u and v weigh vertices 0 and 1, 1 - u - v weighs vertex 2,
              perspective-correct. Row 0 is the bottom row.
    out, _ = interpolate(attr, rast, tri)
        attr [B or 1, V, C] or [V, C] -> out [B, H, W, C]

How it works: every triangle is snapped to the pixel centres inside its
bounding box (triangles are grouped by box size, so the work stays small),
and the nearest triangle per pixel wins through one scatter-min on a
depth+index key. No compiler, no extra package; runs on any CUDA card.
"""
import math

import torch

# the pixel candidates handled at once: bounds the memory a call needs
_BUDGET = 12_000_000


class RasterizeCudaContext:
    """Kept for the callers' sake: nothing to set up here."""

    def __init__(self, device=None):
        self.device = device


RasterizeGLContext = RasterizeCudaContext


def _edge(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


@torch.no_grad()
def _raster_one(pos, tri, H, W):
    dev = pos.device
    pos = pos.float()
    tri = tri.long()
    F = tri.shape[0]
    key_none = torch.iinfo(torch.int64).max
    zbuf = torch.full((H * W,), key_none, dtype=torch.int64, device=dev)
    if F == 0:
        return _decode(zbuf, pos, tri, H, W)
    w = pos[:, 3]
    safe_w = torch.where(w.abs() < 1e-8, torch.full_like(w, 1e-8), w)
    # pixel centre i sits at x == i
    sx = (pos[:, 0] / safe_w * 0.5 + 0.5) * W - 0.5
    sy = (pos[:, 1] / safe_w * 0.5 + 0.5) * H - 0.5
    sz = pos[:, 2] / safe_w
    x = sx[tri]
    y = sy[tri]
    z = sz[tri]
    tw = w[tri]
    area = _edge(x[:, 0], y[:, 0], x[:, 1], y[:, 1], x[:, 2], y[:, 2])
    x0 = torch.ceil(x.min(1).values).clamp(0, W - 1)
    x1 = torch.floor(x.max(1).values).clamp(-1, W - 1)
    y0 = torch.ceil(y.min(1).values).clamp(0, H - 1)
    y1 = torch.floor(y.max(1).values).clamp(-1, H - 1)
    # behind the camera, flat, or no pixel centre inside the box
    ok = (tw > 1e-8).all(1) & (area.abs() > 1e-12) & (x1 >= x0) & (y1 >= y0)
    ok &= (x.max(1).values >= 0) & (x.min(1).values <= W - 1)
    ok &= (y.max(1).values >= 0) & (y.min(1).values <= H - 1)
    idx = torch.nonzero(ok).squeeze(1)
    if idx.numel() == 0:
        return _decode(zbuf, pos, tri, H, W)
    bw = (x1[idx] - x0[idx] + 1).long()
    bh = (y1[idx] - y0[idx] + 1).long()
    side = torch.maximum(bw, bh)
    # group by box size: 1, 2, 4, 8 ... pixels on a side
    size_cls = torch.ceil(torch.log2(side.float().clamp(min=1))).long()
    for c in torch.unique(size_cls).tolist():
        sel = idx[size_cls == c]
        s = 1 << int(c)
        per = max(1, _BUDGET // (s * s))
        for k in range(0, sel.numel(), per):
            _splat(sel[k:k + per], s, x, y, z, area, x0, x1, y0, y1, zbuf,
                   H, W)
    return _decode(zbuf, pos, tri, H, W)


def _splat(t, s, x, y, z, area, x0, x1, y0, y1, zbuf, H, W):
    dev = t.device
    n = t.numel()
    off = torch.arange(s, device=dev, dtype=torch.float32)
    px = x0[t][:, None, None] + off[None, None, :]       # [n, 1, s]
    py = y0[t][:, None, None] + off[None, :, None]       # [n, s, 1]
    px = px.expand(n, s, s)
    py = py.expand(n, s, s)
    inb = (px <= x1[t][:, None, None]) & (py <= y1[t][:, None, None])
    xa, xb, xc = (x[t, i][:, None, None] for i in range(3))
    ya, yb, yc = (y[t, i][:, None, None] for i in range(3))
    ar = area[t][:, None, None]
    l0 = _edge(xb, yb, xc, yc, px, py) / ar
    l1 = _edge(xc, yc, xa, ya, px, py) / ar
    l2 = 1.0 - l0 - l1
    eps = -1e-6
    inside = inb & (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
    zz = (l0 * z[t, 0][:, None, None] + l1 * z[t, 1][:, None, None] +
          l2 * z[t, 2][:, None, None])
    inside &= (zz >= -1.0) & (zz <= 1.0)
    if not bool(inside.any()):
        return
    pix = (py.long() * W + px.long())[inside]
    q = ((zz[inside].double() + 1.0) * 0.5 * 2147483646.0).long()
    tid = t[:, None, None].expand(n, s, s)[inside]
    key = (q << 32) | tid
    zbuf.scatter_reduce_(0, pix, key, reduce="amin")


def _decode(zbuf, pos, tri, H, W):
    dev = pos.device
    rast = torch.zeros((H * W, 4), dtype=torch.float32, device=dev)
    hit = zbuf != torch.iinfo(torch.int64).max
    if bool(hit.any()):
        p = torch.nonzero(hit).squeeze(1)
        t = zbuf[p] & 0xFFFFFFFF
        px = (p % W).float()
        py = (p // W).float()
        w = pos[:, 3]
        sx = (pos[:, 0] / w * 0.5 + 0.5) * W - 0.5
        sy = (pos[:, 1] / w * 0.5 + 0.5) * H - 0.5
        sz = pos[:, 2] / w
        v = tri[t]
        xa, xb, xc = sx[v[:, 0]], sx[v[:, 1]], sx[v[:, 2]]
        ya, yb, yc = sy[v[:, 0]], sy[v[:, 1]], sy[v[:, 2]]
        ar = _edge(xa, ya, xb, yb, xc, yc)
        l0 = _edge(xb, yb, xc, yc, px, py) / ar
        l1 = _edge(xc, yc, xa, ya, px, py) / ar
        l2 = 1.0 - l0 - l1
        zz = l0 * sz[v[:, 0]] + l1 * sz[v[:, 1]] + l2 * sz[v[:, 2]]
        # perspective-correct weights, as the attributes are in 3D
        q0 = l0 / w[v[:, 0]]
        q1 = l1 / w[v[:, 1]]
        q2 = l2 / w[v[:, 2]]
        qs = q0 + q1 + q2
        rast[p, 0] = q0 / qs
        rast[p, 1] = q1 / qs
        rast[p, 2] = zz
        rast[p, 3] = t.float() + 1.0
    return rast.view(H, W, 4)


@torch.no_grad()
def rasterize(ctx, pos, tri, resolution, ranges=None, grad_db=True):
    if pos.dim() == 2:
        pos = pos[None]
    H, W = int(resolution[0]), int(resolution[1])
    out = torch.stack([_raster_one(pos[b], tri, H, W)
                       for b in range(pos.shape[0])])
    return out, None


@torch.no_grad()
def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    if attr.dim() == 2:
        attr = attr[None]
    tri = tri.long()
    B, H, W, _ = rast.shape
    outs = []
    for b in range(B):
        a = attr[b if attr.shape[0] > 1 else 0]
        r = rast[b].reshape(-1, 4)
        tid = r[:, 3].long() - 1
        hit = tid >= 0
        out = torch.zeros((H * W, a.shape[-1]), dtype=a.dtype,
                          device=a.device)
        if bool(hit.any()):
            v = tri[tid[hit]]
            u_ = r[hit, 0:1].to(a.dtype)
            v_ = r[hit, 1:2].to(a.dtype)
            out[hit] = u_ * a[v[:, 0]] + v_ * a[v[:, 1]] + \
                (1 - u_ - v_) * a[v[:, 2]]
        outs.append(out.view(H, W, -1))
    return torch.stack(outs), None


def install_as_nvdiffrast():
    """Answer `import nvdiffrast.torch` with this module, so code written
    for nvdiffrast (inside the TRELLIS.2 packages) runs on it unchanged and
    nvdiffrast itself is never needed."""
    import sys
    import types
    me = sys.modules[__name__]
    pkg = types.ModuleType("nvdiffrast")
    pkg.torch = me
    pkg.__path__ = []
    sys.modules["nvdiffrast"] = pkg
    sys.modules["nvdiffrast.torch"] = me
