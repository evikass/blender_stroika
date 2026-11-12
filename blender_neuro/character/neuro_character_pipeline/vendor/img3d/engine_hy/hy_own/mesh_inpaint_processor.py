"""Pure Python stand-in for Hunyuan3D-2.1's compiled mesh_inpaint_processor.

The original is a C++ extension that needs a compiler on the user's machine.
This does the same "smooth" vertex inpaint: vertices whose UV lands on a
painted texel take that colour, the rest are filled from coloured neighbours
weighted by inverse squared distance, pass after pass, until nothing more
changes. Then each newly coloured vertex is written back to its texel.
"""

import numpy as np


def meshVerticeInpaint(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx,
                       method="smooth"):
    texture = np.asarray(texture, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.uint8)
    vtx_pos = np.asarray(vtx_pos, dtype=np.float32)
    vtx_uv = np.asarray(vtx_uv, dtype=np.float32)
    pos_idx = np.asarray(pos_idx, dtype=np.int64).reshape(-1, 3)
    uv_idx = np.asarray(uv_idx, dtype=np.int64).reshape(-1, 3)

    h, w, ch = texture.shape
    nv = vtx_pos.shape[0]

    cu = uv_idx.reshape(-1)
    cv = pos_idx.reshape(-1)
    tx = np.rint(vtx_uv[cu, 0] * (w - 1)).astype(np.int64)
    ty = np.rint((1.0 - vtx_uv[cu, 1]) * (h - 1)).astype(np.int64)
    tx = np.clip(tx, 0, w - 1)
    ty = np.clip(ty, 0, h - 1)

    vtx_mask = np.zeros(nv, dtype=np.float32)
    vtx_color = np.zeros((nv, ch), dtype=np.float32)
    painted = mask[ty, tx] > 0
    vtx_mask[cv[painted]] = 1.0
    vtx_color[cv[painted]] = texture[ty[painted], tx[painted]]
    if method != "smooth":
        raise ValueError("Invalid method. Use 'smooth'.")

    # neighbours along the face edges (each corner to the next one); every
    # pass colours all the uncoloured vertices that touch a coloured one at
    # once, weighted by inverse squared distance - whole arrays at a time,
    # so a million-face model takes seconds instead of many minutes
    a = pos_idx.reshape(-1)
    b = np.roll(pos_idx, -1, axis=1).reshape(-1)
    d = np.linalg.norm(vtx_pos[a] - vtx_pos[b], axis=1)
    wgt_all = 1.0 / np.maximum(d, 1e-4) ** 2
    for _ in range(100000):
        unc = vtx_mask == 0
        e = unc[a] & (vtx_mask[b] > 0)
        if not e.any():
            break
        ea, eb, ew = a[e], b[e], wgt_all[e]
        den = np.bincount(ea, weights=ew, minlength=nv)
        num = np.stack([np.bincount(ea, weights=ew * vtx_color[eb, c],
                                    minlength=nv) for c in range(ch)], 1)
        new = unc & (den > 0)
        vtx_color[new] = num[new] / den[new][:, None]
        vtx_mask[new] = 1.0

    new_tex = texture.copy()
    new_mask = mask.copy()
    good = vtx_mask[cv] == 1.0
    new_tex[ty[good], tx[good]] = vtx_color[cv[good]]
    new_mask[ty[good], tx[good]] = 255
    return new_tex[:, :, :3], new_mask


def meshVerticeColor(*args, **kwargs):
    raise NotImplementedError("meshVerticeColor is not used by the engine")
