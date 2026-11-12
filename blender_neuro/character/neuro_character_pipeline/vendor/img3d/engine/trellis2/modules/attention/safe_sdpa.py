"""PyTorch attention that also fits on older cards.

RTX 20 / GTX 16 cards (compute 7.x) have no fast bfloat16 attention: PyTorch
falls back to its plain kernel, which builds the whole score matrix at once.
For a detailed model that is several GB and the card runs out of memory, and
it is slow. There we hand the attention over in float16, which uses the
memory-efficient kernel, and work through the queries in slices so even the
plain kernel never needs more than a few hundred MB.
Newer cards are left exactly as they were.
"""
import os
import torch
import torch.nn.functional as F

_OLD = {}
_SIM = bool(os.environ.get("IMG3D_SIM_TURING"))   # test the old-card path
_SLICE_BYTES = 256 * 2 ** 20


def _old_card(dev):
    if _SIM:
        return True
    if dev.type != "cuda":
        return False
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    if key not in _OLD:
        try:
            _OLD[key] = torch.cuda.get_device_capability(key)[0] < 8
        except Exception:
            _OLD[key] = False
    return _OLD[key]


def sdpa_safe(q, k, v):
    """q, k, v: [B, H, L, C] like F.scaled_dot_product_attention."""
    if not _old_card(q.device):
        return F.scaled_dot_product_attention(q, k, v)
    dt = q.dtype
    if dt == torch.bfloat16:
        # float16 holds the same numbers here (q and k are normalised), but
        # check, since a value past 65504 would turn into infinity
        big = max(float(t.abs().amax()) for t in (q, k, v))
        if big < 60000.0:
            q, k, v = q.half(), k.half(), v.half()
    B, H, Lq = q.shape[0], q.shape[1], q.shape[-2]
    Lk = k.shape[-2]
    step = max(128, int(_SLICE_BYTES // (4 * B * H * max(Lk, 1))))
    if Lq <= step:
        out = F.scaled_dot_product_attention(q, k, v)
    else:
        out = torch.cat([F.scaled_dot_product_attention(q[..., s:s + step, :], k, v)
                         for s in range(0, Lq, step)], dim=-2)
    return out.to(dt)
