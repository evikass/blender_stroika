
def install_sdpa_attention():
    """Hi3D's decoder attends inside fixed windows of 'patch_size' points
    with flash-attn. That build gives NaN now and then on this card, which
    empties the whole decode. The windows are all the same size, so plain
    PyTorch attention does the same maths without it."""
    import torch
    import torch.nn.functional as F
    from sparc3d.models.autoencoders import blocks as B
    if getattr(B.SerializedAttention, "_sdpa", False):
        return
    orig = B.SerializedAttention.forward

    def forward(self, point):
        H = self.num_heads
        C = self.channels
        P = self.patch_size
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        n = int(pad.shape[0])
        if n % P != 0:
            return orig(self, point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        qkv = self.qkv(point.feat)[order]
        dt = qkv.dtype
        qkv = qkv.half().reshape(n // P, P, 3, H, C // H)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # (W, H, P, D)
        out = torch.empty((n // P, H, P, C // H), dtype=q.dtype,
                          device=q.device)
        step = 4096
        for s in range(0, n // P, step):
            out[s:s + step] = F.scaled_dot_product_attention(
                q[s:s + step], k[s:s + step], v[s:s + step],
                scale=self.scale)
        feat = out.permute(0, 2, 1, 3).reshape(-1, C).to(dt)[inverse]
        point.feat = self.proj(feat)
        return point
    B.SerializedAttention.forward = forward
    B.SerializedAttention._sdpa = True
