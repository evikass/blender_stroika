"""Make a picture 2 or 4 times bigger with Real-ESRGAN x4 (BSD-3-Clause),
the same weights Image to 3D already downloads for its textures.

    upscale.py <in.png> <out.png> <2|4> <models dir>

stdout: STEP <0..1> / DONE <png> / ERROR <msg>
The network is written out here (RRDBNet, 23 blocks), so nothing extra has
to be installed; the picture goes through in tiles, so any size fits.
"""
import math
import os
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F


def out(*a):
    print(*a, flush=True)


class RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = RDB(nf), RDB(nf), RDB(nf)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, nb=23, nf=64):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        f = self.conv_first(x)
        f = f + self.conv_body(self.body(f))
        f = self.lrelu(self.conv_up1(F.interpolate(f, scale_factor=2,
                                                   mode="nearest")))
        f = self.lrelu(self.conv_up2(F.interpolate(f, scale_factor=2,
                                                   mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(f)))


def load(models):
    path = os.path.join(models, "realesrgan", "RealESRGAN_x4plus.pth")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("params_ema", "params"):
        if k in sd:
            sd = sd[k]
            break
    net = RRDBNet()
    net.load_state_dict(sd, strict=True)
    return net.eval().cuda().half()


@torch.no_grad()
def upscale(net, img, tile=384, pad=16):
    import numpy as np
    from PIL import Image
    x = torch.from_numpy(np.asarray(img.convert("RGB"))).float()
    x = x.div(255.0).permute(2, 0, 1)[None].cuda().half()
    _, _, H, W = x.shape
    outp = torch.zeros((1, 3, H * 4, W * 4), dtype=torch.half, device="cuda")
    ny, nx = math.ceil(H / tile), math.ceil(W / tile)
    done = 0
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * tile, ix * tile
            y1, x1 = min(H, y0 + tile), min(W, x0 + tile)
            py0, px0 = max(0, y0 - pad), max(0, x0 - pad)
            py1, px1 = min(H, y1 + pad), min(W, x1 + pad)
            o = net(x[:, :, py0:py1, px0:px1])
            oy, ox = (y0 - py0) * 4, (x0 - px0) * 4
            outp[:, :, y0 * 4:y1 * 4, x0 * 4:x1 * 4] = \
                o[:, :, oy:oy + (y1 - y0) * 4, ox:ox + (x1 - x0) * 4]
            done += 1
            out("STEP %.3f" % (done / float(ny * nx)))
    y = outp.clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((y * 255.0).round().astype("uint8"))


def main():
    from PIL import Image
    src, dst, scale, models = sys.argv[1], sys.argv[2], int(sys.argv[3]), \
        sys.argv[4]
    t = time.time()
    net = load(models)
    img = Image.open(src).convert("RGB")
    big = upscale(net, img)
    if scale == 2:
        big = big.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    big.save(dst)
    out("LOG %dx%d in %.1fs" % (big.width, big.height, time.time() - t))
    out("DONE " + dst)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        out("ERROR %s" % str(exc)[:200])
