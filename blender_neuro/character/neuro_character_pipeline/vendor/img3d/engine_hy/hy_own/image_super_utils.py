"""RealESRGAN x4 through spandrel, in place of basicsr/realesrgan, which no
longer build on current Python."""
import numpy as np
import torch
from PIL import Image


class imageSuperNet:
    def __init__(self, config) -> None:
        from spandrel import ModelLoader
        m = ModelLoader().load_from_file(config.realesrgan_ckpt_path)
        m = m.cuda().eval()
        try:
            m = m.half()
            self.half = True
        except Exception:
            self.half = False
        self.model = m

    @torch.no_grad()
    def __call__(self, image):
        x = torch.from_numpy(np.array(image.convert("RGB"))).float()
        x = x.div(255.0).permute(2, 0, 1)[None].cuda()
        if self.half:
            x = x.half()
        y = self.model(x)
        y = y.clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
        return Image.fromarray((y * 255.0).round().astype(np.uint8))
