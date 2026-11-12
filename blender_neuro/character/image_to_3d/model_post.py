"""Finishing steps after a download that need PyTorch, run with the engine's
own Python:

    model_post.py convert_bins <Hunyuan3D-2.1 folder>
        The painter's weights come as PyTorch .bin files. They are read in
        safe mode (weights only, no code can run) and saved as .safetensors,
        which load faster and with less memory. The .bin is removed after.
    model_post.py naf <folder>
        Fetches the small NAF upsampler Pixal3D uses (from its GitHub page).
"""

import os
import sys


def convert_bins(folder):
    import torch
    from safetensors.torch import save_file
    paint = os.path.join(folder, "hunyuan3d-paintpbr-v2-1")
    for sub, name in (("unet", "diffusion_pytorch_model"),
                      ("vae", "diffusion_pytorch_model"),
                      ("text_encoder", "model")):
        d = os.path.join(paint, sub)
        src = os.path.join(d, ("pytorch_model.bin" if sub == "text_encoder"
                               else "diffusion_pytorch_model.bin"))
        dst = os.path.join(d, name + ".safetensors")
        if os.path.isfile(dst) and os.path.isfile(dst + ".ok"):
            continue
        if not os.path.isfile(src):
            continue
        print("converting", sub, flush=True)
        sd = torch.load(src, map_location="cpu", weights_only=True)
        seen = {}
        out = {}
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            key = (v.untyped_storage().data_ptr(), v.storage_offset(),
                   tuple(v.shape))
            out[k] = v.clone().contiguous() if key in seen else v.contiguous()
            seen[key] = k
        save_file(out, dst + ".tmp", metadata={"format": "pt"})
        os.replace(dst + ".tmp", dst)
        open(dst + ".ok", "w").write("1")
        del sd, out
        os.remove(src)
    print("converted", flush=True)


def naf(folder):
    import torch
    torch.hub.load("valeoai/NAF", "naf", pretrained=True, trust_repo=True,
                   device="cpu")
    print("naf ok", flush=True)


if __name__ == "__main__":
    step, folder = sys.argv[1], sys.argv[2]
    {"convert_bins": convert_bins, "naf": naf}[step](folder)
