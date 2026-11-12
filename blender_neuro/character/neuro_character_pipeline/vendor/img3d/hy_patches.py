"""Hunyuan3D-2.1 code: fetched from Tencent, not shipped with the add-on.

Image to 3D does not carry any of Tencent's Hunyuan files. When someone
chooses to install the Hunyuan engine, its code is downloaded straight from
Tencent's own repository (a fixed version, checked file by file) and the
model weights from Tencent's Hugging Face pages. The add-on then makes its
few changes to that code here, as small edits, so it runs on current Python
and on smaller graphics cards. Tencent's LICENSE and Notice.txt come with the
download.

    fetch_code(engine_hy_folder)   download + edit the code
    apply(key, path)               the edits for one downloaded file
"""
import hashlib
import os
import urllib.request

REPO = "Tencent-Hunyuan/Hunyuan3D-2.1"
COMMIT = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
RAW = "https://raw.githubusercontent.com/%s/%s/" % (REPO, COMMIT)

# (path in Tencent's repository, sha256): only the code the engine runs
FILES = [
    ("LICENSE", "b79ac5e11ce063b6c6570dbe9686a45a03ba08bd248aa6aa82fb342a23a81c0c"),
    ("torchvision_fix.py", "6571b4a3228c1b83bd9fc9632bc44d164896a3a75dea59e44d5b9c92e448c4ff"),
    ("Notice.txt", "ffccf6b539a82e6084d14ff064dadd22d33384d6164b07c0c5a3141810df0350"),
    ("hy3dpaint/DifferentiableRenderer/MeshRender.py", "5f008d9149c0dcac2ac5ffa7d2e797657b21515878d9b5804f6bf95c08135cce"),
    ("hy3dpaint/DifferentiableRenderer/__init__.py", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ("hy3dpaint/DifferentiableRenderer/camera_utils.py", "a192b81c6d938dfe50a40f0e248a10c174a9c37b311b41b908f35e5b7304512c"),
    ("hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor.cpp", "bb01d43b06c0a267be6fd8fdf29b1b13637f354a56f06b486b1d794ea4c81ba6"),
    ("hy3dpaint/DifferentiableRenderer/mesh_utils.py", "407ae316ceaf47894efcc8636ccb9b96b079882566b9f081f8885869adac71e2"),
    ("hy3dpaint/LICENSE", "e135813e2ce830c6b46c62bd24e9b6094105c46f825a75341ec6ceb1d915222f"),
    ("hy3dpaint/README.md", "b5ad557f07ff025a8165e5524a716358b1c6048a0e0ab43fbf7d8f16756b8c5e"),
    ("hy3dpaint/cfgs/hunyuan-paint-pbr.yaml", "6fdb8942da866d67c0ff7737d2aa513a2f08a4fb0cc1ce310e9f2c19a9deb7df"),
    ("hy3dpaint/convert_utils.py", "0adc9c2b2245132c9318253cf1a8ff79c791be455a2149f59320e3ba4ef052dc"),
    ("hy3dpaint/demo.py", "c37f98c7fbb6fe8ece0e11ae46ccaa6d07996f6c81c4c818ad7b1bd5839c0d6f"),
    ("hy3dpaint/hunyuanpaintpbr/__init__.py", "13a4e0a42826babc68abbe2634e35d40af4dfe0156c3bae078fc6b6139eea6df"),
    ("hy3dpaint/hunyuanpaintpbr/pipeline.py", "0ce6e0cab54e1f73ef3b606c7b66a12b59207e80df1dfedce4c34ebb5ebef8f5"),
    ("hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py", "a41bf0f79864691bc831cc2de9714f6d25baf73463ad3fdcb224dc09fe2e6705"),
    ("hy3dpaint/hunyuanpaintpbr/unet/model.py", "5d135a73c12f0d7765366a72a37264b6bcb15da1551d3e7d60526d04b5eb1dc0"),
    ("hy3dpaint/hunyuanpaintpbr/unet/modules.py", "664983290a63849503c64714948d97444400bd95f24ad67e3a5dc557ba5a7a68"),
    ("hy3dpaint/textureGenPipeline.py", "b940f03354fee976229f8acf2007517ab4ca6e6029e69072fb3bcbe0490622e8"),
    ("hy3dpaint/utils/__init__.py", "9f1da29263d33f2791aecea183b3cbaceadae13a78a701db0b673a56c3c0f75b"),
    ("hy3dpaint/utils/image_super_utils.py", "3688f49deb54b13f1d65c15e07ff42f3ef3b4b5934b73011cf3dc9478d09280d"),
    ("hy3dpaint/utils/multiview_utils.py", "ad90cf21adb8fc225b6cd77cb49fa76b12ded97d122566a74252e92208250f99"),
    ("hy3dpaint/utils/pipeline_utils.py", "a479de70e3d90a6060cf2fd03731e9e29af988880f3adaa5fc4643dd3573bd08"),
    ("hy3dpaint/utils/simplify_mesh_utils.py", "946764073baa70854561392d710ab1fca7d4321d05fbba3dd6be8a446e7acf6d"),
    ("hy3dpaint/utils/torchvision_fix.py", "5e2133d574ed1f6386e11e30ffe960ff0687df9f675545a0d78454cf2ea5272a"),
    ("hy3dpaint/utils/uvwrap_utils.py", "52936fbcc1719c6a6b2e424d9177ac19e106489b4c958c23605ab7f44bcd233b"),
    ("hy3dshape/.gitignore", "f47adfe4050282956c3ef7b626c9d6468d1f3c0f9f148d2df24c1c544564c92e"),
    ("hy3dshape/LICENSE", "e135813e2ce830c6b46c62bd24e9b6094105c46f825a75341ec6ceb1d915222f"),
    ("hy3dshape/NOTICE", "bd1f58f4b2943f31b010a6164d977395f6f86fbd220c111d189128ff0e5709a8"),
    ("hy3dshape/README.md", "6e6aba49700a9accc386d618db7b2884bcfd077938011023c700ccbd86479ceb"),
    ("hy3dshape/configs/hunyuandit-finetuning-flowmatching-dinol518-bf16-lr1e5-4096.yaml", "b6c69a712e68ace5bc94fd03266c6db92abc08d05d16e281607b24dc0d3f7266"),
    ("hy3dshape/configs/hunyuandit-mini-overfitting-flowmatching-dinol518-bf16-lr1e4-4096.yaml", "dafa537ecfb0faef476bdee5b2f02b3248310f89f884cd2f2c03d1743a6d3080"),
    ("hy3dshape/hy3dshape/__init__.py", "d233d923a912621c986201e482b1c7b9a5fc014974e6bb952ceb7b36f3c4020f"),
    ("hy3dshape/hy3dshape/data/dit_asl.py", "ff5c8c9b57fec20998783100fb5fb4f5e4c57943afcafb0d9f6a985803e0eb87"),
    ("hy3dshape/hy3dshape/data/utils.py", "a2c24c5243f0cacfd323ab969bd9ab26928e54393ce439eaf6ab4de73eb951fb"),
    ("hy3dshape/hy3dshape/models/__init__.py", "2f56106c91dfc126f3e20551e8a7a48f049be7b33eda814ae1c184bc979328fe"),
    ("hy3dshape/hy3dshape/models/autoencoders/__init__.py", "f1c91751fa8db7ecf531189aae33a59b5b9e3e10aa0d524edf33f6857eea8cfd"),
    ("hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py", "1cbd34f21102a33eed2106e445ec6a96b896b80e187564f40e2029779bb2cba8"),
    ("hy3dshape/hy3dshape/models/autoencoders/attention_processors.py", "9a461ddde9a2bf5ca0cd3961c3c1b86f9c81772066fe5b815bc497e0b5a8f053"),
    ("hy3dshape/hy3dshape/models/autoencoders/model.py", "b81a4586cc1148def7fe157c18071fb7dc50753cfbf1930932bbcbecbcb35829"),
    ("hy3dshape/hy3dshape/models/autoencoders/surface_extractors.py", "203d1551ccef37e86f9809b9ac0fac68e8e4a2f633ca72a85408535595ae68e5"),
    ("hy3dshape/hy3dshape/models/autoencoders/volume_decoders.py", "95ffab6bd0a11034bc2b7415fef1379e27c4d2238638f831a0456019d90484eb"),
    ("hy3dshape/hy3dshape/models/conditioner.py", "137434197ff42e337cfb83fb67c317cce42c5b73c256bec5744af1497dc98cde"),
    ("hy3dshape/hy3dshape/models/denoisers/__init__.py", "8ab24870a628fb94a3d903d7d149d3eb3d20b32cadf8f6680c6e457562384a41"),
    ("hy3dshape/hy3dshape/models/denoisers/hunyuan3ddit.py", "0e47af4644344c0900adb216164b3be26b2d97085017711a72f27fda66f2b755"),
    ("hy3dshape/hy3dshape/models/denoisers/hunyuandit.py", "cb4d9f3504dea0ed3eba94963a8efd7bf92221b26613a7c8bba58e4ef7b6cd25"),
    ("hy3dshape/hy3dshape/models/denoisers/moe_layers.py", "a8d756a5fbe6941ad0dd68e4d1b0ea4fc6a84418f4c2e2b0a28cb4f2144ec240"),
    ("hy3dshape/hy3dshape/models/diffusion/flow_matching_sit.py", "32925fd014a147825a002edcd07e84b1e616b1da3ab0081ab76401f5c5bc8e3a"),
    ("hy3dshape/hy3dshape/models/diffusion/transport/__init__.py", "3a84f703c88bc0ad9b4be89cd2cb880e59763ef59548fc2d32ac35eab0a8ed42"),
    ("hy3dshape/hy3dshape/models/diffusion/transport/integrators.py", "7edc1ed10518ccb0527ccf59deccbdfc4ddc14221c8299cac859a1e242094e2a"),
    ("hy3dshape/hy3dshape/models/diffusion/transport/path.py", "16321f9f76ab0385ec7433e8500453a5bebc46029a87966926f961d641a64187"),
    ("hy3dshape/hy3dshape/models/diffusion/transport/transport.py", "749ab441b9a301d50c323ebed7b2465a6dded439fe41fa8a1c0d30847b9ac90f"),
    ("hy3dshape/hy3dshape/models/diffusion/transport/utils.py", "28aa55be314c180a57358215adb8cb0abf9d9a6e1992fb841f90578fde782ee0"),
    ("hy3dshape/hy3dshape/pipelines.py", "80d3d66be03e2ebe2f988cefcf488f2e5fa52656e8547c72dd44aaae4ad45714"),
    ("hy3dshape/hy3dshape/postprocessors.py", "3899ef49fbc7453553d264d16c14305a9ee9c6f8a3341a8850b960f018640a17"),
    ("hy3dshape/hy3dshape/preprocessors.py", "284dc24fabbf974bd6793ebd58e95b7700550d90371cacb96ad0889ec558de1e"),
    ("hy3dshape/hy3dshape/rembg.py", "e372426c19bc51447129b72862e330777e7ce4a9e07212c1abdff5d1947292dc"),
    ("hy3dshape/hy3dshape/schedulers.py", "d3b25b84dac26f0e862cfc866da08df2d6592f01a0aaeda965782ab0fc1d8bba"),
    ("hy3dshape/hy3dshape/surface_loaders.py", "c4a821fffd9ea9f121ac07a034faaa9e52350859cc667db955ae91c17ee1c001"),
    ("hy3dshape/hy3dshape/utils/__init__.py", "0be86a4b0a9c7885550f80fd6f9900855246f6fb0d8df6000e07414258d6590f"),
    ("hy3dshape/hy3dshape/utils/ema.py", "cadb5b3a3ac8fb23a8fb3554fac3ac45c147f99d765cfa1f0af0db5f8ac9476c"),
    ("hy3dshape/hy3dshape/utils/misc.py", "12d3378fa71c2569897ac27f8dbf3dcd3857b25f8d008c564e6b8e1df73ea3d0"),
    ("hy3dshape/hy3dshape/utils/trainings/__init__.py", "8b084a9f378124b2b1a51563bf3c22444eb7ff33692017da28b2136ae56987ed"),
    ("hy3dshape/hy3dshape/utils/trainings/callback.py", "e829e12bc17212dd8e343b4a2b263ae46d15755dd7aca0ca8f633a1740ff7e1d"),
    ("hy3dshape/hy3dshape/utils/trainings/lr_scheduler.py", "48462dc6c67b5ecfcdf15d780ffffa728b13b53b6d094e38b7402d45aa47e201"),
    ("hy3dshape/hy3dshape/utils/trainings/mesh.py", "c913ae260f7c535e38a501a2c17031e027884545cd8842fbe2953d8f64aa5656"),
    ("hy3dshape/hy3dshape/utils/trainings/mesh_log_callback.py", "adfa2c81b7167bf617173907cc1dd6f022cc88406a9bc0d2176ed61271f096e5"),
    ("hy3dshape/hy3dshape/utils/trainings/peft.py", "d9975eb27fc688f82265a6d116c80dd3de0dbadf0c6141539f73bcb5448a02cf"),
    ("hy3dshape/hy3dshape/utils/utils.py", "09fda07cb7e7aafbbc2d0a0a808d312d6687bcb1a939c33cdf9f88dc07d42ca0"),
    ("hy3dshape/hy3dshape/utils/visualizers/__init__.py", "8b084a9f378124b2b1a51563bf3c22444eb7ff33692017da28b2136ae56987ed"),
    ("hy3dshape/hy3dshape/utils/visualizers/color_util.py", "4805fade6c1ea57fa8ef661365cdf4bdad923ca9d58dcc9a448b3bc99e037f73"),
    ("hy3dshape/hy3dshape/utils/visualizers/html_util.py", "1ad73fe86a126c76f88d7a91bf3743788601801a3b53bf9f6439df7726ec6af8"),
    ("hy3dshape/hy3dshape/utils/visualizers/pythreejs_viewer.py", "3cb87aac8493e2a22c8a240b4a5d6a4f4869355ef4433f7a25313c406643de24"),
    ("hy3dshape/main.py", "a08768d20f6b1b1ca3c2aeaac118dc80dbd554e2b67fb77db1cfab430e1f261e"),
    ("hy3dshape/minimal_demo.py", "bfb273fe83f5c5cc11dbb228b58218cce80e00b478f539898075bcc35287b502"),
    ("hy3dshape/minimal_demo_with_ckpt.py", "ed876168a34f346eb470a91e244bf240ad30724827d5cf78b4743d2b092bccd1"),
    ("hy3dshape/minimal_vae_demo.py", "6f45195fc12a5fb5adb3574a39c86d010ff75a4ddcbf02a1d074d73d07bfaeb9"),
    ("hy3dshape/requirements.txt", "b09104f55ee2f22438b2944b164125e88a0366a3434d703aa3f33115e1b21632")
]

# the add-on's own files that stand in for two of Tencent's: they replace a
# compiled part and a library that no longer builds on current Python
OWN = {
    "hy3dpaint/utils/image_super_utils.py": "image_super_utils.py",
    "hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor.py":
        "mesh_inpaint_processor.py",
}

_LEAN = """

def _small_pc():
    if os.environ.get("IMG3D_VRAM_GB") or os.environ.get("IMG3D_LOWMEM"):
        return True
    try:
        if torch.cuda.get_device_properties(0).total_memory < 10.5 * 2**30:
            return True
        import psutil
        return psutil.virtual_memory().total < 28 * 2**30
    except Exception:
        return False


def _lean_unet(path, torch_dtype):
    \"\"\"Build the painter as an empty frame straight on the graphics card in
    half precision and fill it weight by weight from the file. The stock
    loader builds it in float32 twice over in main memory first (11 GB).\"\"\"
    st = os.path.join(path, "diffusion_pytorch_model.safetensors")
    if not (_small_pc() and os.path.isfile(st) and torch.cuda.is_available()):
        return None
    import json as _json
    from safetensors import safe_open
    with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
        config = _json.load(f)
    with torch.device("meta"):
        unet = UNet2DConditionModel(**config)
        u = UNet2p5DConditionModel(unet)
        u.unet.conv_in = torch.nn.Conv2d(
            12, unet.conv_in.out_channels,
            kernel_size=unet.conv_in.kernel_size, stride=unet.conv_in.stride,
            padding=unet.conv_in.padding, dilation=unet.conv_in.dilation,
            groups=unet.conv_in.groups, bias=unet.conv_in.bias is not None)
    u = u.to(torch_dtype).to_empty(device="cuda")
    sd = u.state_dict()
    with safe_open(st, framework="pt", device="cuda") as f:
        for k in f.keys():
            sd[k].copy_(f.get_tensor(k))
    del sd
    return u
"""

# each edit: (text to find, text to put instead); "+END" appends
EDITS = {
    # the painter: half precision DINO, the lean loader, .safetensors weights
    "paint_modules": [
        ("        self.dino_v2 = AutoModel.from_pretrained(dino_v2_path)\n",
         "        self.dino_v2 = AutoModel.from_pretrained(dino_v2_path,\n"
         "                                                 torch_dtype=torch.float16)\n"),
        ("        torch_dtype = kwargs.pop(\"torch_dtype\", torch.float32)\n",
         "        torch_dtype = kwargs.pop(\"torch_dtype\", torch.float32)\n"
         "        lean = _lean_unet(pretrained_model_name_or_path, torch_dtype)\n"
         "        if lean is not None:\n"
         "            return lean\n"),
        ("        unet_ckpt = torch.load(unet_ckpt_path, map_location=\"cpu\", weights_only=True)\n",
         "        st_path = os.path.join(pretrained_model_name_or_path,\n"
         "                               \"diffusion_pytorch_model.safetensors\")\n"
         "        if os.path.isfile(st_path):\n"
         "            from safetensors.torch import load_file\n"
         "            unet_ckpt = load_file(st_path, device=\"cpu\")\n"
         "        else:\n"
         "            unet_ckpt = torch.load(unet_ckpt_path, map_location=\"cpu\",\n"
         "                                   weights_only=True)\n"),
        ("+END", _LEAN),
    ],
    # runs outside Blender: bpy is only needed for Tencent's own exporter
    "mesh_utils": [
        ("import bpy\n",
         "try:\n    import bpy\nexcept ImportError:\n    bpy = None\n"),
    ],
    # the weights are already on disk: no download from inside the engine
    "multiview": [
        ("        model_path = huggingface_hub.snapshot_download(\n"
         "            repo_id=config.multiview_pretrained_path,\n"
         "            allow_patterns=[\"hunyuan3d-paintpbr-v2-1/*\"],\n"
         "        )\n",
         "        model_path = config.multiview_pretrained_path\n"
         "        if not os.path.isdir(model_path):\n"
         "            model_path = huggingface_hub.snapshot_download(\n"
         "                repo_id=config.multiview_pretrained_path,\n"
         "                allow_patterns=[\"hunyuan3d-paintpbr-v2-1/*\"],\n"
         "            )\n"),
    ],
    # Hunyuan3D-2mv's settings name the 2.0 code: point them at 2.1's
    "mv_config": [
        ("hy3dgen.shapegen.models.Hunyuan3DDiT",
         "hy3dshape.models.denoisers.hunyuan3ddit.Hunyuan3DDiT"),
        ("hy3dgen.shapegen.models.ShapeVAE",
         "hy3dshape.models.autoencoders.ShapeVAE"),
        ("hy3dgen.shapegen.models.SingleImageEncoder",
         "hy3dshape.models.conditioner.SingleImageEncoder"),
        ("hy3dgen.shapegen.schedulers.FlowMatchEulerDiscreteScheduler",
         "hy3dshape.schedulers.FlowMatchEulerDiscreteScheduler"),
        ("hy3dgen.shapegen.preprocessors.MVImageProcessorV2",
         "hy3dshape.preprocessors.MVImageProcessorV2"),
        ("hy3dgen.shapegen.pipelines.Hunyuan3DDiTFlowMatchingPipeline",
         "hy3dshape.pipelines.Hunyuan3DDiTFlowMatchingPipeline"),
    ],
}

# which downloaded file gets which edits
CODE_EDITS = {
    "hy3dpaint/hunyuanpaintpbr/unet/modules.py": "paint_modules",
    "hy3dpaint/DifferentiableRenderer/mesh_utils.py": "mesh_utils",
    "hy3dpaint/utils/multiview_utils.py": "multiview",
}


def apply(key, path):
    """Make the add-on's edits to one downloaded file (again is harmless)."""
    raw = open(path, "rb").read()
    crlf = b"\r\n" in raw
    s = raw.decode("utf-8").replace("\r\n", "\n")
    for old, new in EDITS[key]:
        if old == "+END":
            if new.strip() not in s:
                s = s.rstrip("\n") + "\n" + new
            continue
        if new in s:
            continue
        if old not in s:
            raise RuntimeError("Hunyuan file %s is not the expected version"
                               % os.path.basename(path))
        s = s.replace(old, new, 1)
    if crlf:
        s = s.replace("\n", "\r\n")
    tmp = path + ".edit"
    with open(tmp, "wb") as fh:
        fh.write(s.encode("utf-8"))
    os.replace(tmp, path)


def _get(url, tries=4):
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Image-to-3D-Blender-addon"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as exc:
            last = exc
    raise RuntimeError("could not download %s: %s" % (url, last))


def fetch_code(dst, say=None, own_dir=None):
    """Download Tencent's Hunyuan3D-2.1 code into dst (the engine_hy folder),
    check every file, then make the add-on's edits."""
    import shutil
    n = len(FILES)
    for i, (rel, sha) in enumerate(FILES):
        p = os.path.join(dst, *rel.split("/"))
        if os.path.isfile(p) and rel not in CODE_EDITS and \
                hashlib.sha256(open(p, "rb").read()).hexdigest() == sha:
            continue
        data = _get(RAW + rel)
        if hashlib.sha256(data).hexdigest() != sha:
            raise RuntimeError("Hunyuan file %s did not download correctly"
                               % rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(data)
        if say:
            say(i / float(n), "Downloading Hunyuan code from Tencent")
    for rel, key in CODE_EDITS.items():
        apply(key, os.path.join(dst, *rel.split("/")))
    own_dir = own_dir or os.path.join(dst, "hy_own")
    for rel, name in OWN.items():
        s = os.path.join(own_dir, name)
        if os.path.isfile(s):
            shutil.copyfile(s, os.path.join(dst, *rel.split("/")))
    open(os.path.join(dst, ".hunyuan_code"), "w").write(COMMIT)
