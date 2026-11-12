"""FLUX.2 [klein] 4B worker, its own process: it needs a newer diffusers than
the Qwen engine, so it gets its own copy (klein_libs) and the main engine
talks to it a job per line.

stdin: JSON job per line {"image","out","text","size","seed"}
stdout: READY / STEP <0..1> / DONE <png> / ERROR <msg> / LOG <text>
"""
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AIR_ROOT", "")
if ROOT and os.path.isdir(os.path.join(ROOT, "klein_libs")):
    sys.path.insert(0, os.path.join(ROOT, "klein_libs"))
os.environ["HF_HUB_OFFLINE"] = "1"
# the add-on downloads Klein into the engine's own models folder; older
# set-ups keep it with AI Render
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models", "black-forest-labs", "FLUX.2-klein-4B")
if not os.path.isfile(os.path.join(D, "model_index.json")):
    D = os.path.join(ROOT, "models", "flux2-klein-4b")


def out(*a):
    print(*a, flush=True)


def small_pc():
    import torch
    if os.environ.get("IMG3D_VRAM_GB") or os.environ.get("IMG3D_LOWMEM"):
        return True
    try:
        if torch.cuda.get_device_properties(0).total_memory < 10.5 * 2**30:
            return True
        import psutil
        return psutil.virtual_memory().total < 28 * 2**30
    except Exception:
        return False


def load_te():
    import torch
    from transformers import Qwen3ForCausalLM, BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16)
    return Qwen3ForCausalLM.from_pretrained(os.path.join(D, "text_encoder"),
                                            quantization_config=q,
                                            device_map={"": "cuda"},
                                            dtype=torch.bfloat16)


def lean_transformer():
    """The image model as an empty 8-bit frame on the card, filled weight by
    weight from the file: the normal load makes a 8 GB bf16 copy in main
    memory first."""
    import torch
    from safetensors import safe_open
    from diffusers import Flux2Transformer2DModel
    td = os.path.join(D, "transformer")
    cfg = Flux2Transformer2DModel.load_config(td)
    with torch.device("meta"):
        tr = Flux2Transformer2DModel.from_config(cfg)
    tr = tr.to(torch.bfloat16)
    tr.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn,
                                compute_dtype=torch.bfloat16)
    tr = tr.to_empty(device="cuda")
    sd = tr.state_dict()
    with safe_open(os.path.join(td, "diffusion_pytorch_model.safetensors"),
                   framework="pt", device="cuda") as f:
        for k in f.keys():
            sd[k].copy_(f.get_tensor(k))
    del sd
    return tr.eval()


def build():
    import torch
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
    small = small_pc()
    if small:
        # small PC: the image model straight onto the card in 8 bits; the
        # prompt model is loaded only when a new prompt has to be read
        # (its result is kept on disk, so normally never)
        tr = lean_transformer()
        te = None
    else:
        # the prompt model in 4 bits and the image model stored in 8 bits:
        # both stay on a 12 GB card together
        te = load_te()
        tr = Flux2Transformer2DModel.from_pretrained(
            os.path.join(D, "transformer"), torch_dtype=torch.bfloat16)
        if os.environ.get("KLEIN_FP8", "1") == "1":
            tr.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn,
                                        compute_dtype=torch.bfloat16)
        tr.to("cuda")
    pipe = Flux2KleinPipeline.from_pretrained(D, text_encoder=te,
                                              transformer=tr,
                                              torch_dtype=torch.bfloat16)
    pipe.vae.to("cuda")
    pipe._small = small
    type(pipe)._execution_device = property(
        lambda self: torch.device("cuda"))
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    pipe._te_swap = False
    if te is not None:
        # the prompt model waits in main memory between pictures
        try:
            te.to("cpu")
            torch.cuda.empty_cache()
            pipe._te_swap = True
        except Exception as exc:
            out("LOG prompt model stays on the card: %s" % exc)
    return torch, pipe


PE_DIR = os.path.join(os.path.dirname(HERE), "models", "klein_prompts")


def prompt_embeds(torch, pipe, text):
    import hashlib
    cache = getattr(pipe, "_pe_cache", None)
    if cache is None:
        cache = pipe._pe_cache = {}
    if text in cache:
        return cache[text]
    path = os.path.join(PE_DIR, hashlib.sha1(text.encode("utf-8"))
                        .hexdigest() + ".pt")
    pe = None
    if os.path.isfile(path):
        try:
            pe = torch.load(path, map_location="cuda", weights_only=True)
        except Exception:
            pe = None
    if pe is None:
        with torch.inference_mode():
            loaded = pipe.text_encoder is None
            if loaded:
                pipe.text_encoder = load_te()
            elif pipe._te_swap:
                pipe.text_encoder.to("cuda")
            try:
                pe = pipe.encode_prompt(prompt=text, device="cuda")
            finally:
                if loaded:
                    pipe.text_encoder = None
                    import gc
                    gc.collect()
                elif pipe._te_swap:
                    pipe.text_encoder.to("cpu")
                torch.cuda.empty_cache()
            pe = pe[0] if isinstance(pe, tuple) else pe
        try:
            os.makedirs(PE_DIR, exist_ok=True)
            torch.save(pe.detach().cpu(), path)
        except Exception:
            pass
    cache.clear()
    cache[text] = pe
    return pe


def run(torch, pipe, job):
    from PIL import Image
    import gc
    pass
    _t0 = time.time()
    img = Image.open(job["image"]).convert("RGB")
    w, h = img.size
    mp = {"DRAFT": 0.5, "HIGH": 1.0, "ULTRA": 1.6}.get(job.get("size"), 1.0)
    area = mp * 1024 * 1024
    W = max(256, int(round(math.sqrt(area * w / float(h)) / 16)) * 16)
    H = max(256, int(round(math.sqrt(area * h / float(w)) / 16)) * 16)
    src = img.resize((W, H), Image.LANCZOS)
    steps = 4

    def step_end(p, i, t, kw):
        out("STEP %.3f" % ((i + 1) / float(steps)))
        return kw
    # every tile of an enhance has the same prompt: read it once (and keep
    # it on disk, so the prompt model is rarely needed at all)
    pe = prompt_embeds(torch, pipe, job["text"])
    g = torch.Generator("cuda").manual_seed(int(job.get("seed", 0)))
    with torch.inference_mode():
        res = pipe(prompt_embeds=pe, image=[src], height=H, width=W,
                   num_inference_steps=steps, generator=g,
                   callback_on_step_end=step_end).images[0]
    res = res.resize((w, h), Image.LANCZOS)
    os.makedirs(os.path.dirname(os.path.abspath(job["out"])), exist_ok=True)
    res.save(job["out"])
    out("LOG tile %.1fs" % (time.time() - _t0))
    out("DONE " + os.path.abspath(job["out"]))


def main():
    t = time.time()
    try:
        torch, pipe = build()
    except Exception as exc:
        traceback.print_exc()
        out("ERROR could not load Klein: %s" % str(exc).splitlines()[-1][:160]
            if str(exc).strip() else "ERROR could not load Klein")
        return
    out("LOG loaded in %.1f s" % (time.time() - t))
    out("READY")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            break
        if line == "PARK":
            # wait in main memory between jobs: the card is free for the
            # other engines, and the next job skips loading from disk
            try:
                pipe.transformer.to("cpu")
                pipe.vae.to("cpu")
            except Exception as exc:
                out("LOG park: %s" % exc)
            # a prompt model that could not move to main memory (4-bit) is
            # dropped: prompts are kept on disk, so it is rarely needed
            try:
                te_ = pipe.text_encoder
                if te_ is not None and next(te_.parameters()).device.type \
                        == "cuda":
                    pipe.text_encoder = None
                    pipe._te_swap = False
                    del te_
            except Exception as exc:
                out("LOG park te: %s" % exc)
            try:
                getattr(pipe, "_pe_cache", {}).clear()
            except Exception:
                pass
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            out("PARKED %d" % (torch.cuda.memory_reserved() // 2**20))
            continue
        try:
            if next(pipe.transformer.parameters()).device.type != "cuda":
                pipe.transformer.to("cuda")
                pipe.vae.to("cuda")
            run(torch, pipe, json.loads(line))
        except Exception as exc:
            traceback.print_exc()
            msg = str(exc).strip().splitlines()[-1] if str(exc).strip() \
                else type(exc).__name__
            if "out of memory" in msg.lower():
                msg = "Not enough GPU memory - try Size: Draft"
            out("ERROR " + msg[:200])
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    main()
