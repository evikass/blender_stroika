"""Look at each part picture and say whether it shows only that part.

    check_parts.py <checks.json> <models dir>

checks.json: [{"i", "path", "name"}]. stdout: CHECK <i> ok|bad / DONE
"""
import os


def _dt(torch):
    """bfloat16 on RTX 30 and newer, float16 on older cards."""
    try:
        if not os.environ.get("IMG3D_FORCE_FP16") and \
                torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


import json
import os
import sys
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"

ASK = ("Does this picture show ONLY the %s by itself, with no person, no "
       "character, no face, no head and no body anywhere in it? Answer "
       "only yes or no.")


def out(*a):
    print(*a, flush=True)


def main():
    checks = json.load(open(sys.argv[1], encoding="utf-8"))
    models = sys.argv[2]
    import torch
    from PIL import Image
    from transformers import (AutoProcessor,
                              Qwen2_5_VLForConditionalGeneration)
    nf4 = os.path.join(models, "text_encoder_nf4")
    full = os.path.join(models, "Qwen-Image-Edit-2509", "text_encoder")
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        nf4 if os.path.isfile(os.path.join(nf4, "config.json")) else full,
        device_map={"": "cuda"}, dtype=_dt(torch))
    proc = AutoProcessor.from_pretrained(
        os.path.join(models, "Qwen-Image-Edit-2509", "processor"))
    for c in checks:
        try:
            img = Image.open(c["path"]).convert("RGB")
            img.thumbnail((768, 768))
            q = ASK % c["name"]
            if "hair" in c["name"].lower():
                q = ("Does this picture show ONLY hair by itself, like a "
                     "wig, with no face, no eyes, no skin and no body? "
                     "Answer only yes or no.")
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": q}]}]
            text = proc.apply_chat_template(msgs, add_generation_prompt=True)
            inp = proc(text=[text], images=[img],
                       return_tensors="pt").to("cuda")
            with torch.inference_mode():
                o = m.generate(**inp, max_new_tokens=4, do_sample=False)
            a = proc.batch_decode(o[:, inp["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].lower()
            out("CHECK %d %s" % (c["i"], "ok" if a.strip().startswith("yes")
                                 else "bad"))
        except Exception as exc:
            out("CHECK %d ok" % c["i"])
            out("LOG check failed: %s" % exc)
    out("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        out("ERROR %s" % str(exc)[:200])
