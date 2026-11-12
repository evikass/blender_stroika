"""Make the input picture from a sentence with FLUX.2 [klein].

    text_image.py <out.png> <engine dir> <seed> <preset 0/1> <prompt...>

stdout: LOG / STEP <0..1> / DONE <png> / ERROR <msg>
"""
import os
import sys
import traceback


def out(*a):
    print(*a, flush=True)


# the "3D Ready" preset: one whole subject, straight on, nothing around it,
# so the picture can go straight into the 3D engines
STYLE = (" The whole subject large in the frame, filling about 90 percent "
         "of the picture, close framing with only a thin margin around it, "
         "front view, centered, alone on a plain seamless light gray studio "
         "backdrop, no floor, no scenery, no extra objects, no text, soft "
         "even studio light, highly detailed realistic 3D render. If the "
         "subject is a person, character or creature, show its full body "
         "standing straight in a relaxed A-pose with the arms slightly away "
         "from the body. If it is a vehicle, a machine, an object or a "
         "building, show only that thing exactly as described, never as a "
         "character and with no person in it.")


def main():
    path, eng, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    preset = sys.argv[4] == "1"
    text = " ".join(sys.argv[5:]).strip()
    if preset:
        text = text.rstrip(". ") + "." + STYLE
    sys.path.insert(0, os.path.join(eng, "engine_hy"))
    import klein_worker_hy as kw
    torch, pipe = kw.build()
    out("LOG Klein ready")
    pe = kw.prompt_embeds(torch, pipe, text)
    g = torch.Generator("cuda").manual_seed(seed)
    steps = 4

    def step_end(p, i, t, kw_):
        out("STEP %.3f" % ((i + 1) / float(steps)))
        return kw_
    with torch.inference_mode():
        img = pipe(prompt_embeds=pe, height=1536, width=1024,
                   num_inference_steps=steps, generator=g,
                   callback_on_step_end=step_end).images[0]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img.save(path)
    out("DONE " + path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        out("ERROR %s" % str(exc)[:200])
