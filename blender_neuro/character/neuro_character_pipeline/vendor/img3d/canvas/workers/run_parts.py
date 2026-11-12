"""Make the part pictures, check them, and make the failed ones again.

    run_parts.py <jobs.json> <engine dir> <vision root>

Runs extract_parts.py (Klein), then check_parts.py (the vision model) on
the results; a picture that still shows the character, a face or a body is
made once more with a stricter instruction and a new seed.
stdout: PART <i> <png> / FAIL <i> <msg> / LOG / ALLPARTS / ERROR <msg>
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NOWIN = 0x08000000 if os.name == "nt" else 0


def out(*a):
    print(*a, flush=True)


def run(cmd, on_line):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace",
                         creationflags=NOWIN)
    for line in p.stdout:
        on_line(line.rstrip("\n"))
    p.wait()


def main():
    jp, eng, vroot = sys.argv[1], sys.argv[2], sys.argv[3]
    jobs = json.load(open(jp, encoding="utf-8"))
    klein = os.path.join(eng, "env_klein", "Scripts", "python.exe")
    done = {}

    redo_set = set()
    checked = set(i for i, p in enumerate(jobs["parts"])
                  if p.get("kind") == "part") if vroot else set()

    def ext_line(line, index_map):
        if line.startswith("PART "):
            i = index_map[int(line.split()[1])]
            done[i] = jobs["parts"][i]["out"]
            # shown right away; the check comes after
            out("PART %d %s" % (i, done[i]))
            out("STATE %d %s" % (i, "checking" if i in checked and
                                 i not in redo_set else "ok"))
        elif line.startswith("FAIL "):
            i = index_map[int(line.split()[1])]
            out("FAIL %d %s" % (i, line.split(" ", 2)[-1]))
        elif line.startswith(("LOG", "ERROR")):
            out(line)

    run([klein, "-u", os.path.join(HERE, "extract_parts.py"), jp, eng],
        lambda l: ext_line(l, list(range(len(jobs["parts"])))))
    # check the plain parts (not the body, the bald head or the views)
    checks = [{"i": i, "path": done[i], "name": p["name"]}
              for i, p in enumerate(jobs["parts"])
              if i in done and p.get("kind") == "part"]
    bad = []
    if checks and vroot:
        cp = jp[:-5] + "_check.json"
        with open(cp, "w", encoding="utf-8") as fh:
            json.dump(checks, fh)
        py = os.path.join(vroot, "env_qwen", "Scripts", "python.exe")

        def chk(line):
            if line.startswith("CHECK ") and line.endswith("bad"):
                bad.append(int(line.split()[1]))
        run([py, "-u", os.path.join(HERE, "check_parts.py"), cp,
             os.path.join(vroot, "models")], chk)
    for i in sorted(done):
        if i in checked:
            out("STATE %d %s" % (i, "redo" if i in bad else "ok"))
    if bad:
        out("LOG again: %s" % ", ".join(jobs["parts"][i]["name"]
                                         for i in bad))
        again = dict(jobs, parts=[dict(jobs["parts"][i], strict=True)
                                  for i in bad], seed=int(jobs.get(
                                      "seed", 0)) + 17)
        jp2 = jp[:-5] + "_again.json"
        with open(jp2, "w", encoding="utf-8") as fh:
            json.dump(again, fh, indent=1)
        done.clear()
        redo_set.update(bad)
        run([klein, "-u", os.path.join(HERE, "extract_parts.py"), jp2, eng],
            lambda l: ext_line(l, bad))
        for i in bad:
            if i not in done:
                out("STATE %d failed" % i)
    out("ALLPARTS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        out("ERROR %s" % str(exc)[:200])
