import sys, time
import numpy as np, torch, cumesh
d = np.load(sys.argv[1])
t = time.time()
v = torch.from_numpy(d["v"]).float().cuda()
f = torch.from_numpy(d["f"]).int().cuda()
cm = cumesh.CuMesh()
cm.init(v, f)
import math
# smaller, even charts: the defaults can leave one huge chart on a very
# detailed model (Hi3D) that xatlas then needs many minutes for
vo, fo, uv, vmap = cm.uv_unwrap(
    compute_charts_kwargs=dict(
        threshold_cone_half_angle_rad=math.radians(60),
        area_penalty_weight=1.0),
    return_vmaps=True)
np.savez(sys.argv[2], vm=vmap.cpu().numpy().astype(np.int64), ix=fo.cpu().numpy().astype(np.int64),
         uv=uv.cpu().numpy().astype(np.float32))
print("uv %.1fs" % (time.time() - t), len(d["f"]), flush=True)
