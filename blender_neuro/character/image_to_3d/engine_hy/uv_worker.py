import sys
import numpy as np
import xatlas
d = np.load(sys.argv[1])
vm, ix, uv = xatlas.parametrize(d["v"], d["f"])
np.savez(sys.argv[2], vm=vm, ix=ix, uv=uv)
