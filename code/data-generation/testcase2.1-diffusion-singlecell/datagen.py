import numpy as np
import copy

settings = {
    "dim": 2,
    "degree": 1,
    "dof renumbering": 0,
    "diffusion exp": 0,
    "num ref": 0,
    "marked point": "0,0,0",
    "toll": "1e-9",
    "hermitian": "false",
    "output results": "false",
    "view size": "75",
    "stats filename": "stats.csv",
    "solver mode": "0",
}


def generate_marked_points(dim, ref):
    h = 2 / 2**ref
    ts = np.linspace(h / 2, 1 - h / 2, 2 ** (ref - 1))
    pts = 1e-8 * np.ones((ts.size * dim, 3))
    for i in range(dim):
        for j in range(i + 1):
            pts[ts.size * i : ts.size * (i + 1), j] = ts
    return [f"{p[0]:.16e},{p[1]:.16e},{p[2]:.16e}" for p in pts]


def create_jsons():
    jsons = []
    for dim in [3]:
        settings["dim"] = dim
        for nr in range(2, 7):
            settings["num ref"] = nr
            for dr in [0]:
                settings["dof renumbering"] = dr
                for de in [0, 1, 2, 4, 8]:
                    settings["diffusion exp"] = de
                    for mpt in generate_marked_points(dim, nr):
                        settings["marked point"] = mpt

                        jsons.append(copy.deepcopy(settings))
    return jsons
