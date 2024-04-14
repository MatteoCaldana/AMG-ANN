import random
import copy
import math

random.seed(0)

JSON_DEFAULTS_STRUCTURED = {
    "dim": "3",
    "tol": "1e-8",
    "num refinements": "2", 
    "make view": "false",
    "view size": "-1",
    "evaluate errors": "false",
    "output results": "false",
}

def build_epsv_binary(pattern_size, mode):
    epsv = []
    size = pattern_size**mode
    for u in range(size):
        ijk = []
        for t in range(mode - 1, 0, -1):
            ijk.append(u // pattern_size**t)
            u = u % pattern_size**t
        ijk.append(u % pattern_size)
        epsv.append(sum(ijk) % 2)
    return epsv

def create_jsons():
    jsons = []
    a, b = (0, 0)
    settings = copy.deepcopy(JSON_DEFAULTS_STRUCTURED)
    for _ in range(2):
        for deg in [1]:
            settings["deg"] = deg
            for mode0 in range(int(settings["dim"])):
                mode = mode0 + 1
                settings["mode"] = mode
                for pattern_size in [64]:
                    nvals = pattern_size**mode
                    vals = [f'{(b - a) * random.random() + a}' for _ in range(nvals)]
                    settings["epsv"] = ','.join(vals)

                    settings["pattern size"] = pattern_size
                    settings["cycles"] = 8 - deg - int(math.log2(pattern_size))

                    settings["strong threshold"] = "0.05,0.96,0.05"
                    settings["max row sum"] = "0.9,0.9,0.05"
                    settings["symmetric operator"] = "1,1"

                    jsons.append(copy.deepcopy(settings))
    return jsons

