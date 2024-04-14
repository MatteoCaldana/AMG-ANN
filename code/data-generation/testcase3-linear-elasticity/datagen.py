import random
import copy

random.seed(0)

JSON_DEFAULTS_ELASTICITY = {
    "dim": "3",
    "num refinements": "3",
    "tol": "1e-8",
    "make view": "false",
    "view size": "-1",
    "output results": "false",
    "evaluate errors": "false",
}

def create_jsons():
    json_defaults = JSON_DEFAULTS_ELASTICITY.copy()
    jsons = []

    for deg in [1, 2, 3]:
        json_defaults["deg"] = deg
        json_defaults["cycles"] = 6 - deg
        for mode in [1, 2, 3]:
            json_defaults["mode"] = mode
            for ps in [2, 4, 8]:
                json_defaults["pattern size"] = ps
                for sharp in ["true"]:
                    json_defaults["sharp"] = sharp
                    for max_diff in [1, 2, 4, 8]:
                        json_defaults["max young exp"] = max_diff
                        json_defaults["strong threshold"] = "0.15,0.91,0.025"
                        for renumbering in range(4):
                            json_defaults["renumbering"] = renumbering
                            for _ in range(20):
                                json_defaults["seed"] = random.randint(0, 2**31)
                                jsons.append(copy.deepcopy(json_defaults))
    return jsons