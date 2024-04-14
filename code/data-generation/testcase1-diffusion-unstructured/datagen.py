import random
import copy

random.seed(0)

JSON_DEFAULTS_UNSTRUCTURED = {
    "dim": "3",
    "toll": "1e-8",
    "make view": "false",
    "view size": "-1",
    "evaluate errors": "false",
    "output results": "false",
    "solution id": "0",
    "solution freq": "1",
}


def create_jsons():
    json_defaults = JSON_DEFAULTS_UNSTRUCTURED.copy()
    jsons = []

    for deg in [1, 2, 3]:
        json_defaults["degree"] = deg
        for mesh in ["Simplex", "PlateWithHole.2", "HyperBall", "Cylinder.2", "Cube"]:
            json_defaults["mesh filename"] = mesh
            for renumbering in range(4):
                json_defaults["dof renumbering"] = renumbering
                for base_ref in [0, 1, 2]:
                    json_defaults["num base ref"] = base_ref
                    json_defaults["ncycles"] = 8 - deg - base_ref
                    for max_diff in [1, 3, 10]:
                        json_defaults["max diffusion exp"] = max_diff
                        json_defaults["strong threshold"] = "0.05,0.96,0.025"
                        for _ in range(20):
                            json_defaults["random seed"] = random.randint(0, 2**31)
                            jsons.append(copy.deepcopy(json_defaults))
    return jsons
