DATA_PATH = "../../data"
TEST_VARIABLES = {
    "testcase1-diffusion-unstructured": [
        "dim",
        "mesh_ref",
        "degree",
        "sol_id",
        "freq",
        "mesh",
        "renumbering",
        "seed",
        "maxdiff",
    ],
    "testcase2-diffusion-structured": [
        "dim",
        "ndof",
        "mesh_ref",
        "degree",
        "sol_id",
        "sol_pattern_size",
        "epsv",
        "mode",
    ],
    "testcase2.1-diffusion-singlecell": [
        "dim",
        "ndof",
        "mesh_ref",
        "degree",
        "renumbering",
        "diff",
        "mpt",
    ],
    "testcase3-linear-elasticity": [
        "dim",
        "ndof",
        "mesh_ref",
        "degree",
        "seed",
        "mode",
        "pattern_size",
        "max_young",
        "sharp",
        "renumbering",
    ],
}
