import os
import json
import random
import datetime
import importlib.util
import argparse
import threading
import pandas as pd

random.seed(0)


def s_print(*a, **b):
    """Thread safe print function"""
    with threading.Lock():
        print(*a, **b)


def import_function_from_file(app, file_path, function_name):
    spec = importlib.util.spec_from_file_location(f"{app}-datagen_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def prepare_data_folder(output_path, app_path):
    os.makedirs(output_path, exist_ok=True)
    os.system(f"rm -r {output_path}/*")

    # find the executable in the build folder
    exec = [
        x
        for x in os.listdir(app_path)
        if (not os.path.isdir(app_path + "/" + x))
        and x.find(".") == -1
        and x != "Makefile"
    ]
    assert len(exec) == 1
    return exec[0]


def run_app(exec, app_path, output_path, jsons, id):
    n = len(jsons)
    for i in range(n):
        s_print(datetime.datetime.now(), f"Thread {id} at iteration {i+1} of {n}")
        conf = jsons[i]
        conf["stats filename"] = f"./{output_path}/stats{id:02d}-{i:04d}.csv"
        conf_path = f"./{output_path}/settings{id:02d}-{i:04d}.json"
        with open(conf_path, "w") as f:
            json.dump(conf, f)
        os.system(
            f"{app_path}/{exec} {conf_path} &> {output_path}/out{id:02d}-{i:04d}.log"
        )


def get_apps(apps):
    if apps == "":
        apps = sorted(
            [
                app
                for app in os.listdir("../data-generation")
                if app.startswith("testcase")
            ]
        )
    else:
        apps = apps.split(",")
    return apps


def build(apps, cleanbuild):
    if cleanbuild:
        print("Clean builing all the testcases...")
        for app in apps:
            os.system(
                f"cd ../data-generation/{app} && "
                "rm -rf build && mkdir build && "
                "cd build && cmake .. && make"
            )
    else:
        print("Skipping clean build")


def generate_data(apps, n_threads, view_size, clean_logs):
    for app in apps:
        print("App: ", app)
        # load the function that defines the simulations to be performed
        settings_generation_function = import_function_from_file(
            app, f"../data-generation/{app}/datagen.py", "create_jsons"
        )
        jsons = settings_generation_function()
        if view_size > 0:
            for i in range(len(jsons)):
                jsons[i]["strong threshold"] = "0,0,0"
                jsons[i]["make view"] = "true"
                jsons[i]["view size"] = view_size
        # prepare folder
        app_path = f"../data-generation/{app}/build/"
        output_path = f"../../data/raw/{app}/"
        if view_size <= 0:
            output_path += "/times"
        else:
            output_path += f"/pooling{view_size}"
        exec = prepare_data_folder(output_path, app_path)
        # launch simulations in parallel
        threads = []
        for i in range(n_threads):
            local_jsons = jsons[i::n_threads]
            print("Launch thread", i, "with", len(local_jsons), "simulations")
            x = threading.Thread(
                target=run_app, args=(exec, app_path, output_path, local_jsons, i)
            )
            threads.append(x)
            x.start()

        for x in threads:
            x.join()

        # compact data in one file
        ls = os.listdir(output_path)
        dfs = []
        for file in ls:
            if file.endswith(".csv"):
                dfs.append(pd.read_csv(f"{output_path}/{file}"))
        pd.concat(dfs).to_csv(f"{output_path}/stats.csv", index=False)

        # clean logs
        if clean_logs:
            os.system(f"rm {output_path}/*.json")
            os.system(f"rm {output_path}/*.log")
            os.system(f"rm {output_path}/*[0-9].csv")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cleanbuild",
        action=argparse.BooleanOptionalAction,
        help="Clean already present build files and compile from scratch",
        default=True,
    )
    parser.add_argument(
        "--cleanlogs",
        action=argparse.BooleanOptionalAction,
        help="Remove temporary log files generated while running simulations",
    )
    parser.add_argument(
        "--nostats",
        default=False,
        action="store_true",
        help="Do not solve the problems (usefull if you want just to do the pooling)",
    )
    parser.add_argument(
        "--pooling",
        default="",
        type=str,
        help="Comma separated list of the pooling sizes to compute (may be empty)",
    )
    parser.add_argument(
        "--apps",
        default="",
        type=str,
        help="Test cases to run, if empty all are run",
    )
    parser.add_argument(
        "-n",
        default=1,
        type=int,
        help="Number of thread for parallelization of computations",
    )
    return parser.parse_args()


def main():
    parsed = parse_arguments()
    print("Generate all the data necessary to run the code")
    print("WARNING: This may take days on a cluster!")
    apps = get_apps(parsed.apps)
    print("Selected apps:", apps)
    print("-----------------------------------------------")
    build(apps, parsed.cleanbuild)
    print("-----------------------------------------------")
    print("Solving problems...")
    # -1 means to solve problems and save time statistics
    view_sizes = [] if parsed.nostats else [-1]
    if parsed.pooling != "":
        view_sizes += [int(n) for n in parsed.pooling.split(",")]
    for view_size in view_sizes:
        generate_data(apps, parsed.n, view_size, parsed.cleanlogs)


if __name__ == "__main__":
    main()
