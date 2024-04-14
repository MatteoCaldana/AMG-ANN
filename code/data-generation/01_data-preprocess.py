import argparse

import sys
sys.path.append('../data-modeling')

import data_preprocessing
data_generation = __import__("00_data-generation")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pooling",
        default="",
        type=str,
        help="Pooling size to parse",
    )
    parser.add_argument(
        "--window",
        default=21,
        type=int,
        help="Savitzky-Golay filter window",
    )
    parser.add_argument(
        "--order",
        default=7,
        type=int,
        help="Savitzky-Golay filter order",
    )
    parser.add_argument(
        "--validation",
        default=0.1,
        type=float,
        help="Validation dataset fraction",
    )
    parser.add_argument(
        "--test",
        default=0.1,
        type=float,
        help="Test dataset fraction",
    )
    parser.add_argument(
        "--apps",
        default="",
        type=str,
        help="Test cases to run, if empty all are run",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Plot and save data",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_arguments()
    apps = data_generation.get_apps(parsed.apps)

    for app in apps:
        print("Preprocessing data for app:", app)
        df = data_preprocessing.preprocess_data(app, parsed.window, parsed.order)
        if parsed.viz:
            data_preprocessing.plot_smoothed_data(df, app)
        dfs = data_preprocessing.split_train_validation_test(
            df, app, parsed.pooling, val=parsed.validation, test=parsed.test
        )
