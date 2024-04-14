#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import argparse
import platform
import os

import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import count_params

from cnn_models import generic_build_model, mse_mvar, mse_var, mse
import constants

print("TF version:", tf.__version__)
print("inter threads:", tf.config.threading.get_inter_op_parallelism_threads())
print("intra threads:", tf.config.threading.get_intra_op_parallelism_threads())


np.random.seed(0)
tf.random.set_seed(0)
tf.keras.utils.set_random_seed(0)

VIEW_TYPE_CHANNELS = {"sum": 1, "max": 1, "max_ppnp": 2, "sum+max": 3, "sum+max+c": 4}
VERBOSE = 1 if platform.system() == "Windows" else 1
NORMALIZATION_MODES = [
    "pure",
    "resc",
    "pure_log",
    "resc_log",
    "nothing",
    "mean",
    "dataset_pure",
    "dataset_pure_log",
    "gaussian",
    "resc_gaussian",
]
UNBOUNDED_NORMALIZATION_MODES = ["nothing", "mean", "gaussian", "resc_gaussian"]
VIEW_TYPES = ["", "_max_pp", "_max_np"]
NORM_MODES_COUNT = ["pure", "pure_log"]
TO_DROP = [
    "Unnamed: 0",
    "setting",
    "timestamp",
    "maxrowsum",
    "symop",
    "tol",
    "t_amg_setup",
    "nrows",
    "nze",
    "sparsity",
    "grid",
    "operator",
    "memory",
    "t_solve",
    "p_res",
    "version",
    "res",
    "rhov",
]


def load_time_stats(path, parse_epsv=True):
    print(f"Read: {path}/stats.csv.gz")
    df = pd.read_csv(f"{path}/stats.csv.gz")
    print("Dropping cols")
    df = df.drop(list(set(df.columns) & set(TO_DROP)), axis=1)

    if parse_epsv:
        print("Parsing epsv")
        df["epsv"] = df["epsv"].apply(
            lambda x: tuple([float(t) for t in x[1:-1].split(", ")])
        )
    print("Reading done")
    return df


def parse_view(field, dtype):
    def _parse_view(row):
        nums = np.array([dtype(x) for x in row[field].split(",")], dtype=dtype)
        assert len(nums) == row["view_size"] ** 2, "view wrong size"
        return nums.reshape((row["view_size"],) * 2)

    return _parse_view


def parse_view_df(df, parse_epsv=True):
    if "epsv" in df.columns:
        print("Parsing scalar fields")
        for int_field in [
            "dim",
            "mesh_ref",
            "degree",
            "sol_id",
            "sol_pattern_size",
            "mode",
            "timestamp",
            "view_size",
            "t_view",
        ]:
            if df[int_field].dtype != int:
                print(" ", int_field)
                df[int_field] = df[int_field].apply(lambda x: int(x))

        if parse_epsv:
            print("Parse epsv")
            df["epsv"] = df["epsv"].apply(
                lambda v: tuple([float(x) for x in v.split(",")])
            )

    for vt in ["", "_max_pp", "_max_np"]:
        print(f"Parse view{vt}")
        df[f"view{vt}"] = df.apply(parse_view(f"view{vt}", float), axis=1)
    print("Parse view count")
    df["view_count"] = df.apply(parse_view("view_count", int), axis=1)
    return df


def load_view_df(path, view_size, parse_epsv):
    print(f"Read: {path}/view.csv.gz")
    view_df = pd.read_csv(f"{path}/view.csv.gz")
    print("Filter view size")
    view_df = view_df[view_df["view_size"] == view_size]
    print("Parse")
    view_df = parse_view_df(view_df, parse_epsv)
    if "setting" in view_df.columns:
        view_df = view_df.drop(["setting"], axis=1)
    print("Parse done")
    return view_df


def norm_view(x, mode, dataset_max, view_type):
    matrx = x[f"view{view_type}"]
    count = x["view_count"]

    def extended_log(x):
        return np.log(np.abs(x) + 1) * np.sign(x)

    if mode == "pure":
        return matrx / abs(matrx).max()
    elif mode == "resc":
        tmp = np.where(count > 0, matrx / count, 0.0)
        tmp /= abs(tmp).max()
        return tmp
    elif mode == "pure_log":
        tmp = extended_log(matrx)
        tmp /= abs(tmp).max()
        return tmp
    elif mode == "resc_log":
        tmp = np.where(count > 0, matrx / count, 0.0)
        tmp = extended_log(tmp)
        tmp /= abs(tmp).max()
        return tmp
    elif mode == "nothing":
        return matrx
    elif mode == "mean":
        return np.where(count > 0, matrx / count, 0.0)
    elif mode == "dataset_pure":
        return matrx / dataset_max
    elif mode == "log_nothing":
        if matrx.min() < 0:
            return extended_log(matrx)
        else:
            return np.log(matrx + 1e-30)
    elif mode == "gaussian":
        return (matrx - np.mean(matrx)) / np.std(matrx)
    elif mode == "resc_gaussian":
        tmp = np.where(count > 0, matrx / count, 0.0)
        return (tmp - np.mean(tmp)) / np.std(tmp)
    else:
        raise KeyError("Unrecognised matrix normalization mode", mode)


def normalize_view_df(df, normalization_modes, dtype=np.float64, check_ok=True):
    maxview = [
        df[f"view{x}"].apply(lambda x: np.abs(x).max()).max() for x in VIEW_TYPES
    ]

    def cast(x, dtype):
        return x if x.dtype == dtype else x.astype(dtype)

    for i, vt in enumerate(VIEW_TYPES):
        for nm in normalization_modes:
            print(f"Normalizing view{vt} as", nm)
            df[f"view{vt}_{nm}"] = df.apply(
                lambda x: cast(norm_view(x, nm, maxview[i], vt), dtype), axis=1
            )
    for nm in set(normalization_modes) & set(NORM_MODES_COUNT):
        print("Normalizing view count as", nm)
        df[f"view_count_{nm}"] = df.apply(
            lambda x: cast(norm_view(x, nm, maxview[i], vt), dtype), axis=1
        )

    if check_ok:
        for index, row in df.iterrows():
            for vt in VIEW_TYPES:
                for nm in normalization_modes:
                    matrix = row[f"view{vt}_{nm}"]
                    if np.isnan(matrix).any() or np.isinf(matrix).any():
                        for vt in VIEW_TYPES + ["_count"]:
                            print(row[f"view{vt}"])
                        raise ValueError(
                            "view at index",
                            index,
                            "has inf or nan entry, normalization",
                            nm,
                            vt,
                        )
                    if (np.any(matrix > 1) or np.any(matrix < -1)) and (
                        nm not in UNBOUNDED_NORMALIZATION_MODES
                    ):
                        raise ValueError(
                            "view at index",
                            index,
                            "is not properly normalized with",
                            nm,
                        )
    return df


def df_to_model_input(df, normalization_mode, view_type, inputp, tgt):
    if view_type == "sum":
        view = np.stack(df[f"view_{normalization_mode}"])[..., np.newaxis]
    elif view_type == "max":
        view = np.maximum(
            np.stack(df[f"view_max_pp_{normalization_mode}"]),
            np.stack(df[f"view_max_np_{normalization_mode}"]),
        )[..., np.newaxis]
    elif view_type == "max_ppnp":
        view = np.stack(
            [
                np.stack(df[f"view_max_pp_{normalization_mode}"]),
                np.stack(df[f"view_max_np_{normalization_mode}"]),
            ],
            axis=-1,
        )
    elif view_type == "sum+max":
        view = np.stack(
            [
                np.stack(df[f"view_{normalization_mode}"]),
                np.stack(df[f"view_max_pp_{normalization_mode}"]),
                np.stack(df[f"view_max_np_{normalization_mode}"]),
            ],
            axis=-1,
        )
    elif view_type == "sum+max+c":
        view = np.stack(
            [
                np.stack(df[f"view_{normalization_mode}"]),
                np.stack(df[f"view_max_pp_{normalization_mode}"]),
                np.stack(df[f"view_max_np_{normalization_mode}"]),
                np.stack(df[f"view_count_{normalization_mode}"]),
            ],
            axis=-1,
        )
    else:
        raise KeyError("Unrecognised view type:", view_type)

    return view.copy(), df[inputp].copy(), df[tgt].copy()


def fit_model(
    model,
    path,
    view_data,
    param_data,
    target_data,
    epochs,
    batch_size,
    validation_split,
    patience,
):
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"\nEnd epoch {epoch:04d} at: {datetime.datetime.now()}")
            print("----------------------------------------------------------")

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=path + "/model.h5",  # '/model.{epoch:03d}-{loss:.2e}.h5',
        monitor="loss",
        mode="min",
        save_best_only=True,
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=patience,
        monitor="loss",
        min_delta=1e-9,
        min_lr=1e-8,
    )
    callbacks = [CustomCallback(), model_checkpoint_callback, lr_scheduler]

    # fit
    history = model.fit(
        [view_data, param_data.to_numpy()],
        target_data,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=VERBOSE,
    )

    model.save(f"{path}/model.h5")

    # history
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch
    hist.to_csv(f"{path}/history.csv", index=False)

    return hist


def load_data(data_path, hyper_df):
    assert hyper_df["view_size"].nunique() == 1
    assert "join_on" in hyper_df.columns
    assert hyper_df["join_on"].nunique() == 1
    assert hyper_df["view_size"].nunique() == 1

    norm_modes = hyper_df["norm_mode"].unique()
    view_size = (hyper_df.loc[0, "view_size"],)
    join_on = hyper_df.loc[0, "join_on"].split(";")
    parse_epsv = False

    para_df = load_time_stats(data_path, parse_epsv=parse_epsv)
    view_df = load_view_df(data_path, view_size, parse_epsv)
    print("Making join")
    df = pd.merge(para_df, view_df, on=join_on, how="inner", suffixes=("", "_y"))
    print(f"Sizes: {len(para_df)} xx {len(view_df)} = {len(df)}")
    assert len(para_df) == len(df)
    df = normalize_view_df(df, norm_modes, dtype=np.float64)

    df["log2ndof"] = df["ndof"].apply(np.log2)
    return df


def main(output_dir, data_dir, metadata_path):
    hyper_df = pd.read_csv(metadata_path)
    df = load_data(data_dir, hyper_df)
    df = df.sample(frac=1.0, replace=False, random_state=0)

    timestamp_global = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for index, row in hyper_df.iterrows():
        # make folder
        timestamp_local = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_path = f"{output_dir}/model{timestamp_local}/"
        os.makedirs(model_path, exist_ok=True)
        hyper_df.at[index, "name"] = f"model{timestamp_local}"

        inputs = row["inputs"].split(";")
        # build model
        if "pretrain" in row:
            model_pretrain_path = row["pretrain"]
            model = tf.keras.models.load_model(
                f"{output_dir}/{model_pretrain_path}",
                custom_objects={"mse_mvar": mse_mvar, "mse_var": mse_var},
            )
            if row["learn_err"]:
                metrics = ["mae", mse, mse_var]
                loss = mse_mvar
            else:
                loss = "mse"
                metrics = []
            model.compile(
                loss=loss, metrics=metrics, optimizer=tf.keras.optimizers.Adam(row["lr"])
            )
        else:
            model = generic_build_model(
                len(inputs),
                row["cnn_type"],
                *row[["w1", "d1", "w2", "d2", "bn", "w3", "d3", "kernel_size"]],
                row["act"],
                row["opt"],
                view_size=row["view_size"],
                channels=VIEW_TYPE_CHANNELS[row["view_type"]],
                learn_err=bool(row["learn_err"]),
            )

        if "freeze" in row:
            for i in range(int(row["freeze"])):
                model.layers[i + 1].trainable = False

        trainable_count = count_params(model.trainable_weights)
        non_trainable_count = count_params(model.non_trainable_weights)
        hyper_df.at[index, "trainable_params"] = int(trainable_count)
        hyper_df.at[index, "non_trainable_params"] = non_trainable_count

        print("trainable_params", int(trainable_count))
        print("non_trainable_params", int(non_trainable_count))

        # fit model
        if "ndof_y" in df.columns:
            assert (df.ndof == df.ndof_y).all()
        view_data, param_data, target_data = df_to_model_input(
            df,
            row["norm_mode"],
            row["view_type"],
            tgt=row["tgt"],
            inputp=inputs,
        )

        if "pretrain" in row:
            results = model.evaluate([view_data, param_data], target_data)
            print("Pretrain model has loss: ", results)

        history = fit_model(
            model,
            model_path,
            view_data,
            param_data,
            target_data,
            epochs=row["epochs"],
            batch_size=row["batch_size"],
            validation_split=row["val_split"],
            patience=row["patience"],
        )

        # save stats
        hyper_df.at[index, "mse"] = history["mse"].min()
        hyper_df.at[index, "mae"] = history["mae"].min()

        hyper_df.to_csv(f"{output_dir}/hyper_params_fit_{timestamp_global}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=constants.DATA_PATH)
    parser.add_argument("--data_dir", default=constants.DATA_PATH)
    parser.add_argument("--metadata", default="")
    parsed = parser.parse_args()

    output_dir = parsed.output_dir
    data_dir = parsed.data_dir
    metadata_path = parsed.metadata
    main(output_dir, data_dir, metadata_path)
