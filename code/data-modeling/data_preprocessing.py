import pandas as pd
import matplotlib.pyplot as plt
import constants
import os
import numpy as np
import scipy.signal
import matplotlib as mpl


def insert_every_n(string, char, n):
    return char.join([string[i : i + n] for i in range(0, len(string), n)])


def plot_smoothed_data(df, app):
    default_fontsize = mpl.rcParams["font.size"]
    mpl.rcParams.update({"font.size": 14})

    keys = constants.TEST_VARIABLES[app]
    gb = df.groupby(keys)

    save_path = f"{constants.DATA_PATH}/raw/{app}/dump/smoothed_data"
    os.makedirs(save_path, exist_ok=True)
    os.system(f"rm {save_path}/*")
    for name, group in gb:
        plt.figure()
        plt.plot(group["theta"], 1e-6 * group["t"], "o-", label="data")
        plt.plot(group["theta"], 1e-6 * scipy.signal.savgol_filter(group["t"], 21, 7), "--", label="Savitzky-Golay(21, 7)")
        plt.plot(group["theta"], 1e-6 * scipy.signal.savgol_filter(group["t"], 5, 2), "--", label="Savitzky-Golay(5, 2)")
        plt.legend()
        plt.ylabel("time [s]")
        plt.xlabel(r"$\theta$")
        title = ".".join(f"{keys[i]}{name[i]}" for i in range(len(name)))
        #plt.title(insert_every_n(title, "\n", 50), wrap=True)
        plt.savefig(f"{save_path}/{hash(title)}.png")
        plt.close()
        
    mpl.rcParams.update({"font.size": default_fontsize})


def add_rho_column(simul_df):
    if "res" in simul_df.columns and "p_res" not in simul_df.columns:
        print("WARNING")
        simul_df["p_res"] = simul_df["res"].copy()
    simul_df["res"] = simul_df["p_res"].apply(
        lambda x: np.array([float(i) for i in f"{x}".split(",")])
    )
    simul_df["rho"] = simul_df["res"].apply(lambda x: calc_rho(x))

    max_last_res = max(simul_df["res"].apply(lambda x: x[-1]))
    simul_df["irhov"] = simul_df["res"].apply(lambda x: interp_rho(x, max_last_res))

    simul_df["rhov"] = simul_df["rho"]
    simul_df["rho"] = simul_df["rhov"].apply(lambda x: x[-1])
    simul_df["irho"] = simul_df["irhov"].apply(lambda x: x[-1])
    return


def interp_rho(res, res_point):
    if res.size == 1:
        return (1, np.array([1]), np.array([1]))
    if res[-1] == res_point:
        return (len(res) - 1, res_point, calc_rho(res)[-1])
    elif res[-1] < res_point:
        lle = float(len(res))
        i = np.interp(np.log(res_point), np.log(res[-1:-3:-1]), [lle - 1, lle - 2])
        irho = np.exp(np.log(res_point / res[0]) / i)
        return (i, res_point, irho)
    else:
        print("Warning: res_point is not the max")
        return


def calc_rho(res):
    if res.size == 1:
        return np.array([1])
    return np.array(
        [np.exp(np.log(res[i + 1] / res[0]) / (i + 1)) for i in range(len(res) - 1)]
    )


def normalize_rho_t(df, tcc, cols=["t", "tsg", "rho", "irho"]):
    gb = df.groupby(tcc)
    norm = gb[cols].transform(lambda x: (x - x.mean()) / x.std())
    norm = norm.rename(columns={k: k + "n" for k in cols})
    norml = gb[cols].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    norml = norml.rename(columns={k: k + "nl" for k in cols})
    return pd.concat([norm, norml, df], axis=1)


def best_theta(gdf):
    t_min = gdf["t"].min()
    theta_opt = gdf[gdf["t"] == t_min].iloc[0]["theta"]
    t_default = gdf[(gdf["theta"] - 0.5).abs() < 1e-8].iloc[0]["t"]
    gain = (t_default - t_min) / t_default
    return pd.DataFrame(
        [
            {
                "t_default": t_default,
                "t_min": t_min,
                "theta_opt": theta_opt,
                "gain": gain,
            }
        ]
    )


def preprocess_data(app, window=21, order=7):
    filepath = f"{constants.DATA_PATH}/raw/{app}/times/stats.csv"
    df = pd.read_csv(filepath)
    df["t"] = df["t_solve"] + df["t_amg_setup"]
    if len(df.groupby(constants.TEST_VARIABLES[app] + ["theta"])) != len(df):
        print("WARNING: some measurements are done more than once, taking the mean")
        col_types = df.dtypes.to_dict()
        number_col = [k for k in col_types if col_types[k].name != "object"]
        object_col = [k for k in col_types if col_types[k].name == "object"]
        df1 = df.groupby(constants.TEST_VARIABLES[app] + ["theta"])[number_col].mean()
        df2 = df.groupby(constants.TEST_VARIABLES[app] + ["theta"])[object_col].first()
        df = pd.concat([df1, df2], axis=1)
    df["tsg"] = df.groupby(constants.TEST_VARIABLES[app])["t"].transform(
        lambda x: scipy.signal.savgol_filter(x, window, order)
    )
    add_rho_column(df)
    df = normalize_rho_t(df, constants.TEST_VARIABLES[app])
    return df


def split_train_validation_test(df, app, pooling, test=0.1):
    keys = constants.TEST_VARIABLES[app]
    gb = df.groupby(keys)
    n_test = int(np.round(test * len(gb)))
    n_train = len(gb) - n_test
    idxs = [*[0] * n_train, *[1] * n_test]
    np.random.shuffle(idxs)
    df_idxs = [[], []]
    for i, idx in enumerate(gb.groups.values()):
        df_idxs[idxs[i]].append(idx)
    dfs = [df.loc[np.concatenate(df_idx)] for df_idx in df_idxs]

    view_filepath = f"{constants.DATA_PATH}/raw/{app}/pooling{pooling}/stats.csv"
    view_df = pd.read_csv(view_filepath).reset_index()

    for i, dataset in enumerate(["train", "test"]):
        path = f"{constants.DATA_PATH}/processed/{app}/{dataset}/"
        os.makedirs(path, exist_ok=True)
        dfs[i].to_csv(f"{path}/stats.csv.gz", index=False)

        join = pd.merge(dfs[i], view_df, on=keys)
        assert len(join) == len(dfs[i])
        view_idx = join["index"].unique()
        view_df.loc[view_idx].to_csv(f"{path}/view.csv.gz", index=False)


if __name__ == "__main__":
    app = "testcase1-diffusion-unstructured"
    df = preprocess_data(app, 15, 7)
    plot_smoothed_data(df, app)
    dfs = split_train_validation_test(df, app, 50, val=0.1, test=0.1)
