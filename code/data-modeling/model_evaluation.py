import os
import tensorflow as tf
import pandas as pd
import re
import numpy as np
import json
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import train_ann
from cnn_models import mse_mvar, mse_var, mse
from data_preprocessing import insert_every_n


def perf_realization(x, xmax):
    if xmax == 0:
        if x == 0:
            return 1
        else:
            return x
    else:
        return x / xmax


def get_summary_perf(perf_df, sigma_bar=np.inf):
    print("Sigma bar: ", sigma_bar)
    err_col = [x for x in perf_df.columns if x.endswith("nn_err")]
    assert len(err_col) == 1
    err_col = err_col[0]
    perf_df["perf"] = perf_df.apply(
        lambda x: x["perf"] if x[err_col] < sigma_bar else 0.0, axis=1
    )

    goodbad_df = list(perf_df.groupby(perf_df["perf"] < 0))
    if len(goodbad_df) == 1:
        goodbad_df.append(
            (not goodbad_df[0][0], pd.DataFrame([], columns=goodbad_df[0][1].columns))
        )
    good_df, bad_df = goodbad_df[0][1], goodbad_df[1][1]
    if good_df.perf.min() < 0.0:
        good_df, bad_df = bad_df, good_df
    # how often are we better than the literature?
    good_fraction = len(good_df) / len(perf_df)
    print(f"we do equal or better in {good_fraction*100:.2f}% of the cases")
    print(
        "we perform "
        f"{perf_df['perf'].mean()*100:.2f}%(mean) "
        f"{perf_df['perf'].median()*100:.2f}%(median) better"
    )
    print(
        "when we do better, we perform "
        f"{good_df['perf'].mean()*100:.2f}%(mean) "
        f"{good_df['perf'].median()*100:.2f}%(median) better"
    )
    # how much better w.r.t. the argmin tgt?
    good_df["perf_realization"] = perf_df.apply(
        lambda x: perf_realization(x["perf"], x["perf_max"]), axis=1
    )
    print(
        "when we do better, we realize "
        f"{good_df['perf_realization'].mean()*100:.2f}%(mean) "
        f"{good_df['perf_realization'].median()*100:.2f}%(median) of max perf"
    )
    # how much better in absolute?
    print(
        "when we do worse, our performances are "
        f"{bad_df['perf'].mean()*100:.2f}%(mean) "
        f"{bad_df['perf'].median()*100:.2f}%(median) worse"
    )
    print("-----------------------------------------------------------")

    return {
        "frac_good": len(good_df) / len(perf_df),
        "mean_gain": perf_df["perf"].mean(),
        "medi_gain": perf_df["perf"].median(),
        "perf_mean": good_df["perf_realization"].mean(),
        "perf_medi": good_df["perf_realization"].median(),
        "lost_mean": bad_df["perf"].mean(),
        "lost_medi": bad_df["perf"].median(),
        "perf_vec": ",".join([f"{x:.17e}" for x in perf_df["perf"]]),
        "perf_vec_max": ",".join([f"{x:.17e}" for x in perf_df["perf_max"]]),
    }


def find_metadata(models_path, model_id):
    hyper_pattern = "hyper_params_fit_[0-9_-]+.csv"
    hyper_paths = [x for x in os.listdir(models_path) if re.match(hyper_pattern, x)]
    found = False
    for hyper_path in hyper_paths:
        hyper_path = rf"{models_path}/{hyper_path}"
        hyper_df = pd.read_csv(hyper_path)
        hyper_row = hyper_df[hyper_df["name"] == model_id]
        if len(hyper_row) == 1:
            found = True
            break

    assert found
    return hyper_row


def eval_model(models_path, model_id, data_path):
    test_config = {"N": 101, "lvls": 1, "t0": 0.5, "delta": 0.45, "reduce": 5}
    theta_tobeat = 0.5
    save_path = f"{models_path}/{model_id}"
    os.makedirs(save_path, exist_ok=True)

    hyper_row = find_metadata(models_path, model_id)

    df = train_ann.load_data(data_path, hyper_row)

    print("Load model")
    model = tf.keras.models.load_model(
        f"{models_path}/{model_id}/model.h5",
        custom_objects={"mse_mvar": mse_mvar, "mse_var": mse_var, "mse": mse},
    )

    if "ndof_y" in df.columns:
        assert (df.ndof == df.ndof_y).all()

    hyper_row = hyper_row.squeeze()
    assert "inputs" in hyper_row
    inputs = hyper_row["inputs"].split(";")

    print("Shallow test")
    view_data, param_data, tgt = train_ann.df_to_model_input(
        df,
        hyper_row["norm_mode"],
        hyper_row["view_type"],
        tgt=hyper_row["tgt"],
        inputp=inputs,
    )
    pred = model.predict([view_data, param_data])

    print("Saving shallow")
    np.savetxt(f"{save_path}/pred.csv", pred, delimiter=",")
    np.savetxt(f"{save_path}/tgt.csv", tgt.to_numpy(), delimiter=",")

    print("Test model")
    tobeat_df = test_model(
        model,
        df,
        hyper_row,
        test_config,
        theta_tobeat,
        hyper_row["join_on"].split(";"),
        "t",
    )
    drop_columns = [col for col in tobeat_df.columns if "view" in col]
    tobeat_df = tobeat_df.drop(columns=drop_columns)
    tobeat_df.to_csv(f"{save_path}/perf_df.csv", index=False)
    summary = get_summary_perf(tobeat_df)
    with open(f"{save_path}/summary.json", "w") as fp:
        json.dump(
            {
                "models_path": models_path,
                "model_id": model_id,
                "data_path": data_path,
                **summary,
            },
            fp,
        )


def interp1d_tgt(df, theta, tgt):
    interp = scipy.interpolate.interp1d(
        df["theta"],
        df[tgt],
        fill_value=(
            df[tgt][df[tgt].first_valid_index()],
            df[tgt][df[tgt].last_valid_index()],
        ),
        bounds_error=False,
    )
    return interp(theta)


def test_model(model, df, hyper_row, config, theta_tobeat, join_on, perf_tgt):
    tgt = hyper_row["tgt"]
    tobeat_df = df[np.isclose(df["theta"], theta_tobeat)]
    print("'tobeat_df' size is:", len(tobeat_df))
    tobeat_df = tobeat_df.reset_index(drop=True)
    tobeat_df["theta_nn"] = np.nan
    tobeat_df[f"{tgt}_nn"] = np.nan
    tobeat_df["theta_argmin"] = np.nan
    tobeat_df[f"{tgt}_min"] = np.nan
    tobeat_df[f"{tgt}_interp"] = np.nan

    test_case_cols = join_on
    print("Using test case cols:", test_case_cols)
    print("Making groups...")
    gb = {idx: df for idx, df in df.groupby(test_case_cols)}
    print("Start iterations")
    for index, row in tobeat_df.iterrows():
        print(index, "/", len(tobeat_df))

        test_case_df = gb[tuple(row[test_case_cols])]
        test_case_df = test_case_df.sort_values("theta")
        idx_min_tgt = test_case_df[tgt].idxmin()
        tobeat_df.loc[index, "theta_argmin"] = test_case_df.loc[idx_min_tgt, "theta"]
        tobeat_df.loc[index, f"{tgt}_min"] = test_case_df.loc[idx_min_tgt, tgt]
        # suppose the transfromation between tgt and perf_tgt does not change the position of the minima
        tobeat_df.loc[index, f"{perf_tgt}_min"] = test_case_df.loc[
            idx_min_tgt, perf_tgt
        ]
        tobeat_df.loc[index, "t_min"] = test_case_df.loc[idx_min_tgt, "t"]
        idx_max_tgt = test_case_df[tgt].idxmax()
        tobeat_df.loc[index, "t_max"] = test_case_df.loc[idx_max_tgt, "t"]

        theta, tgt_nn, tgt_err = pred_theta(
            model,
            hyper_row["norm_mode"],
            hyper_row["view_type"],
            row,
            config,
            (
                hyper_row["inputs"].split(";")
                if "inputs" in hyper_row
                else ["log2ndof", "theta", "degree"]
            ),
        )
        tobeat_df.loc[index, "theta_nn"] = theta
        tobeat_df.loc[index, f"{tgt}_nn"] = tgt_nn
        tobeat_df.loc[index, f"{tgt}_nn_err"] = tgt_err
        tgt_interp1d = interp1d_tgt(test_case_df, theta, tgt)
        tobeat_df.loc[index, f"{tgt}_interp"] = tgt_interp1d
        perf_tgt_interp1d = interp1d_tgt(test_case_df, theta, perf_tgt)
        tobeat_df.loc[index, f"{perf_tgt}_interp"] = perf_tgt_interp1d

    # performance according to nn target (however this is normalized)
    tobeat_df[f"perf_{tgt}"] = tobeat_df.apply(
        lambda x: 1 - x[f"{tgt}_interp"] / (x[tgt] + (x[tgt] == 0.0)), axis=1
    )
    tobeat_df[f"perf_max_{tgt}"] = tobeat_df.apply(
        lambda x: 1 - x[f"{tgt}_min"] / (x[tgt] + (x[tgt] == 0.0)), axis=1
    )
    # performance according to real target
    tobeat_df["perf"] = tobeat_df.apply(
        lambda x: 1 - x[f"{perf_tgt}_interp"] / x[perf_tgt], axis=1
    )
    tobeat_df["perf_max"] = tobeat_df.apply(
        lambda x: 1 - x[f"{perf_tgt}_min"] / x[perf_tgt], axis=1
    )
    return tobeat_df


def get_view(row, view_type, norm_mode):
    if view_type == "sum":
        view = row[f"view_{norm_mode}"][..., np.newaxis]
    elif view_type == "max":
        max_pp = row[f"view_max_pp_{norm_mode}"]
        max_np = row[f"view_max_np_{norm_mode}"]
        view = np.maximum(max_np, max_pp)[..., np.newaxis]
    elif view_type == "max_ppnp":
        max_pp = row[f"view_max_pp_{norm_mode}"]
        max_np = row[f"view_max_np_{norm_mode}"]
        view = np.stack([max_pp, max_np], axis=-1)
    elif view_type == "sum+max":
        ssum = row[f"view_{norm_mode}"]
        max_pp = row[f"view_max_pp_{norm_mode}"]
        max_np = row[f"view_max_np_{norm_mode}"]
        view = np.stack([ssum, max_pp, max_np], axis=-1)
    elif view_type == "sum+max+c":
        ssum = row[f"view_{norm_mode}"]
        max_pp = row[f"view_max_pp_{norm_mode}"]
        max_np = row[f"view_max_np_{norm_mode}"]
        c = row[f"view_count_{norm_mode}"]
        view = np.stack([ssum, max_pp, max_np, c], axis=-1)
    else:
        raise KeyError("Unrecognised view type:", view_type)
    return view


def pred_theta(ann, norm_mode, view_type, row, config, inputs):
    N = config["N"]
    lvls = config["lvls"]
    t0 = config["t0"]
    delta = config["delta"]
    reduce = config["reduce"]

    ts = []
    for _ in range(lvls):
        param = np.column_stack(
            tuple(
                [
                    (
                        np.repeat(row[x], N)
                        if x != "theta"
                        else np.linspace(t0 - delta, t0 + delta, N)
                    )
                    for x in inputs
                ]
            )
        )
        view = get_view(row, view_type, norm_mode)
        view = np.repeat(view[np.newaxis, ...], N, axis=0)

        pred = ann.predict([view, param], verbose=2)
        if len(pred.shape) > 1 and pred.shape[1] > 1:
            err = np.sqrt(pred[:, 1])
            pred = pred[:, 0]
        else:
            err = 0

        pred = np.clip(pred, 0.0, 1.0)

        index = np.argmin(pred)
        t0 = param[index, 1]
        tgt_pred = pred[index]
        delta /= reduce
        ts.append(t0)

    return t0, tgt_pred, np.mean(err * (1 - pred))


def plot_problem_predictions(models_path, model_id, data_path, save_path):
    join_on = find_metadata(models_path, model_id).squeeze()["join_on"]
    join_on = join_on.split(";")
    data_df = train_ann.load_data(data_path, find_metadata(models_path, model_id))
    os.makedirs(save_path, exist_ok=True)
    pred = np.loadtxt(f"{models_path}/{model_id}/pred.csv", delimiter=",")

    data_df["tnl_nn"] = pred[:, 0]
    if pred.shape[1] == 2:
        data_df["tnl_nn_err"] = pred[:, 1]

    perf_df = pd.read_csv(f"{models_path}/{model_id}/perf_df.csv")
    perf_df = perf_df.set_index(join_on)

    for name, group in data_df.groupby(join_on):
        plt.figure()

        plt.plot(group["theta"], group["tnl_nn"], label="ANN prediction")
        if pred.shape[1] == 2:
            plt.fill_between(
                group["theta"],
                group["tnl_nn"] - np.sqrt(group["tnl_nn_err"]),
                group["tnl_nn"] + np.sqrt(group["tnl_nn_err"]),
                alpha=0.3,
                color="b",
            )

        plt.scatter(group["theta"], group["tnl"], color="#ff7f0e", label="data (mean)")
        t = group["t"]
        tsgnl = (group["tsg"] - t.min()) / (t.max() - t.min())
        plt.plot(group["theta"], tsgnl, label="smoothed data")

        plt.axvline(
            x=perf_df.loc[name, "theta_nn"],
            color="k",
            linestyle="--",
            label=r"$\theta^*$",
        )
        plt.axvline(x=0.5, color="r", linestyle="--", label=r"Default $\theta$")

        plt.ylabel("Normalized computational cost $t$")
        plt.xlabel(r"$\theta$")

        title = ".".join(f"{join_on[i]}{name[i]}" for i in range(len(name)))
        plt.title(insert_every_n(title, "\n", 50), wrap=True)
        plt.savefig(f"{save_path}/{hash(title)}.png")
        plt.close()


def plot_variance_scatter(models_path, model_id, save_path):
    pred = np.loadtxt(f"{models_path}/{model_id}/pred.csv", delimiter=",")
    target = np.loadtxt(f"{models_path}/{model_id}/tgt.csv", delimiter=",")
    y = target.reshape((-1, 1))
    t = pred[:, 0]
    if pred.shape[1] == 2:
        s = np.sqrt(pred[:, 1])
    else:
        s = np.exp(1)

    s_log = np.log(s)
    s_log = -s_log / np.abs(s_log).max()

    idx = np.argsort(s)[::-1]

    t = t[idx]
    y = y[idx]
    s = s[idx]
    s_log = s_log[idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.grid("on", linestyle="--")
    eps = 1e-3
    im = plt.scatter(
        y, t, alpha=s_log, c=s_log, s=20000 * np.where(s < eps, eps, s), cmap="jet"
    )
    im = plt.scatter(
        y,
        t,
        alpha=1.0,
        c=s,
        s=0.0,
        cmap=plt.cm.jet_r,
        norm=matplotlib.colors.LogNorm(),
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r"estimated error $\tilde{\sigma}$")

    plt.xlim([0, 1])

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.axis("square")
    plt.title("ANN predictions")
    plt.xlabel(r"$t$ [true]")
    plt.ylabel(r"$\tilde{t}$ [predicted]")

    plt.tight_layout()
    plt.savefig(f"{save_path}/variance_prediction_scatter.png", dpi=200)


def plot_performance_hist(models_path, model_id, save_path):
    perf_df = pd.read_csv(f"{models_path}/{model_id}/perf_df.csv")

    plt.hist(perf_df["perf_max"], bins=20, label=r"$P_{MAX}$")
    plt.hist(perf_df["perf"], bins=20, label=r"$P$", alpha=0.7)
    plt.axvline(perf_df["perf"].mean(), color="r", linestyle="dashed", label=r"$P_m$")
    plt.axvline(perf_df["perf"].median(), color="k", linestyle="dashed", label=r"$P_M$")
    plt.xlim([max(-0.5, perf_df["perf"].min()), perf_df["perf_max"].max()])
    plt.xlabel("Peformace")
    plt.ylabel("# of problems")
    plt.legend()
    plt.show()
    plt.savefig(f"{save_path}/performace_hist{model_id}.png", dpi=200)



def plot_comp_cost(models_path, model_id, save_path):
    df = pd.read_csv(f"{models_path}/{model_id}/perf_df.csv")

    df["x"] = df.ndof
    df["y1"] = df.t / df.ndof
    df["y2"] = df.t_interp / df.ndof

    alpha = 0.1
    s = 200
    slopes_classic = []
    fig, axs = plt.subplots(1, 1)
    axs = np.array([axs])
    i = 0
    for j in [1, 2, 3]:
        axs[i].plot(0, 0, label=f"p={j}", linewidth=0)
    for value in [1, 2, 3]:
        c2 = plt.cm.autumn_r(value / 3)
        c1 = plt.cm.winter_r(value / 3)

        data = df[df.degree == value]

        sns.regplot(
            data=data,
            x="x",
            y="y1",
            color="black",
            scatter=False,
            logx=True,
            ax=axs[i],
            ci=None,
            line_kws={"linewidth": 5},
        )

        p = sns.regplot(
            data=data,
            x="x",
            y="y1",
            color=c1,
            scatter=False,
            logx=True,
            ax=axs[i],
            ci=None,
            label=r"$\theta=0.5$",
        )

        slope, intercept, r, p, sterr = scipy.stats.linregress(
            np.log(data["x"]), data["y1"]
        )
        print(
            f"p={value}, classic",
            " ".join(f"{x:.4e}" for x in [slope, intercept, r, p, sterr]),
        )
        slopes_classic.append(slope)

        sns.scatterplot(
            data=df[df.degree == value],
            x="x",
            y="y1",
            color=c1,
            s=s,
            alpha=alpha,
            ax=axs[i],
        )

    slopes_optimal = []
    for value in [1, 2, 3]:
        c2 = plt.cm.autumn_r(value / 3)
        c1 = plt.cm.winter_r(value / 3)

        data = df[df.degree == value]

        p = sns.regplot(
            data=data,
            x="x",
            y="y2",
            color="black",
            scatter=False,
            logx=True,
            ax=axs[i],
            ci=None,
            line_kws={"linewidth": 5},
        )

        sns.regplot(
            data=data,
            x="x",
            y="y2",
            color=c2,
            scatter=False,
            logx=True,
            ax=axs[i],
            ci=None,
            label=r"$\theta=\theta^*$",
        )

        slope, intercept, r, p, sterr = scipy.stats.linregress(
            np.log(data["x"]), data["y2"]
        )
        print(
            f"p={value}, optimal",
            " ".join(f"{x:.4e}" for x in [slope, intercept, r, p, sterr]),
        )
        slopes_optimal.append(slope)

        sns.scatterplot(
            data=data, x="x", y="y2", color=c2, s=s, alpha=alpha, ax=axs[i],
        )

    slope_ratio = np.array(slopes_classic) / np.array(slopes_optimal)
    print(slope_ratio)

    max_gain = np.max(1 - 1 / np.where(slope_ratio > 0, slope_ratio, 1))
    print(max_gain)
    # axs[i].text(
    #     0.05,
    #     0.5,
    #     f"Up to {max_gain*100:.2f}%\ngain in scaling",
    #     transform=axs[i].transAxes,
    #     fontsize=18,
    # )

    axs[i].set_xscale("log")

    axs[i].set_xlabel("#DoFs")
    axs[i].set_ylabel("")
    leg = axs[i].legend(
        ncol=3, loc="upper left", fontsize=13, bbox_to_anchor=(-0.1, 1.01)
    )
    leg.get_frame().set_edgecolor("b")
    leg.get_frame().set_linewidth(0.0)
    leg.set_zorder(1)
    
    axs[0].set_ylabel("normalized cost ($t/$#DoFs)")
    plt.savefig(f"{save_path}/comp_cost{model_id}.png")


def get_elbow_distance(v):
    p1 = np.array([0, v[0]])
    p2 = np.array([v.size - 1, v[-1]])
    d = np.empty_like(v)
    for i, p3 in enumerate(v):
        p3 = np.array([i, p3])
        d[i] = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    return d

def sigma_bar_study(models_path, model_id, save_path):
    df = pd.read_csv(f"{models_path}/{model_id}/perf_df.csv")
    df = df.sort_values("tnl_nn_err", ascending=False)
    perf_ = df["perf"].to_numpy()
    perf_bad = np.where(perf_ < 0, 1, 0)
    fig, ax1 = plt.subplots(figsize=(8, 7))
    ax2 = ax1.twinx()
    # plt.plot(perf_bad_cumsum / count, '-o')
    err = np.sqrt(np.sort(df.tnl_nn_err)[::-1])
    d_err = get_elbow_distance(err)

    ax2.plot(err, "b", linewidth=3)
    ax2.axvline(np.argmax(d_err), color="b", linestyle=":", linewidth=3)
    # plt.plot(d, '-o')

    pb = np.cumsum(perf_bad[::-1])[::-1] / len(df)

    # d_pb = get_elbow_distance(pb)
    ax1.plot([0], [0], "b", linewidth=3)
    ax1.plot([0], [0], "b:", linewidth=3)

    idx = np.argmax(d_err)
    ax1.plot(pb, "r", linewidth=3)
    # ax1.axvline(np.argmax(d_pb), color="r", linestyle=":", linewidth=3)

    cumperf = (perf_[::-1].cumsum() / len(df))[::-1]
    ax1.plot(cumperf, "r--", linewidth=3)
    # ax1.grid("on")
    ax1.grid(which="both", linestyle="-")

    fontsize = 18
    plt.title(
        r"$\bar{\sigma} = n$-th largest error indicator $\hat{\sigma}$",
        fontsize=fontsize + 4,
    )
    ax1.set_xlabel(r"$n$", fontsize=fontsize)
    ax1.set_ylabel(r"performance", fontsize=fontsize, color="r")
    ax2.set_ylabel(r"error indicator $\hat{\sigma}$", fontsize=fontsize, color="b")

    ax1.legend(
        [
            r"sorted $\hat{\sigma}$",
            r"$\hat{\sigma}$ elbow",
            r"$1 - PB$",
            # r"$PB$ elbow",
            r"$P_m$",
        ],
        fontsize=fontsize - 3,
    )

    x_pos = len(d_err) / 2
    y_pos = cumperf.max() * 0.9
    print(y_pos)
    ax1.text(
        x_pos,
        y_pos,
        f"$P_m$: {(cumperf[idx]-cumperf[0])*100:.1f}%\nPB: +{(pb[0]-pb[idx])*100:.1f}%",
    )
    color = (0.8,) * 3
    ax1.plot([0, x_pos], [pb[0], y_pos], "-", color=color)
    ax1.plot([x_pos, idx], [y_pos, pb[idx]], "-", color=color)
    ax1.plot([0, x_pos], [cumperf[0], y_pos], "-", color=color)
    ax1.plot([x_pos, idx], [y_pos, cumperf[idx]], "-", color=color)
    ax1.tick_params(axis="both", which="major", labelsize=fontsize)
    ax2.tick_params(axis="both", which="major", labelsize=fontsize)
    plt.tight_layout()
    #plt.savefig("../paper-plots/paper-3d/sigma_bar_choice_v3.png")

if __name__ == "__main__":
    app = "testcase1-diffusion-unstructured"

    models_path = f"../../data/models/{app}"
    data_path = f"../../data/processed/{app}/train/"
    model_id = "model2024-04-17_08-14-43"
    save_path = f"{models_path}/{model_id}/dump"

    # plot_problem_predictions(models_path, model_id, data_path, save_path)
    plot_variance_scatter(models_path, model_id, save_path)
    # plot_performance_hist(models_path, model_id, save_path)
    # plot_comp_cost(models_path, model_id, save_path)
    sigma_bar_study(models_path, model_id, save_path)
