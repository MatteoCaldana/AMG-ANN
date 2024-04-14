import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import constants

mpl.rcParams.update({"font.size": 14})


def parse_csv_array(string):
    return np.array([float(x) for x in string.split(",")])


def jointplot_hexbin():
    ff = ["t", "srow", "snze", "srownze", "rho"]
    ttt = {k: [] for k in ff}
    apps =  ["testcase1-diffusion-unstructured"]
    for app in apps:
        df = pd.read_csv(f"{constants.DATA_PATH}/processed/{app}/train/stats.csv.gz")

        join_c = constants.TEST_VARIABLES[app]

        for f in ["nrows", "sparsity", "nze"]:
            df[f] = df[f].apply(parse_csv_array)
        df["sspa"] = df.sparsity.apply(np.sum)
        df["srow"] = df.nrows.apply(np.sum)
        df["snze"] = df.nze.apply(np.sum)
        df["srownze"] = df.apply(lambda x: np.sum(x["nze"] * x["nrows"]), axis=1)
        df["srow2"] = df.nrows.apply(lambda x: np.sum(x**2))
        df["snze2"] = df.nze.apply(lambda x: np.sum(x**2))

        itt = df.groupby(join_c)["t"].idxmin()
        for f in ff:
            its = df.groupby(join_c)[f].idxmin()
            assert (itt.index == its.index).all()
            ts = df.loc[its]["theta"].to_numpy()
            ttt[f].append(ts)

    ff_lab = [r"$\sum_k n_k$", r"$\sum_k nnz_k$", r"$\sum_k n_k nnz_k$", r"$\rho$"]
    for i in range(len(ff_lab)):
        f = ff[i + 1]
        idxs = [np.random.choice(x.size, x.size, replace=False) for x in ttt["t"]]
        x = [ttt["t"][i][idxs[i]] for i in range(len(apps))]
        y = [ttt[f][i][idxs[i]] for i in range(len(apps))]
        sns.jointplot(
            x=np.concatenate(x),
            y=np.concatenate(y),
            kind="hex",
            marginal_kws={"bins": 15},
            gridsize=15,
        )
        plt.xlabel(r"$\theta^*$ = argmin($t$)")
        plt.ylabel(rf"$\theta^*$ = argmin({ff_lab[i]})")
        plt.tight_layout()
        plt.savefig(
            f"{constants.DATA_PATH}/paper-plots/02-argmin-time-vs-{f}.png", dpi=300
        )


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def plot_spectrum_analysis(save_path=None):
    df = pd.read_csv(
        constants.DATA_PATH
        + "/raw/testcase2.1-diffusion-singlecell/stat_compound_spectrum.csv"
    )

    gb = df.groupby(["dim", "mesh_ref", "diff", "mpt"])
    df = gb.aggregate({k: "mean" for k in ["min", "max", "amin", "amax"]})
    df = df.reset_index()
    df = df.drop(columns=["amin", "amax"])

    df["q"] = df["max"] / df["min"]
    df["h"] = 2.0 / 2 ** df["mesh_ref"]
    df["mptraw"] = df["mpt"]
    df["mpt"] = df["mptraw"].apply(lambda x: np.array([float(y) for y in x.split(",")]))
    df["d"] = df["mpt"].apply(lambda x: x[0])

    df = df[df.dim == 3]

    cmap = truncate_colormap(mpl.cm.turbo, 0.7, 1.0)

    fig = plt.figure(figsize=(10, 8))
    N = 4
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.voxels(np.ones((1,) * 3), facecolors=(0, 0, 1, 0.3), edgecolor=(0, 0, 0, 0.5))

    ax.voxels(np.ones((N,) * 3), facecolors=(1, 1, 1, 0.1), edgecolor=(0, 0, 0, 0.5))
    ax.scatter(*[[N / 2]] * 3, s=100, color="r")
    ax.scatter(*[[1 / 2]] * 3, s=100, color="r")
    ax.plot(*[[1 / 2, N / 2]] * 3, color="r")
    ax.text(-1, 0, 6.5, r"$d$: Distance of cell from center of $\Omega$", color="r")
    ax.text(
        0, -2, -1, r"$\mu_{MAX}=10^\varepsilon$" + "\nLarge diffusion cell", color="b"
    )

    ax.set_axis_off()

    ax = fig.add_subplot(2, 2, 3)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\lambda_{max} / \lambda_{min}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$h$")
    ax.grid(True)
    gb2 = df.groupby("diff")
    for name, gdf in gb2:
        gdf = gdf.sort_values("d", ascending=False)
        sc = ax.scatter(
            gdf.h, gdf.q, c=gdf.d, alpha=0.5, cmap=cmap.reversed(), linewidth=10
        )
        x = gdf.h
        y = gdf.q.min() / gdf.h**2 / 10
        ax.plot(x, y, "k--")
        ax.text(x.min(), y.max() * 2, f"$10^{{{int(name)}}}$", color="b")
    ax.text(x.min() - 0.002, 20, "$\mu_{MAX}$", color="b")
    ax.plot(x, y, "k--", label="$h^{-2}$")
    ax.legend()
    ax.set_ylim([10, 10**11])
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Distance $d$")
    ax.set_title("Conditioning vs mesh size")

    ax = fig.add_subplot(2, 2, 4)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\lambda_{max} / \lambda_{min}$")
    ax.set_xlabel(r"distance from center $d$")
    ax.grid(True)
    gb2 = df.groupby("diff")
    for diff, gdf in gb2:
        gb3 = gdf.groupby("mesh_ref")
        maxref = gb2.mesh_ref.max()[diff]
        minref = gb2.mesh_ref.min()[diff]
        for ref, gdf in gb3:
            gdf = gdf.sort_values("q", kind="stable", ascending=False)
            gdf = gdf.sort_values("d", kind="stable")
            ax.plot(gdf.d, gdf.q, c=cmap((ref - minref) / (maxref - minref)))
        ax.text(-0.1, gdf.q.max() * 1.2, f"$10^{{{int(diff)}}}$", color="b")
    ax.text(-0.1, 20, r"$\mu_{MAX}$", color="b")
    ax.set_ylim([10, 10**11])
    ax.set_xlim([-0.125, 1])
    sc = ax.scatter(df.d, df.q, c=df.mesh_ref, cmap=cmap, s=0)
    cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(minref, maxref + 1))
    cbar.ax.set_yticklabels(2 / 2 ** np.arange(minref, maxref + 1))
    cbar.set_label("Mesh size $h$")
    ax.set_title("Conditioning vs $d$")

    plt.tight_layout()
    if not save_path is None:
        plt.savefig("{save_path}/spectrum_pt1.png")


if __name__ == "__main__":
    jointplot_hexbin()
    plot_spectrum_analysis()
