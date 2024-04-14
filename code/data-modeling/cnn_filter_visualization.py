# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np

from train_ann import load_data, df_to_model_input, load_view_df, normalize_view_df
from cnn_models import mse_mvar, mse_var, mse
import constants
from model_evaluation import find_metadata

colors1 = plt.cm.YlOrBr_r(np.linspace(0.0, 1, 128))
colors2 = plt.cm.GnBu(np.linspace(0, 1, 128))

colors = np.vstack((colors1, [(1, 1, 1, 1)] * 3, colors2))
mymap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)

mpl.rcParams.update({"font.size": 14})


def get_partial_models(model):
    model_dummies = []
    for i, layer in enumerate(model.layers):
        if "conv" not in layer.name:
            continue

        filters, _ = layer.get_weights()
        print(i, layer.name, filters.shape)

        model_dummies.append(
            tf.keras.Model(inputs=model.inputs, outputs=model.layers[i].output)
        )
        model_dummies[-1].summary()
    return model_dummies


def feature_maps_viz(data_dir, models_path, model_id, save_path=None):
    model_path = f"{models_path}/{model_id}/model.h5"
    hyper_row = find_metadata(models_path, model_id)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"mse_mvar": mse_mvar, "mse_var": mse_var, "mse": mse},
    )
    layers_mod = get_partial_models(model)
    hyper_row["join_on"] = ";".join(constants.TEST_VARIABLES["testcase2.1-diffusion-singlecell"])
    df = load_data(data_dir, hyper_row)
    df["log2ndof"] = df["ndof"].apply(np.log2)
    df = df[np.abs(df.theta - 0.5) < 1e-6]

    df["dist"] = df["mpt"].apply(
        lambda x: np.linalg.norm([float(t) for t in x.split(",")])
    )
    print(df.ndof.unique())
    ndofs_v = [4913, 35937, 274625]
    df2 = df[df.ndof.isin(ndofs_v)]
    df2 = df2[np.abs(df2["diff"] - 2.0) < 1e-4]
    dist_tab = df2.groupby("ndof")["dist"].unique()

    for ndof in ndofs_v:
        print("ndof", ndof)
        dists = np.sort(dist_tab[ndof])
        dists = [dists[0], dists[-1]]
        print("dist", dists)
        for di, dist in enumerate(dists):
            print("DD", di, dists)
            df3 = df2[(df2.ndof == ndof) & (df2.dist == dist)]
            print("compacting data")
            view_data, param_data, _ = df_to_model_input(
                df3,
                "pure_log",
                "sum+max+c",
                tgt="tnl",
                inputp=["log2ndof", "theta", "degree"],
            )

            for l in range(3):
                print("L", l)
                p = layers_mod[l].predict([view_data, param_data])
                print(p.min(), p.max())
                fig, axs = plt.subplots(4, 4, figsize=(10, 8))
                for i in range(4):
                    for j in range(4):
                        im = axs[i, j].imshow(
                            p[0, :, :, i + j * 8],
                            cmap=mymap,
                            norm=mpl.colors.TwoSlopeNorm(
                                0, vmin=np.min(p), vmax=np.max(p)
                            ),
                        )
                        axs[i, j].set_yticks([])
                        axs[i, j].set_xticks([])

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(im, cax=cbar_ax)

                if l == 0:
                    fig.text(
                        0.5,
                        1 - 0.08,
                        f"#DoF:{ndof}, cell {'furthest' if di == 1 else 'nearest'} to center",
                        ha="center",
                        fontsize=36,
                    )
                if ndof == 125 and di == 0:
                    fig.text(
                        0.04,
                        0.5,
                        f"CNN Layer {l + 1}",
                        va="center",
                        rotation="vertical",
                        fontsize=36,
                    )
                print("HH", ndof, dist, l)
                if not save_path is None:
                    plt.savefig(f"{save_path}/cnn_filter-ndof{ndof}-dist{dist}-layer{l}.png")
                    plt.close()


def pooling_visualization(data_path, save_path=None):
    df = load_view_df(data_path, 75, False)
    df = normalize_view_df(df, ["pure_log"], dtype=np.float64)

    rows = [4, 5, 13, 16, 27]
    fig, axs = plt.subplots(3, len(rows), figsize=(20, 13))
    for i, idx in enumerate(rows):
        row = df.iloc[idx]
        for j, v in enumerate(["_max_pp_pure_log", "_max_np_pure_log", "_pure_log"]):
            pcm = axs[j, i].imshow(
                row[f"view{v}"], cmap=mymap, norm=mpl.colors.CenteredNorm()
            )
            if i == axs.shape[1] - 1:
                fig.colorbar(pcm, ax=axs[j, i], fraction=0.046, pad=0.10)
            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])

    axs[0, 0].set_ylabel(r"Pooling feature $\mathrm{V}_{ij1}$")
    axs[1, 0].set_ylabel(r"Pooling feature $\mathrm{V}_{ij2}$")
    axs[2, 0].set_ylabel(r"Pooling feature $\mathrm{V}_{ij3}$")

    for i in range(len(rows)):
        axs[0, i].set_title(f"Example {i+1}")

    if not save_path is None:
        plt.savefig(f"{save_path}/views.png")


if __name__ == "__main__":
    app = "testcase1-diffusion-unstructured"
    feature_maps_viz(
        constants.DATA_PATH + "/processed/testcase2.1-diffusion-singlecell/train",
        constants.DATA_PATH + f"/models/{app}",
        "model2024-04-17_08-14-43",
        constants.DATA_PATH + "/processed/testcase2.1-diffusion-singlecell/dump"
    )
    # pooling_visualization(
    #     constants.DATA_PATH + "/processed/testcase1-diffusion-unstructured/train"
    # )
