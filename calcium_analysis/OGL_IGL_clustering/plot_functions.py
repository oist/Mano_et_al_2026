import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, wilcoxon
import warnings
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as sch

def characterize_neurons(df_r, df_e):
    """
    Characterize neural activity for each neuron across multiple sessions and update summary metrics.

    This function computes a set of features (alpha, beta, gamma, and rho) to describe the activity
    of each neuron based on trial-averaged calcium signals. It uses Wilcoxon signed-rank tests
    to evaluate stimulus-driven activation, and computes:
        - Alpha: average response change after stimulus onset
        - Beta: post-response linear slope (trend)
        - Gamma: latency to peak (or trough) in the post-stimulus window
        - Rho: correlation between baseline and delayed response epochs

    Parameters:
    ----------
    df_r : pandas.DataFrame
    df_e : pandas.DataFrame
    """
    def _pvalue_alpha(signal, i):
        s = 420 * i
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            baseline = np.nanmean(signal[:, s:s+40], axis=1)
            response = np.nanmean(signal[:, s+40:s+120], axis=1)
        bad_mask = np.isnan(baseline) | np.isnan(response)
        baseline = baseline[~bad_mask]
        response = response[~bad_mask]
        stat, pval = wilcoxon(response - baseline)
        return pval
    
    def _compute_alpha(signal, i):
        s = 420 * i
        alpha = []
        b = np.mean(signal[s:s+40]) # baseline
        r = np.mean(signal[s+40:s+120]) # response
        return r - b

    def _compute_beta(d, i):
        s = 420 * i
        slope, _ = np.polyfit(np.arange(160), d[s+240:s+400], 1)
        return slope

    def _compute_gamma(d, i):
        s = 420 * i
        a = d[s:s+40]
        b = d[s+40:s+120]
        if b.mean() > a.mean():
            peak_arg = np.argmax(b)
            # gamma = 1 / peak_arg if peak_arg != 0 else np.nan
            gamma = peak_arg if peak_arg != 0 else np.nan
        else:
            peak_arg = np.argmin(b)
            # gamma = (-1) * np.abs(1 / peak_arg) if peak_arg != 0 else np.nan
            gamma = peak_arg if peak_arg != 0 else np.nan
        return gamma

    def _compute_rho(d):
        rho, _ = pearsonr(d[0:200], d[420:620])
        return rho

    recordings = df_r["recording"].sort_values().unique()
    for rec in recordings:
        neuron_ids = df_r[df_r["recording"]==rec]["neuron"]
        tmp1 = df_e.xs(rec, level="recording")
        for n in neuron_ids:
            M = []
            df = tmp1.xs(n, level="neuron")
            for s in range(3):
                tmp2 = df.xs(s, level="session")
                mat = tmp2.reset_index().pivot(index="trial", columns="time", values="y").to_numpy()
                M.append(mat)
            M = np.hstack(M)
            M = M[~np.all(np.isnan(M), axis=1)] # remove trials with NaN values
            mean = np.nanmean(M, axis=0)

            new_data = {
                "pval_alpha1": _pvalue_alpha(M, 0),
                "pval_alpha2": _pvalue_alpha(M, 1),
                "pval_alpha3": _pvalue_alpha(M, 2),
                "alpha1": _compute_alpha(mean, 0),
                "alpha2": _compute_alpha(mean, 1),
                "alpha3": _compute_alpha(mean, 2),
                "beta1": _compute_beta(mean, 0),
                "gamma1": _compute_gamma(mean, 0),
                "rho": _compute_rho(mean)
            }

            idx = df_r.index[(df_r['recording'] == rec) & (df_r['neuron'] == n)]
            for key, val in new_data.items():
                df_r.loc[idx, key] = val

def plot_index(df, cluster_cmap, figsize=(2,5), figsavename=None):

    cluster_ids = df["cluster"].sort_values().unique()
    num_clusters = cluster_ids.size

    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(num_clusters, 6, wspace=0.05, hspace=0.1)

    # Cluster Color Annotations
    for (k, c) in enumerate(cluster_ids):
        ax = fig.add_subplot(gs[k, 0])
        circle = mpatches.Circle((1, 1), 0.5, linewidth=0, facecolor=cluster_cmap[c])
        ax.add_patch(circle)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')
        ax.set_ylabel(f"{c:.0f}", labelpad=2)
        ax.yaxis.label.set(rotation='horizontal', ha='right', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ["top", "right", "bottom", "left"]:
            ax.spines[sp].set_visible(False)

    for (k, c) in enumerate(cluster_ids):
        sub_df = df[df["cluster"] == c]
        for j in range(5):
            if j == 0:
                key = "alpha1"
                raw_metric = sub_df[key].to_numpy()
                ref_99 = np.percentile(df[key], 99)
                ref_1 = np.abs(np.percentile(df[key], 1))
                normalized = np.where(
                    raw_metric >= 0,
                    raw_metric / ref_99,
                    raw_metric / ref_1
                )
                metric = np.clip(normalized, -1, 1)
                title = r'$\alpha_1$'
            elif j == 1:
                key = "alpha2"
                raw_metric = sub_df[key].to_numpy()
                ref_99 = np.percentile(df[key], 99)
                ref_1 = np.abs(np.percentile(df[key], 1))
                normalized = np.where(
                    raw_metric >= 0,
                    raw_metric / ref_99,
                    raw_metric / ref_1
                )
                metric = np.clip(normalized, -1, 1)
                title = r'$\alpha_2$'
            elif j == 2:
                key = "alpha3"
                raw_metric = sub_df[key].to_numpy()
                ref_99 = np.percentile(df[key], 99)
                ref_1 = np.abs(np.percentile(df[key], 1))
                normalized = np.where(
                    raw_metric >= 0,
                    raw_metric / ref_99,
                    raw_metric / ref_1
                )
                metric = np.clip(normalized, -1, 1)
                title = r'$\alpha_3$'
            elif j == 3:
                key = "beta1"
                raw_metric = sub_df[key].to_numpy()
                ref_99 = np.percentile(df[key], 99)
                ref_1 = np.abs(np.percentile(df[key], 1))
                normalized = np.where(
                    raw_metric >= 0,
                    raw_metric / ref_99,
                    raw_metric / ref_1
                )
                metric = np.clip(normalized, -1, 1)
                title = r'$\beta_1$'
            # elif j == 4:
            #     key = "gamma1"
            #     metric = 2 * (sub_df[key] - np.nanpercentile(df[key],1)) / (np.nanpercentile(df[key],99) - np.nanpercentile(df[key],1)) - 1
            #     metric = np.clip(metric, -1, 1)
            #     metric = metric[sub_df["pval_alpha1"] < 0.01]
            #     title = r'$\gamma_1$'
            elif j == 4:
                metric = sub_df["rho"]
                title = r'$\rho$'
            h, bin_edges = np.histogram(metric, range=[-1,1], bins=10, density=False)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            ax = fig.add_subplot(gs[k, j+1])
            pos = ax.get_position()
            ax.set_position([pos.x0+0.005, pos.y0 + 0.005, pos.width-0.01, pos.height-0.01])
            ax.bar(bin_centers, h, width=bin_edges[1]-bin_edges[0], align="center", lw=0, alpha=1, color="gray")
            ax.set_xlim(-1,1)
            ax.set_ylim(0, h.max()+1)
            ax.set_yticks([])
            ax.set_xticks([-1,0,1])
            ax.set_xticklabels([])
            ax.tick_params(axis='x', length=1.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            if k == 0:
                ax.set_xlabel(title, labelpad=5)
                ax.xaxis.set_label_position('top')  # Moves the x-axis label to the top

    if figsavename:
        plt.savefig(figsavename)
        print(f"Figure saved as {figsavename}")
    plt.show()

def plot_index2(df, cluster_cmap, cmap, figsize=(2,5), figsavename=None):

    cluster_ids = df["cluster"].sort_values().unique()
    num_clusters = cluster_ids.size

    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(num_clusters, 3, width_ratios=[0.6, 1, 0.6], wspace=0.05, hspace=0.1)

    # Cluster Color Annotations
    for (k, c) in enumerate(cluster_ids):
        ax = fig.add_subplot(gs[k, 0])
        circle = mpatches.Circle((1, 1), 0.5, linewidth=0, facecolor=cluster_cmap[c])
        ax.add_patch(circle)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')
        ax.set_ylabel(f"{c:.0f}", labelpad=2)
        ax.yaxis.label.set(rotation='horizontal', ha='right', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ["top", "right", "bottom", "left"]:
            ax.spines[sp].set_visible(False)

    # Draw dendrogram
    cols = [col for col in df.columns if col.startswith("spca_")]
    # means = df.filter(regex="^spca_").groupby('cluster').mean()
    means = df.groupby('cluster')[cols].mean()
    D = dist.pdist(means, metric="cosine")
    Z = sch.linkage(D, method="complete")
    ax = fig.add_subplot(gs[:,2])
    with plt.rc_context({'lines.linewidth': 0.5}):
        sch.dendrogram(Z, orientation='left', ax=ax, no_labels=True, color_threshold=0, above_threshold_color='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.invert_xaxis()
    for sp in ["top", "right", "bottom", "left"]:
        ax.spines[sp].set_visible(False)

    # draw barcode
    for (k, c) in enumerate(cluster_ids):
        sub_df = df[df["cluster"] == c]

        # compute barcode
        key = "alpha1"
        raw_metric = sub_df[key].to_numpy()
        ref_99 = np.percentile(df[key], 99)
        ref_1 = np.abs(np.percentile(df[key], 1))
        normalized = np.where(
            raw_metric >= 0,
            raw_metric / ref_99,
            raw_metric / ref_1
        )
        alpha1 = np.median(normalized)

        key = "alpha2"
        raw_metric = sub_df[key].to_numpy()
        ref_99 = np.percentile(df[key], 99)
        ref_1 = np.abs(np.percentile(df[key], 1))
        normalized = np.where(
            raw_metric >= 0,
            raw_metric / ref_99,
            raw_metric / ref_1
        )
        alpha2 = np.median(normalized)

        key = "alpha3"
        raw_metric = sub_df[key].to_numpy()
        ref_99 = np.percentile(df[key], 99)
        ref_1 = np.abs(np.percentile(df[key], 1))
        normalized = np.where(
            raw_metric >= 0,
            raw_metric / ref_99,
            raw_metric / ref_1
        )
        alpha3 = np.median(normalized)

        key = "beta1"
        raw_metric = sub_df[key].to_numpy()
        ref_99 = np.percentile(df[key], 99)
        ref_1 = np.abs(np.percentile(df[key], 1))
        normalized = np.where(
            raw_metric >= 0,
            raw_metric / ref_99,
            raw_metric / ref_1
        )
        beta1 = np.median(normalized)

        rho = np.median(sub_df["rho"])

        barcode = np.array([alpha1, alpha2, alpha3, beta1, rho])
        barcode = barcode[np.newaxis, :]

        ax = fig.add_subplot(gs[k,1])
        pos = ax.get_position()
        ax.set_position([pos.x0+0.005, pos.y0 + 0.005, pos.width-0.01, pos.height-0.01])
        x = np.arange(barcode.shape[1] + 1)
        y = np.arange(barcode.shape[0] + 1)
        ax.pcolormesh(x, y, barcode, cmap=cmap, vmin=-1, vmax=1, edgecolors='w', linewidth=0.25)
        for v in range(barcode.shape[1]):
            ax.axvline(x=v, color='w', linewidth=1)
        ax.set_yticks([])
        ax.set_xticks([])
        for sp in ['top', 'right', 'left', 'bottom']:
            ax.spines[sp].set_visible(False)

    if figsavename:
        plt.savefig(figsavename)
        print(f"Figure saved as {figsavename}")
    plt.show()

def plot_cluster_average(df, cluster_cmap, stimulus_train, ymin=0, ymax=1, figsize=(5,5), figsavename=None):
    def _plot_stimulus_train(axis, t, y):
        axis.plot(t, y, color="gray", lw=1)
        axis.set_xlim([t[0], t[-1]])
        axis.set_ylim([-0.25, 1.25])
        axis.axis("off")

    def _configure_axis2(ax, label, ylim):
        ax.set_ylabel(label, labelpad=-2)
        ax.yaxis.label.set(rotation='horizontal', ha='right', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(ylim)
        for sp in ["top", "right", "bottom", "left"]:
            ax.spines[sp].set_visible(False)

    time_cols = [col for col in df.columns if col.startswith("time_")]
    means = df.groupby('cluster')[time_cols].mean()
    num_clusters = means.shape[0]
    t1 = np.arange(int(means.shape[1]/3))

    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(num_clusters+3, 5,
                          width_ratios=[0.02, 0.3, 0.3, 0.3, 0.1],
                          height_ratios=[0.6, 0.6] + [1,] * num_clusters + [0.8],
                          wspace=0.1, hspace=0.1)

    # Draw dendrogram
    D = dist.pdist(means, metric="cosine")
    Z = sch.linkage(D, method="complete")
    ax = fig.add_subplot(gs[2:num_clusters+2,4])
    with plt.rc_context({'lines.linewidth': 0.5}):
        sch.dendrogram(Z, orientation='left', ax=ax, no_labels=True, color_threshold=0, above_threshold_color='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.invert_xaxis()
    for sp in ["top", "right", "bottom", "left"]:
        ax.spines[sp].set_visible(False)

    # show cluster mean
    for (k, c) in enumerate(sorted(df["cluster"].unique())):
        for i in range(3):
            ax = fig.add_subplot(gs[k+2, i+1])
            y = means.loc[c][i * (420):(i + 1) * (420)]
            ax.plot(t1, y, color="k", zorder=11, linewidth=1)
            ax.fill_between(t1, ymin, y, color=cluster_cmap[c], zorder=10, alpha=0.8, linewidth=0, antialiased=True)
            ax.vlines([40, 120, 200, 240, 400], ymin, ymax, colors="lightgray", linestyles="dashed", lw=0.5)
            ax.set_xlim([t1[0], t1[-1]])
            ax.set_ylim([ymin, ymax])
            ax.axis("off")

    # show stimulus train
    t2 = np.arange(stimulus_train.size)
    _plot_stimulus_train(fig.add_subplot(gs[0, 1]), t2, stimulus_train)
    _plot_stimulus_train(fig.add_subplot(gs[1, 1]), t2, np.zeros_like(t2))
    _plot_stimulus_train(fig.add_subplot(gs[0, 2]), t2, stimulus_train)
    _plot_stimulus_train(fig.add_subplot(gs[1, 2]), t2, np.ones_like(t2))
    _plot_stimulus_train(fig.add_subplot(gs[0, 3]), t2, np.ones_like(t2))
    _plot_stimulus_train(fig.add_subplot(gs[1, 3]), t2, 1- stimulus_train)

    for i in range(num_clusters):
        ax = fig.add_subplot(gs[i+2, 0])
        _configure_axis2(ax, f"{i+1}", [0, 1])

    ax = fig.add_subplot(gs[0, 0])
    _configure_axis2(ax, "Int.", [-0.25, 1.25])
    ax = fig.add_subplot(gs[1, 0])
    _configure_axis2(ax, "DoLP", [-0.25, 1.25])

    ax = fig.add_subplot(gs[num_clusters+2, 1])
    ax.plot([t2[0], t2[5000]], [-0.5,-0.5], color='black', lw=2)
    ax.text(t2[2500], 0.5, '5 sec', color='black', ha='center', va='center', rotation=0)
    ax.set_xlim(t2[0], t2[-1])
    ax.set_ylim(-1, 1)
    ax.axis("off")

    if figsavename:
        plt.savefig(figsavename)
        print(f"Figure saved as {figsavename}")
    plt.show()

def plot_clustering(df, cluster_cmap, stimulus_train, vmin=0, vmax=1, sort_within_cluster=False,
                    colormap_name="viridis", figsize=(5,5), figsavename=None):
    def _plot_stimulus_train(axis, t, y):
        axis.plot(t, y, color="gray", lw=0.75)
        axis.set_xlim([t[0], t[-1]])
        axis.set_ylim([-0.25, 1.25])
        axis.axis("off")

    def _configure_axis(ax, label, yticks, ylim):
        ax.set_ylabel(label, labelpad=-10)
        ax.set_yticks(yticks)
        ax.tick_params(axis='y', pad=2)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        for sp in ["top", "right", "bottom"]:
            ax.spines[sp].set_visible(False)

    def _configure_axis2(ax, label, ylim):
        ax.set_ylabel(label, labelpad=-2)
        ax.yaxis.label.set(rotation='horizontal', ha='right', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(ylim)
        for sp in ["top", "right", "bottom", "left"]:
            ax.spines[sp].set_visible(False)

    df_sorted = df.sort_values(by='cluster')
    data = df_sorted.filter(regex="^time_").to_numpy()
    cluster_ids = df_sorted["cluster"]

    if sort_within_cluster:
        data_sorted = np.zeros_like(data)
        for c in np.unique(cluster_ids):
            m = data[cluster_ids == c]
            D = dist.pdist(m, metric="cosine")
            Z = sch.linkage(D, method="complete")
            row_order = sch.leaves_list(Z)
            data_sorted[cluster_ids == c] = m[row_order]
        data = data_sorted

    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(4, 5, width_ratios=[0.02, 0.3, 0.3, 0.3, 0.03], height_ratios=[0.02, 0.02, 1, 0.04], wspace=0.05, hspace=0.05)

    # Cluster Color Annotations
    ax = fig.add_subplot(gs[2, 4])
    for k, i in enumerate(cluster_ids):
        ax.add_patch(mpatches.Rectangle((0, data.shape[0] - k - 1), 1, 1, color=cluster_cmap[i], lw=0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, data.shape[0])
    ax.axis('off')

    # plot stimulus train
    t = np.arange(stimulus_train.size)
    _plot_stimulus_train(fig.add_subplot(gs[0, 1]), t, stimulus_train)
    _plot_stimulus_train(fig.add_subplot(gs[1, 1]), t, np.zeros_like(t))
    _plot_stimulus_train(fig.add_subplot(gs[0, 2]), t, stimulus_train)
    _plot_stimulus_train(fig.add_subplot(gs[1, 2]), t, np.ones_like(t))
    _plot_stimulus_train(fig.add_subplot(gs[0, 3]), t, np.ones_like(t))
    _plot_stimulus_train(fig.add_subplot(gs[1, 3]), t, 1- stimulus_train)

    # Plot Heatmaps
    for i in range(3):
        ax = fig.add_subplot(gs[2, 1 + i])
        ax.imshow(data[:, i * (420):(i + 1) * (420)], cmap=colormap_name, vmin=vmin, vmax=vmax, aspect='auto', interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

    # Draw the scale bar
    ax = fig.add_subplot(gs[3, 1])
    ax.plot([t[0], t[5000]], [-0.5,-0.5], color='black', lw=2)
    ax.text(t[2500], 0.6, '5 sec', color='black', ha='center', va='center', rotation=0)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-1, 1)
    ax.axis("off")

    # axis labels
    ax = fig.add_subplot(gs[2, 0])
    _configure_axis(ax, "Neurons", [0, data.shape[0]], [data.shape[0], 0])
    ax = fig.add_subplot(gs[0, 0])
    _configure_axis2(ax, "Int.", [-0.25, 1.25])
    ax = fig.add_subplot(gs[1, 0])
    _configure_axis2(ax, "DoLP", [-0.25, 1.25])

    if figsavename:
        plt.savefig(figsavename)
        print(f"Figure saved as {figsavename}")
    plt.show()

def plot_spca_features(df, cluster_cmap, vmin=0, vmax=1, sort_within_cluster=False,
                      colormap_name="viridis", figsize=(5,5), figsavename=None):

    def _configure_axis(ax, label, yticks, ylim):
        ax.set_ylabel(label, labelpad=-10)
        ax.set_yticks(yticks)
        ax.tick_params(axis='y', pad=2)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        for sp in ["top", "right", "bottom"]:
            ax.spines[sp].set_visible(False)

    df_sorted = df.sort_values(by='cluster')
    data = df_sorted.filter(regex="^spca_").to_numpy()
    cluster_ids = df_sorted["cluster"]

    if sort_within_cluster:
        data_sorted = np.zeros_like(data)
        for c in np.unique(cluster_ids):
            m = data[cluster_ids == c]
            D = dist.pdist(m, metric="euclidean")
            Z = sch.linkage(D, method="average")
            row_order = sch.leaves_list(Z)
            data_sorted[cluster_ids == c] = m[row_order]
        data = data_sorted

    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.02, 0.3, 0.03], wspace=0.05, hspace=0.05)

    # Cluster Color Annotations
    ax = fig.add_subplot(gs[0, 2])
    for k, i in enumerate(cluster_ids):
        ax.add_patch(mpatches.Rectangle((0, data.shape[0] - k - 1), 1, 1, color=cluster_cmap[i], lw=0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, data.shape[0])
    ax.axis('off')

    # Plot Heatmaps
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(data, cmap=colormap_name, vmin=vmin, vmax=vmax, aspect='auto', interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    # axis labels
    ax = fig.add_subplot(gs[0, 0])
    _configure_axis(ax, "Neurons", [0, data.shape[0]], [data.shape[0], 0])

    if figsavename:
        plt.savefig(figsavename)
        print(f"Figure saved as {figsavename}")
    plt.show()
