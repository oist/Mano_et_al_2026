import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from scipy.signal import butter, filtfilt, chirp
import matplotlib.patches as mpatches

def compute_qi_custom(df, neuron_id, session):
    """
    Compute the quality index (QI) of a specified neuron
    """
    df_filtered = df.xs((session, neuron_id), level=["session", "neuron"])
    M = df_filtered.reset_index().pivot(index="trial", columns="time", values="y").to_numpy()
    M = M[~np.all(np.isnan(M), axis=1)] # remove trials with NaN values
    M = M[:, 0:200]
    a = np.var(np.nanmean(M, axis=0)) # variance of the trial-averaged response over time
    b = np.mean(np.nanvar(M, axis=1))
    qi = a / b
    return qi

def compute_response_index(x, cutoff, fs, order=4, sigma=5):
    # high pass filter
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, x)
    y = ndi.gaussian_filter1d(y, sigma)

    return np.percentile(y, 95) - np.percentile(y, 5)


def make_plot(df, neuron_id, stimulus_train, vmin=-0.05, vmax=0.25, cmap_name="bone", figsize=(10,2.5), figsavename=None, show_plot=True):
    t = df.index.get_level_values("time").unique()
    t /= 1000 # convert msec to sec

    fig, axs = plt.subplots(4, 6, figsize=figsize, dpi=300, gridspec_kw={'height_ratios': [1, 1, 0.2, 0.2]})

    # for vertical polarization
    for i in range(6):
        df_filtered = df.xs((i, neuron_id), level=["session", "neuron"])
        mat = df_filtered.reset_index().pivot(index="trial", columns="time", values="y").to_numpy()
        axs[0,i].imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap_name, aspect="auto", interpolation="nearest")
        if i == 0:
            axs[0,i].set_ylabel("Trials")
        axs[0,i].set_xticks([])

        mean = np.nanmean(mat, axis=0)
        error = np.nanstd(mat, axis=0)
        mean = ndi.uniform_filter1d(mean, size=3)
        error = ndi.uniform_filter1d(error, size=3)
        axs[1,i].plot(t, mean, color="k", lw=1)
        axs[1,i].fill_between(t, mean-error, mean+error, color="k", alpha=0.2, linewidth=0, antialiased=True)
        if i == 0:
            axs[1,i].set_ylabel("Activity")
        axs[1,i].set_ylim([vmin, vmax])
        axs[1,i].set_xlim([t[0], t[-1]])
        axs[1,i].set_xticks([])
        for sp in ["top", "right", "bottom", "left"]:
            axs[1,i].spines[sp].set_visible(False)

    # plot luminance and polarization trace
    t2 = np.linspace(t[0], t[-1], stimulus_train.size)
    axs[2,0].plot(t2, stimulus_train, color="gray", lw=1)
    axs[3,0].plot(t2, np.zeros_like(t2), color="gray", lw=1)
    axs[2,1].plot(t2, stimulus_train, color="gray", lw=1)
    axs[3,1].plot(t2, np.ones_like(t2), color="gray", lw=1)
    axs[2,2].plot(t2, np.ones_like(t2), color="gray", lw=1)
    axs[3,2].plot(t2, 1 - stimulus_train, color="gray", lw=1)
    axs[2,3].plot(t2, stimulus_train, color="gray", lw=1)
    axs[3,3].plot(t2, np.zeros_like(t2), color="gray", lw=1)
    axs[2,4].plot(t2, stimulus_train, color="gray", lw=1)
    axs[3,4].plot(t2, np.ones_like(t2), color="gray", lw=1)
    axs[2,5].plot(t2, np.ones_like(t2), color="gray", lw=1)
    axs[3,5].plot(t2, 1 - stimulus_train, color="gray", lw=1)

    for i in range(6):
        if i == 0:
            axs[2,i].set_ylabel("I")
        axs[2,i].set_yticks([0, 1])
        axs[2,i].set_xticks([])
        axs[2,i].set_xlim([t[0], t[-1]])
        axs[2,i].set_ylim([-0.25, 1.25])
        for sp in ["top", "right", "bottom", "left"]:
            axs[2,i].spines[sp].set_visible(False)

    for i in range(6):
        if i == 0:
            axs[3,i].set_ylabel("DoLP")
        axs[3,i].set_yticks([0, 1])
        axs[3,i].set_xlabel("Time (s)")
        axs[3,i].set_xlim([t[0], t[-1]])
        axs[3,i].set_ylim([-0.25, 1.25])
        for sp in ["top", "right", "bottom", "left"]:
            axs[3,i].spines[sp].set_visible(False)

    plt.tight_layout()
    if figsavename:
        plt.savefig(figsavename, bbox_inches='tight')
        print(f"Figure saved as {figsavename}")
    if show_plot:
        plt.show()
    plt.close()

def plot_clustering2(df, cluster_cmap, stimulus_train, vmin=0, vmax=1, dt=0.05, figsize=(5,5), figsavename=None):
    def _plot_stimulus_train(axis, t, y):
        axis.plot(t, y, color="gray", lw=0.75)
        axis.set_xlim([t[0], t[-1]])
        axis.set_ylim([-0.25, 1.25])
        axis.axis("off")

    def _configure_axis(ax, label, yticks, ylim, labelpad=0, aspect=10):
        ax.set_aspect(aspect)
        ax.set_ylabel(label, labelpad=labelpad)
        ax.set_yticks(yticks)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    df_sorted = df.sort_values(by='cluster')
    data = df_sorted.filter(regex="^time_").to_numpy()
    cluster_ids = df_sorted["cluster"]

    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(4, 6, width_ratios=[0.02, 0.3, 0.3, 0.3, 0.3, 0.04], height_ratios=[1, 0.04, 0.04, 0.04], wspace=0.05, hspace=0.1)

    # Cluster Color Annotations
    ax = fig.add_subplot(gs[0, 5])
    for k, i in enumerate(cluster_ids):
        ax.add_patch(mpatches.Rectangle((0, data.shape[0] - k - 1), 1, 1, color=cluster_cmap[i], lw=0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, data.shape[0])
    ax.axis('off')

    # Plot Heatmaps
    for i in range(4):
        ax = fig.add_subplot(gs[0, 1 + i])
        ax.imshow(data[:, i * (200):(i + 1) * (200)], cmap="bone_r", vmin=vmin, vmax=vmax, aspect='auto', interpolation="nearest")
        ax.axis("off")
    
    # plot stimulus train
    t = np.arange(stimulus_train.size)
    _plot_stimulus_train(fig.add_subplot(gs[1, 1]), t, stimulus_train)
    _plot_stimulus_train(fig.add_subplot(gs[2, 1]), t, np.ones_like(t))
    _plot_stimulus_train(fig.add_subplot(gs[1, 2]), t, np.ones_like(t))
    _plot_stimulus_train(fig.add_subplot(gs[2, 2]), t, 1- stimulus_train)
    _plot_stimulus_train(fig.add_subplot(gs[1, 3]), t, stimulus_train)
    _plot_stimulus_train(fig.add_subplot(gs[2, 3]), t, np.ones_like(t))
    _plot_stimulus_train(fig.add_subplot(gs[1, 4]), t, np.ones_like(t))
    _plot_stimulus_train(fig.add_subplot(gs[2, 4]), t, 1- stimulus_train)

    # Draw the scale bar
    ax = fig.add_subplot(gs[3, 1])
    ax.plot([t[0], t[5000]], [-0.5,-0.5], color='black', lw=2)
    ax.text(t[2500], 0.8, '5 sec', color='black', ha='center', va='center', rotation=0)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-1, 1)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 0])
    _configure_axis(ax, "Neurons", [0, data.shape[0]], [data.shape[0], 0], labelpad=-10)
    ax = fig.add_subplot(gs[1, 0])
    _configure_axis(ax, "", [0, 1], [-0.25, 1.25])
    ax = fig.add_subplot(gs[2, 0])
    _configure_axis(ax, "", [0, 1], [-0.25, 1.25])

    if figsavename:
        plt.savefig(figsavename)
        print(f"Figure saved as {figsavename}")
    plt.show()

def plot_cluster_average(df, cluster_cmap, stimulus_train, ymin=0, ymax=1, figsize=(5,5), figsavename=None):
    def _plot_stimulus_train(axis, t, y):
        axis.plot(t, y, color="gray", lw=0.75)
        axis.set_xlim([t[0], t[-1]])
        axis.set_ylim([-0.25, 1.25])
        axis.axis("off")

    def _configure_axis(ax, label, yticks, ylim, labelpad=0, aspect=10):
        ax.set_aspect(aspect)
        ax.set_ylabel(label, labelpad=labelpad)
        ax.set_yticks(yticks)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _configure_axis2(ax, label, labelpad=0, aspect=10):
        ax.set_aspect(aspect)
        ax.set_ylim(0,1)
        ax.set_xlim(0,1)
        ax.text(0.5, 0.5, label, ha="center", va="center")        
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ["top", "right", "bottom", "left"]:
            ax.spines[sp].set_visible(False)

    time_cols = [col for col in df.columns if col.startswith("time_")]
    means = df.groupby('cluster')[time_cols].mean()
    num_clusters = means.shape[0]
    t = np.arange(int(means.shape[1]/4))

    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(num_clusters+3, 6,
                          width_ratios=[0.05, 0.3, 0.3, 0.3, 0.3, 0.2],
                          height_ratios=[1,] * num_clusters + [0.5, 0.5, 0.5],
                          wspace=0.1, hspace=0.1)

    # Draw dendrogram
    D = dist.pdist(means, metric="cosine")
    Z = sch.linkage(D, method="complete")
    ax = fig.add_subplot(gs[0:num_clusters,5])
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
        for i in range(4):
            ax = fig.add_subplot(gs[k, i+1])
            y = means.loc[c][i * (200):(i + 1) * (200)]
            ax.plot(t, y, color="k", zorder=11, linewidth=1)
            ax.fill_between(t, ymin, y, color=cluster_cmap[c], zorder=10, alpha=0.8, linewidth=0, antialiased=True)
            ax.vlines([40, 120, 200, 240, 400], ymin, ymax, colors="lightgray", linestyles="dashed", lw=0.5)
            ax.set_xlim([t[0], t[-1]])
            ax.set_ylim([ymin, ymax])
            ax.axis("off")

    t = np.arange(stimulus_train.size)
    _plot_stimulus_train(fig.add_subplot(gs[num_clusters, 1]), t, stimulus_train)
    _plot_stimulus_train(fig.add_subplot(gs[num_clusters+1, 1]), t, np.ones_like(t))
    _plot_stimulus_train(fig.add_subplot(gs[num_clusters, 2]), t, np.ones_like(t))
    _plot_stimulus_train(fig.add_subplot(gs[num_clusters+1, 2]), t, 1- stimulus_train)
    _plot_stimulus_train(fig.add_subplot(gs[num_clusters, 3]), t, stimulus_train)
    _plot_stimulus_train(fig.add_subplot(gs[num_clusters+1, 3]), t, np.ones_like(t))
    _plot_stimulus_train(fig.add_subplot(gs[num_clusters, 4]), t, np.ones_like(t))
    _plot_stimulus_train(fig.add_subplot(gs[num_clusters+1, 4]), t, 1- stimulus_train)
    
    for i in range(num_clusters):
        ax = fig.add_subplot(gs[i, 0])
        _configure_axis2(ax, f"{i+1}", [0, 1])

    ax = fig.add_subplot(gs[num_clusters, 0])
    _configure_axis(ax, "", [0, 1], [-0.25, 1.25])
    ax = fig.add_subplot(gs[num_clusters+1, 0])
    _configure_axis(ax, "", [0, 1], [-0.25, 1.25])

    ax = fig.add_subplot(gs[num_clusters+2, 1])
    ax.plot([t[0], t[5000]], [-0.5,-0.5], color='black', lw=2)
    ax.text(t[2500], 0.5, '5 sec', color='black', ha='center', va='center', rotation=0)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-1, 1)
    ax.axis("off")

    if figsavename:
        plt.savefig(figsavename)
        print(f"Figure saved as {figsavename}")
    plt.show()
