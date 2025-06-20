import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from scipy.signal import butter, filtfilt, chirp

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
