# visualization_functions

import matplotlib.pyplot as plt
import numpy as np

# modified from ibldsp.plots
def show_outside_channels(raw, fs, channel_labels, xfeats):
    """
    # modified from ibldsp.plots
    """
    from ibldsp.plots import voltageshow
    
    nc, ns = raw.shape
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # removes DC offset
    ns_plot = np.minimum(ns, 3000)
    fig, ax = plt.subplots(
        1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 8, 0.2]}
    )

    ax[0].plot(xfeats["xcor_lf"], np.arange(nc))
    ax[0].plot(xfeats["xcor_lf"][(iko := channel_labels == 3)], np.arange(nc)[iko], "y*")
    ax[0].set(ylabel="channel #", ylim=[0, nc],
        xlabel="LF coherence",
        title="outside")
    ax[0].sharey(ax[0])
    voltageshow(raw[:, :ns_plot], fs, ax=ax[1], cax=ax[2])
    ax[1].sharey(ax[0])
    fig.tight_layout()
    return fig, ax