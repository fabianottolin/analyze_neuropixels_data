# visualization_functions

import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.widgets as si_widgets


def show_outside_channels(raw, fs, channel_labels, xfeats):
    """
    # modified from ibldsp.plots
    """
    from ibldsp.plots import voltageshow
    
    nc, ns = raw.shape
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # removes DC offset
    ns_plot = np.minimum(ns, 3000)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 8, 0.2]})

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


def curation_plot(sorting_analyzer, curation_method, curation_labels, output_folder):
    curation_plot = si_widgets.plot_unit_labels(sorting_analyzer, curation_labels, ylims=(-300, 100))
    curation_plot.figure.suptitle(f"{curation_method} label")
    plt.show()
    filepath = output_folder.figures_folder/f"curation_{curation_method}"/f"{output_folder.recording_identifier}_curation.png"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    curation_plot.figure.savefig(filepath, dpi=300)
    print(f"Figure saved under '{filepath}'")

    # if method bombcell theres more plots I could add in future