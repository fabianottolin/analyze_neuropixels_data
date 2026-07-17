# visualization_functions

import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.widgets as si_widgets


def show_outside_channels(raw_recording, outside_boundary, xfeats, detection_method = "ibl", sample_duration = 5):
    """
    # modified from ibldsp.plots

    ------ Parameters ------
    xfeats:
        Features used to determine outside channels
    """
    from ibldsp.plots import voltageshow

    start_sample = np.random.randint(0, int(raw_recording.ns - sample_duration * raw_recording.fs)) # random start sample for batch
    raw = raw_recording[start_sample:start_sample + int(sample_duration * raw_recording.fs), :raw_recording.nc - raw_recording.nsync].T # random 5s sample
    
    n_channels, n_samples = raw.shape
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # removes DC offset
    ns_plot = n_samples #np.minimum(n_samples, 3000)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 8, 0.2]})

    match detection_method:
        case "ibl":
            relevant_feature = xfeats["xcor_lf"]
            feature_label = "Low frequency coherence"
        case "manual":
            relevant_feature = np.zeros(n_channels)
            feature_label = "Manually selected"
            ax[0].set(xticks=[], xticklabels=[])
        case _:
            raise ValueError("Unknown detection method, choose either 'ibl' or 'manual'")

    if outside_boundary is None:
        iko = np.zeros(n_channels, dtype=bool)
    else:
        iko = np.arange(n_channels) >= outside_boundary

    ax[0].plot(relevant_feature, np.arange(n_channels), color="dimgray")
    ax[0].plot(relevant_feature[iko], np.arange(n_channels)[iko], "y*")
    ax[0].set(ylabel="Channel #", ylim=[0, n_channels], xlabel=feature_label, title="Outside channels")
    voltageshow(raw[:, :ns_plot], raw_recording.fs, ax=ax[1], cax=ax[2])
    ax[1].sharey(ax[0])
    ax[1].set_ylabel("")
    ax[1].set_title("LFP voltage")
    ax[1].set_xticklabels([round(i + start_sample/raw_recording.fs, 0) for i in ax[1].get_xticks()])
    fig.suptitle("Outside channel detection", fontsize = 18)
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