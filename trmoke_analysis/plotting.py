# Generic plotting scripts for some easy overview plots

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# --------------------------------
# Loopwise overview plot functions
# --------------------------------


def plot_loopwise_moke_pp(processed_dataset: xr.Dataset, detector_idx: int = 0):
    """
    Create loopwise overview plots for MOKE pump-probe data.
    """
    scan_param = processed_dataset.attrs["scan_param"]
    param1 = processed_dataset.attrs["param1"]
    param2 = processed_dataset.attrs["param2"]

    for i, j in np.ndindex(len(processed_dataset[param1].values), len(processed_dataset[param2].values)):
        fig, ax = plt.subplots(2, 1, dpi=300, figsize=(5, 5), sharex=True)

        # Create selection dictionary for isel
        sel_dict = {"detector": detector_idx, param1: i, param2: j}

        # Plot sum (symmetric component) - loopwise
        processed_dataset.signal_sum_loopwise.isel(**sel_dict).plot.line(x=scan_param, ax=ax[0], alpha=0.75)
        # Overlay average
        processed_dataset.signal_sum.isel(**sel_dict).plot.line(
            x=scan_param, ax=ax[0], color="dimgray", linewidth=2, label="Average", alpha=0.5
        )

        # Plot diff (antisymmetric component) - loopwise
        processed_dataset.signal_diff_loopwise.isel(**sel_dict).plot.line(x=scan_param, ax=ax[1], alpha=0.75)
        # Overlay average
        processed_dataset.signal_diff.isel(**sel_dict).plot.line(
            x=scan_param, ax=ax[1], color="dimgray", linewidth=2, label="Average", alpha=0.5
        )

        ax[1].set_title(None)
        ax[0].set_xlabel(None)
        ax[0].set_ylabel("signal_sum_loopwise")
        ax[1].set_ylabel("signal_diff_loopwise")

        ax[0].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
        ax[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)

        ax[0].legend(ncols=4)
        ax[1].legend(ncols=4)

        plt.tight_layout()
        plt.show()


def plot_loopwise_hysteresis(processed_dataset: xr.Dataset, detector_idx: int = 0):
    """
    Create loopwise overview plots for hysteresis data.
    """
    scan_param = processed_dataset.attrs["scan_param"]
    param1 = processed_dataset.attrs["param1"]
    param2 = processed_dataset.attrs["param2"]

    for i, j in np.ndindex(len(processed_dataset[param1].values), len(processed_dataset[param2].values)):
        fig, ax = plt.subplots(2, 1, dpi=300, figsize=(5, 5), sharex=True)

        # Create selection dictionary for isel
        sel_dict = {"detector": detector_idx, param1: i, param2: j}

        # Plot up and down signals - loopwise
        processed_dataset.signal_up_pumped_loopwise.isel(**sel_dict).plot.line(x=scan_param, ax=ax[0], alpha=0.75)
        processed_dataset.signal_down_pumped_loopwise.isel(**sel_dict).plot.line(x=scan_param, ax=ax[0], alpha=0.75)
        processed_dataset.signal_up_unpumped_loopwise.isel(**sel_dict).plot.line(x=scan_param, ax=ax[1], alpha=0.75)
        processed_dataset.signal_down_unpumped_loopwise.isel(**sel_dict).plot.line(x=scan_param, ax=ax[1], alpha=0.75)

        # Overlay averages
        processed_dataset.signal_up_pumped.isel(**sel_dict).plot.line(
            x=scan_param, ax=ax[0], color="dimgray", linewidth=2, label="Up Pumped Avg", alpha=0.5
        )
        processed_dataset.signal_down_pumped.isel(**sel_dict).plot.line(
            x=scan_param, ax=ax[0], color="dimgray", linewidth=2, label="Down Pumped Avg", alpha=0.5
        )
        processed_dataset.signal_up_unpumped.isel(**sel_dict).plot.line(
            x=scan_param, ax=ax[1], color="dimgray", linewidth=2, label="Up Unpumped Avg", alpha=0.5
        )
        processed_dataset.signal_down_unpumped.isel(**sel_dict).plot.line(
            x=scan_param, ax=ax[1], color="dimgray", linewidth=2, label="Down Unpumped Avg", alpha=0.5
        )

        ax[1].set_title(None)
        ax[0].set_xlabel(None)
        ax[0].set_ylabel("signal (pumped)")
        ax[1].set_ylabel("signal (unpumped)")

        ax[0].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
        ax[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
        ax[0].axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
        ax[1].axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)

        ax[0].legend(ncols=4)
        ax[1].legend(ncols=4)

        plt.tight_layout()
        plt.show()


# --------------------------------
# Averaged overview plot functions
# --------------------------------


def plot_diode_signal(processed_dataset: xr.Dataset):
    """
    Plot diode signal for pump-probe data.
    """
    fig, ax = plt.subplots(dpi=300, figsize=(5, 2.5))
    processed_dataset.diode_signal.plot.line(ax=ax)
    ax.grid()
    plt.tight_layout()
    plt.show()


def plot_overview_moke_pp(processed_dataset: xr.Dataset, detector_idx: int = 0):
    """
    Create overview plots for MOKE pump-probe data.
    """
    scan_param = processed_dataset.attrs["scan_param"]
    param1 = processed_dataset.attrs["param1"]

    for param1_idx in range(len(processed_dataset[param1].values)):
        fig, ax = plt.subplots(2, 1, dpi=300, figsize=(5, 5), sharex=True)

        # Create selection dictionary
        sel_dict = {"detector": detector_idx, param1: param1_idx}

        # Plot sum (symmetric component)
        processed_dataset.signal_sum.isel(**sel_dict).plot.line(x=scan_param, ax=ax[0], add_legend=True)
        # Plot diff (antisymmetric component)
        processed_dataset.signal_diff.isel(**sel_dict).plot.line(x=scan_param, ax=ax[1], add_legend=True)

        ax[1].set_title(None)
        ax[0].set_xlabel(None)
        ax[0].set_ylabel("signal_sum")
        ax[1].set_ylabel("signal_diff")

        ax[0].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
        ax[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
        ax[0].axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
        ax[1].axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)

        plt.tight_layout()
        plt.show()


def plot_overview_hysteresis(processed_dataset: xr.Dataset, detector_idx: int = 0):
    """
    Create overview plots for hysteresis data.
    """
    scan_param = processed_dataset.attrs["scan_param"]
    param1 = processed_dataset.attrs["param1"]

    for param1_idx in range(len(processed_dataset[param1].values)):
        fig, ax = plt.subplots(2, 1, dpi=300, figsize=(5, 5), sharex=True)

        # Create selection dictionary
        sel_dict = {"detector": detector_idx, param1: param1_idx}

        # Plot up and down signals - pumped
        processed_dataset.signal_up_pumped.isel(**sel_dict).plot.line(x=scan_param, ax=ax[0], add_legend=True)
        processed_dataset.signal_down_pumped.isel(**sel_dict).plot.line(x=scan_param, ax=ax[0], add_legend=True)
        # Plot up and down signals - unpumped
        processed_dataset.signal_up_unpumped.isel(**sel_dict).plot.line(x=scan_param, ax=ax[1], add_legend=True)
        processed_dataset.signal_down_unpumped.isel(**sel_dict).plot.line(x=scan_param, ax=ax[1], add_legend=True)

        ax[1].set_title(None)
        ax[0].set_xlabel(None)
        ax[0].set_ylabel("signal (pumped)")
        ax[1].set_ylabel("signal (unpumped)")

        ax[0].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
        ax[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
        ax[0].axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
        ax[1].axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)

        plt.tight_layout()
        plt.show()
