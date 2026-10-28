# Generic plotting scripts for some easy overview plots

import re

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
        fig, ax = plt.subplots(2, 1, dpi=100, figsize=(5, 5), sharex=True)

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
        fig, ax = plt.subplots(2, 1, dpi=100, figsize=(5, 5), sharex=True)

        # Create selection dictionary for isel
        sel_dict = {"detector": detector_idx, param1: i, param2: j}

        # Plot up and down signals - loopwise
        processed_dataset.signal_up_pumped_loopwise.isel(**sel_dict).plot.line(x=scan_param, ax=ax[0], alpha=0.75)
        ax[0].set_prop_cycle(None)  # Reset color cycle so that up and down signals have same colors
        processed_dataset.signal_down_pumped_loopwise.isel(**sel_dict).plot.line(x=scan_param, ax=ax[0], alpha=0.75)
        processed_dataset.signal_up_unpumped_loopwise.isel(**sel_dict).plot.line(x=scan_param, ax=ax[1], alpha=0.75)
        ax[1].set_prop_cycle(None)  # Reset color cycle so that up and down signals have same colors
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
    fig, ax = plt.subplots(dpi=100, figsize=(5, 2.5))
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
        fig, ax = plt.subplots(2, 1, dpi=100, figsize=(5, 5), sharex=True)

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
        fig, ax = plt.subplots(2, 1, dpi=100, figsize=(5, 5), sharex=True)

        # Create selection dictionary
        sel_dict = {"detector": detector_idx, param1: param1_idx}

        # Plot up and down signals - pumped
        processed_dataset.signal_up_pumped.isel(**sel_dict).plot.line(x=scan_param, ax=ax[0], add_legend=True)
        ax[0].set_prop_cycle(None)  # Reset color cycle so that up and down signals have same colors
        processed_dataset.signal_down_pumped.isel(**sel_dict).plot.line(x=scan_param, ax=ax[0], add_legend=True)
        # Plot up and down signals - unpumped
        processed_dataset.signal_up_unpumped.isel(**sel_dict).plot.line(x=scan_param, ax=ax[1], add_legend=True)
        ax[1].set_prop_cycle(None)  # Reset color cycle so that up and down signals have same colors
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


# --------------------------------
# Calibration plot functions
# --------------------------------

# Simple View MOKE colour convention for detectors 1, 2, 3 (detector numbers in diode calibrations are 1-based)
_DETECTOR_COLORS = ("tab:blue", "tab:red", "tab:green")


def _calibration_title(attrs: dict, kind: str, title: str | None) -> str:
    """Return the given title, or build one from the kind, scan_id and timestamp stored in the file attributes."""
    if title is not None:
        return title
    return f"{kind} calibration {attrs.get('scan_id', '')} ({attrs.get('timestamp', '')})"


def plot_fluence_calibration(dataset: xr.Dataset, title: str | None = None):
    """
    Plot a fluence calibration: fluence and diode voltage against the waveplate angle, and fluence against
    the diode voltage together with the linear fit stored in the file (fluence = linearity_m * voltage + linearity_b).

    Parameters
    ----------
    dataset : xr.Dataset
        Fluence calibration as written by the trMOKE measurement software (e.g. prop.data.f0010)
    title : str, optional
        Figure title; defaults to the kind, scan_id and timestamp from the file attributes
    """
    attrs = dataset.attrs
    angle_unit = attrs.get("waveplate_angle_unit", "°")
    fluence_unit = attrs.get("fluence_unit", "mJ/cm²")
    voltage_unit = attrs.get("diode_voltage_unit", "V")

    angle = dataset.waveplate_angle.values
    fluence = dataset.fluence.values
    voltage = dataset.diode_voltage.values

    fig, ax = plt.subplots(2, 1, dpi=100, figsize=(5, 5))

    # Panel 0: fluence (left axis) and diode voltage (right axis) vs waveplate angle, as in the live plot
    ax[0].plot(angle, fluence, "o-", color="C0")
    ax[0].set_xlabel(f"waveplate angle ({angle_unit})")
    ax[0].set_ylabel(f"fluence ({fluence_unit})", color="C0")
    ax_voltage = ax[0].twinx()
    ax_voltage.plot(angle, voltage, "s--", color="C1")
    ax_voltage.set_ylabel(f"diode voltage ({voltage_unit})", color="C1")
    ax_voltage.tick_params(axis="y", labelcolor="C1")

    # Panel 1: fluence vs diode voltage with the stored linear fit (used for the fluence stabilisation)
    m, b = attrs["linearity_m"], attrs["linearity_b"]
    fit_x = np.array([voltage.min(), voltage.max()])
    ax[1].plot(voltage, fluence, "o", color="C0", label="data")
    ax[1].plot(fit_x, m * fit_x + b, "--", color="dimgray", label=f"fit: m = {m:.3g}, b = {b:.3g}")
    ax[1].set_xlabel(f"diode voltage ({voltage_unit})")
    ax[1].set_ylabel(f"fluence ({fluence_unit})")
    ax[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
    ax[1].legend()

    fwhm_x, fwhm_y = attrs["FWHM_x_µm"], attrs["FWHM_y_µm"]
    title = _calibration_title(attrs, "fluence", title)
    ax[0].set_title(f"{title}\nFWHM {fwhm_x:.0f}×{fwhm_y:.0f} µm, {attrs['repetition_rate']} Hz")

    plt.tight_layout()
    plt.show()


def plot_field_calibration(dataset: xr.Dataset, title: str | None = None):
    """
    Plot a field calibration: magnetic field of the up and down voltage sweep and their mean against the PSU voltage.

    Parameters
    ----------
    dataset : xr.Dataset
        Field calibration as written by the trMOKE measurement software (e.g. prop.data.b0010)
    title : str, optional
        Figure title; defaults to the kind, scan_id and timestamp from the file attributes
    """
    attrs = dataset.attrs
    voltage_unit = attrs.get("voltage_unit", "V")
    field_unit = attrs.get("field_unit", "mT")
    voltage = dataset.psu_voltage.values

    fig, ax = plt.subplots(dpi=100, figsize=(5, 3.5))
    ax.plot(voltage, dataset.magnetic_field_up.values, "o-", label="up")
    ax.plot(voltage, dataset.magnetic_field_down.values, "s-", label="down")
    ax.plot(voltage, dataset.magnetic_field.values, "-", color="dimgray", linewidth=2, alpha=0.6, label="mean")
    ax.set_xlabel(f"PSU voltage ({voltage_unit})")
    ax.set_ylabel(f"magnetic field ({field_unit})")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.legend()

    title = _calibration_title(attrs, "field", title)
    if attrs.get("field_sign_flipped", 0):
        title += " (sign flipped)"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_diode_calibration(tree: xr.DataTree | xr.Dataset, title: str | None = None):
    """
    Plot a diode calibration (Simple View MOKE): detector voltage with its standard error against the waveplate
    angle for every detector stored in the file, together with the linear fit (voltage = fit_m * angle + fit_b).

    Parameters
    ----------
    tree : xr.DataTree or xr.Dataset
        Diode calibration as loaded by the proposal (e.g. prop.data.d0010), one group per detector. A plain
        dataset with the variables at the root (legacy single-detector file) is plotted as one series.
    title : str, optional
        Figure title; defaults to the kind, scan_id and timestamp from the file attributes
    """
    if isinstance(tree, xr.DataTree):
        series = [
            (int(match.group(1)), node.to_dataset())
            for name, node in tree.children.items()
            if (match := re.fullmatch(r"detector_(\d+)", name))
        ]
        series.sort(key=lambda s: s[0])
        if not series:  # legacy flat file: the variables sit at the root
            series = [(int(tree.attrs.get("detector", 1)), tree.to_dataset())]
    else:
        series = [(int(tree.attrs.get("detector", 1)), tree)]

    fig, ax = plt.subplots(dpi=100, figsize=(5, 3.5))
    for number, ds in series:
        color = _DETECTOR_COLORS[(number - 1) % len(_DETECTOR_COLORS)]
        angle = ds.waveplate_angle.values
        label = ds.attrs.get("channel", f"detector {number}")
        ax.errorbar(
            angle, ds.voltage.values, yerr=ds.voltage_error.values, fmt="o", color=color, capsize=2, label=label
        )
        m, b = ds.attrs["fit_m"], ds.attrs["fit_b"]
        fit_x = np.array([angle.min(), angle.max()])
        fit_label = f"m = {m:.3g} ± {ds.attrs['fit_m_err']:.2g} V/°, R² = {ds.attrs['fit_r_squared']:.3f}"
        ax.plot(fit_x, m * fit_x + b, "-", color=color, alpha=0.7, label=fit_label)
    ax.set_xlabel("waveplate angle (°)")
    ax.set_ylabel("voltage (V)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.legend(fontsize="small")

    title = _calibration_title(tree.attrs, "diode", title)
    if "setup" in tree.attrs:
        title += f" ({tree.attrs['setup']})"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()
