# %%
# ------------------
# Processing recipe for static and time-resolved hysteresis scans
# ------------------

import numpy as np
import xarray as xr

if __package__:
    from . import helpers
else:
    import helpers


def process_hysteresis(data: xr.Dataset, spice: dict) -> xr.Dataset:
    # Extract parameters
    fluences = np.sort(np.unique(data.fluence.values))
    delays = np.sort(np.unique(data.delay.values))
    params = ["fluence", "delay"]
    params_nr = np.array([len(fluences), len(delays)])
    sorted = np.argsort(params_nr)[::-1]
    scan_param = "field"  # for hysteresis, field is always the scan parameter
    param2 = params[sorted[0]]
    param1 = params[sorted[1]]

    # Step 1: Separate chopped/unchopped data based on chopper signal and identify sweep directions
    # Chopper > 2 means unchopped (pumped), < 2 means chopped (unpumped)
    is_unchopped = data.chopper_signal > 2

    # Create separate datasets for chopped and unchopped
    data_unchop = data.where(is_unchopped, drop=True)
    data_chop = data.where(~is_unchopped, drop=True)

    # Create sweep direction arrays based on field values to identify up/down sweeps
    sweep_dir_unchop = data_unchop.field.differentiate("frames")
    sweep_dir_chop = data_chop.field.differentiate("frames")

    data_unchop_up = data_unchop.where(sweep_dir_unchop > 0, drop=True)
    data_unchop_down = data_unchop.where(sweep_dir_unchop < 0, drop=True)
    data_chop_up = data_chop.where(sweep_dir_chop > 0, drop=True)
    data_chop_down = data_chop.where(sweep_dir_chop < 0, drop=True)

    diode_signal = np.abs(data_unchop.diode_signal.values - data_chop.diode_signal.values)

    # Step 2: Extract and reorganize the four branches separately.
    # xarray returns new objects, so each branch must be reassigned explicitly.

    data_unchop_up = reshape_branch(data_unchop_up)
    data_unchop_down = reshape_branch(data_unchop_down)
    data_chop_up = reshape_branch(data_chop_up)
    data_chop_down = reshape_branch(data_chop_down)

    # Step 3: Apply drift correction if specified in spice
    if spice.get("drift_correct", False):
        data_unchop_up, data_unchop_down = drift_correction(data_unchop_up, data_unchop_down)
        data_chop_up, data_chop_down = drift_correction(data_chop_up, data_chop_down)

    # Step 4: Apply the averaging and sigmaclipping as specified in spice
    sigma = spice.get("sigma", -1)
    data_unchop_up_avg = helpers.average_with_clip(data_unchop_up, sigma=sigma)
    data_unchop_down_avg = helpers.average_with_clip(data_unchop_down, sigma=sigma)
    data_chop_up_avg = helpers.average_with_clip(data_chop_up, sigma=sigma)
    data_chop_down_avg = helpers.average_with_clip(data_chop_down, sigma=sigma)

    # Step 5: Shift the hysteresis up/down and left/right to center around zero, as specified from spice

    shift_y = spice.get("shift_hysteresis", False)
    shift_x = spice.get("shift_hysteresis_x", False)

    data_unchop_up_avg, data_unchop_down_avg = shift_hysteresis(
        data_unchop_up_avg, data_unchop_down_avg, shift_y=shift_y, shift_x=shift_x
    )
    data_chop_up_avg, data_chop_down_avg = shift_hysteresis(
        data_chop_up_avg, data_chop_down_avg, shift_y=shift_y, shift_x=shift_x
    )

    # Step 6: Calculate some hysteresis properties
    remanence_unchop = (np.abs(data_unchop_up_avg.interp(field=0)) + np.abs(data_unchop_down_avg.interp(field=0))) / 2
    remanence_chop = (np.abs(data_chop_up_avg.interp(field=0)) + np.abs(data_chop_down_avg.interp(field=0))) / 2
    saturation_unchop = (
        np.abs(data_unchop_up_avg.isel(field=-1))
        + np.abs(data_unchop_down_avg.isel(field=-1))
        + np.abs(data_unchop_up_avg.isel(field=0))
        + np.abs(data_unchop_down_avg.isel(field=0))
    ) / 4
    saturation_chop = (
        np.abs(data_chop_up_avg.isel(field=-1))
        + np.abs(data_chop_down_avg.isel(field=-1))
        + np.abs(data_chop_up_avg.isel(field=0))
        + np.abs(data_chop_down_avg.isel(field=0))
    ) / 4

    # Step 7: Create final xarray Dataset
    t0 = spice.get("t0", 0)

    processed_dataset = xr.Dataset(
        data_vars={
            "signal_up_pumped": data_unchop_up_avg.detector_data,
            "signal_down_pumped": data_unchop_down_avg.detector_data,
            "signal_up_unpumped": data_chop_up_avg.detector_data,
            "signal_down_unpumped": data_chop_down_avg.detector_data,
            "signal_up_pumped_loopwise": data_unchop_up.detector_data,
            "signal_down_pumped_loopwise": data_unchop_down.detector_data,
            "signal_up_unpumped_loopwise": data_chop_up.detector_data,
            "signal_down_unpumped_loopwise": data_chop_down.detector_data,
            "remanence_pumped": remanence_unchop.detector_data,
            "remanence_unpumped": remanence_chop.detector_data,
            "saturation_pumped": saturation_unchop.detector_data,
            "saturation_unpumped": saturation_chop.detector_data,
            "diode_signal": (["frames"], diode_signal),
        },
        coords={
            "detector": data_chop_up.detector.values,
            "field": data_chop_up.field.values,
            "loop": data_chop_up.loop.values,
            "delay": data_chop_up.delay.values - t0,  # Shift time axis by t0 (0 by default)
            "fluence": data_chop_up.fluence.values,
            "frames": data.frames.values[::2],
        },
        attrs=data.attrs,
    )

    # Add scan metadata
    processed_dataset.attrs["scan_param"] = scan_param
    processed_dataset.attrs["param1"] = param1
    processed_dataset.attrs["param2"] = param2

    # Add spice parameters to dataset attributes for record-keeping
    for key, value in spice.items():
        processed_dataset.attrs[f"spice_{key}"] = value

    return processed_dataset


# ------------------
# Hysteresis specific helper functions
# ------------------


def reshape_branch(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.drop_vars(["chopper_signal", "diode_signal"])
    ds = ds.set_index(frames=["fluence", "field", "delay", "loop"]).unstack("frames")
    ds["loop"] = ds.loop.astype(np.int64)
    return ds


def drift_correction(ds_up: xr.Dataset, ds_down: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    # Correct linear drift by comparing hysteresis endpoints between adjacent loops.
    # 3a: Remove incomplete loops where either branch contains missing field values.
    valid_up = ds_up["detector_data"].notnull().all(dim="field")
    valid_down = ds_down["detector_data"].notnull().all(dim="field")
    valid_mask = valid_up & valid_down
    ds_up = ds_up.where(valid_mask, drop=True).copy()
    ds_down = ds_down.where(valid_mask, drop=True).copy()

    # 3b: Calculate drift in each loop by comparing the start and end point of the hysteresis loop.
    drifts = []
    for loop in range(ds_up.sizes["loop"]):
        start_up = ds_up["detector_data"].isel(loop=loop, field=0)
        end_down = ds_down["detector_data"].isel(loop=loop, field=0)
        drift = start_up - end_down
        drifts.append(drift)

    drifts = xr.concat(drifts, dim=ds_up.coords["loop"])
    avg_drift = drifts.mean(dim="loop")
    step_array = np.linspace(0, 0.5, ds_up.sizes["field"])

    loop_index = xr.DataArray(np.arange(ds_up.sizes["loop"]), dims=("loop",), coords={"loop": ds_up["loop"]})
    step_index = xr.DataArray(step_array, dims=("field",), coords={"field": ds_up["field"]})

    drift_up = (loop_index + step_index) * avg_drift
    drift_down = (loop_index + step_index + 0.5) * avg_drift

    ds_up["detector_data"] = ds_up["detector_data"] - drift_up
    ds_down["detector_data"] = ds_down["detector_data"] - drift_down

    return ds_up, ds_down


def shift_hysteresis(
    ds_up: xr.Dataset, ds_down: xr.Dataset, shift_y: bool = False, shift_x: bool = False
) -> tuple[xr.Dataset, xr.Dataset]:

    if shift_y:
        if shift_x:
            # first shift along y with minmax method first (not super precise, but needed for shifting along x)
            up_min = ds_up["detector_data"].min(dim="field")
            up_max = ds_up["detector_data"].max(dim="field")
            down_min = ds_down["detector_data"].min(dim="field")
            down_max = ds_down["detector_data"].max(dim="field")
            shift = (up_max + up_min + down_max + down_min) / 4
            ds_up["detector_data"] -= shift
            ds_down["detector_data"] -= shift
            # shift along x by aligning coercive fields
            field_axis = ds_up["field"].values
            for det in range(ds_up["detector_data"].sizes["detector"]):
                up_slice = ds_up["detector_data"].isel(detector=det).values
                down_slice = ds_down["detector_data"].isel(detector=det).values

                coercive_up = np.abs(np.interp(0, up_slice, field_axis))
                coercive_down = np.abs(np.interp(0, down_slice, field_axis))
                x_shift = (coercive_up + coercive_down) / 2
                shifted_field = field_axis - x_shift

                up_new = np.interp(field_axis, shifted_field, up_slice)
                down_new = np.interp(field_axis, shifted_field, down_slice)

                ds_up["detector_data"][dict(detector=det)] = up_new
                ds_down["detector_data"][dict(detector=det)] = down_new
        else:
            # only shift along y. Use the more precise overall mean method for vertical alignment
            shift = (ds_up["detector_data"].mean(dim="field") + ds_down["detector_data"].mean(dim="field")) / 2
            ds_up["detector_data"] -= shift
            ds_down["detector_data"] -= shift
    elif shift_x:
        print(
            "Warning: Shifting along x without shifting along y is not supported! Please also enable shifting along y."
        )

    return ds_up, ds_down
