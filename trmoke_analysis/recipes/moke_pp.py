# ------------------
# Processing recipe for "standard" pump-probe MOKE scans
# ------------------

import numpy as np
import xarray as xr

if __package__:
    from . import helpers
else:
    import helpers


def process_moke_pp(data: xr.Dataset, spice: dict) -> xr.Dataset:
    # Extract parameters
    fluences = np.sort(np.unique(data.fluence.values))
    fields = np.sort(np.unique(data.field.values))
    fields_abs = np.sort(np.unique(np.abs(data.field.values)))  # without sign
    delays = np.sort(np.unique(data.delay.values))
    loops = np.int64(np.sort(np.unique(data.loop.values)))
    params = ["fluence", "field", "delay"]
    params_nr = np.array([len(fluences), len(fields_abs), len(delays)])
    sorted = np.argsort(params_nr)[::-1]
    scan_param = params[sorted[0]]
    param2 = params[sorted[1]]
    param1 = params[sorted[2]]

    # Step 1: Separate chopped/unchopped data based on chopper signal
    # Chopper > 2 means unchopped (pumped), < 2 means chopped (unpumped)
    is_unchopped = data.chopper_signal > 2

    # Create separate datasets for chopped and unchopped
    data_unchop = data.where(is_unchopped, drop=True)
    data_chop = data.where(~is_unchopped, drop=True)

    diode_signal = np.abs(data_unchop.diode_signal.values - data_chop.diode_signal.values)

    # Step 2: Calculate pumped - unpumped for each detector
    processed_data = {}
    for det_idx in range(3):
        processed_data[f"detector_{det_idx}"] = helpers.calculate_pump_unpump_difference(
            data_unchop, data_chop, det_idx
        )

    # Step 3: Organize into xarray with proper structure
    # Create empty arrays for each detector: [fluence, field, delay, loop]
    det0_array = np.full((len(fluences), len(fields), len(delays), len(loops)), np.nan)
    det1_array = np.full((len(fluences), len(fields), len(delays), len(loops)), np.nan)
    det2_array = np.full((len(fluences), len(fields), len(delays), len(loops)), np.nan)

    # Fill arrays directly from processed_data
    for det_idx, det_array in enumerate([det0_array, det1_array, det2_array]):
        for item in processed_data[f"detector_{det_idx}"]:
            # Find indices for each coordinate
            fluence_idx = np.where(fluences == item["fluence"])[0][0]
            field_idx = np.where(fields == item["field"])[0][0]
            delay_idx = np.where(delays == item["delay"])[0][0]
            loop_idx = np.where(loops == item["loop"])[0][0]
            # Store signal
            det_array[fluence_idx, field_idx, delay_idx, loop_idx] = item["signal"]

    # Stack detectors into single array: [detector, fluence, field, delay, loop]
    signal_loopwise = np.stack([det0_array, det1_array, det2_array], axis=0)

    # Step 4: Calculate field-up minus field-down
    # signal_loopwise is [detector, fluence, field, delay, loop]
    # For each unique |field| value, find the corresponding + and - indices
    unique_field_mags = np.unique(np.abs(fields))
    n_fields = len(unique_field_mags)

    # Create arrays directly in desired order: [detector, fluence, field, delay, loop]
    signal_sum = np.full((3, len(fluences), n_fields, len(delays), len(loops)), np.nan)
    signal_diff = np.full((3, len(fluences), n_fields, len(delays), len(loops)), np.nan)

    for i, field_mag in enumerate(unique_field_mags):
        # Find indices for +field and -field
        idx_up = np.where(np.abs(fields - field_mag) < 0.01)[0]
        idx_down = np.where(np.abs(fields + field_mag) < 0.01)[0]

        if len(idx_up) > 0 and len(idx_down) > 0:
            # signal_loopwise is [detector, fluence, field, delay, loop]
            up_data = signal_loopwise[:, :, idx_up[0], :, :]
            down_data = signal_loopwise[:, :, idx_down[0], :, :]

            # Calculate sum and difference
            signal_sum[:, :, i, :, :] = up_data + down_data
            signal_diff[:, :, i, :, :] = up_data - down_data

    # Step 5: Shift t0 (from spice)
    t0 = spice.get("t0", 0)

    delays -= t0  # Shift time axis by t0 (0 by default)

    # Step 6: Average over loops (last dimension) and apply sigmaclipping if specified in spice
    # signal arrays are [detector, fluence, field, delay, loop]
    sigma = spice.get("sigma", -1)
    signal_sum_avg = helpers.average_with_clip(signal_sum, sigma=sigma)  # Result: [detector, fluence, field, delay]
    signal_diff_avg = helpers.average_with_clip(signal_diff, sigma=sigma)  # Result: [detector, fluence, field, delay]

    # Step 7: Create final xarray Dataset
    processed_dataset = xr.Dataset(
        data_vars={
            # Loopwise data
            "signal_sum_loopwise": (["detector", "fluence", "field", "delay", "loop"], signal_sum),
            "signal_diff_loopwise": (["detector", "fluence", "field", "delay", "loop"], signal_diff),
            # Loop-averaged data
            "signal_sum": (["detector", "fluence", "field", "delay"], signal_sum_avg),
            "signal_diff": (["detector", "fluence", "field", "delay"], signal_diff_avg),
            # Diode data
            "diode_signal": (["frames"], diode_signal),
        },
        coords={
            "detector": data.detector.values,
            "field": unique_field_mags,
            "loop": loops,
            "delay": delays,
            "fluence": fluences,
            "frames": data.frames.values[::2],
        },
        attrs=data.attrs,  # Copy metadata from original dataset
    )

    # Add scan metadata
    processed_dataset.attrs["scan_param"] = scan_param
    processed_dataset.attrs["param1"] = param1
    processed_dataset.attrs["param2"] = param2

    # Step 8: Apply final adjustment: Remove offsets before t0 by subtracting mean of pre-t0 signal
    # also apply to loopwise signals
    pre_t0 = processed_dataset.sel(delay=processed_dataset.delay < 0)
    offset_sum = pre_t0.signal_sum.mean(dim="delay")
    offset_diff = pre_t0.signal_diff.mean(dim="delay")
    processed_dataset = processed_dataset.assign(
        {
            "signal_sum": processed_dataset["signal_sum"] - offset_sum,
            "signal_diff": processed_dataset["signal_diff"] - offset_diff,
            "signal_sum_loopwise": processed_dataset["signal_sum_loopwise"] - offset_sum,
            "signal_diff_loopwise": processed_dataset["signal_diff_loopwise"] - offset_diff,
        }
    )

    # Add spice parameters to dataset attributes for record-keeping
    for key, value in spice.items():
        processed_dataset.attrs[f"spice_{key}"] = value

    return processed_dataset
