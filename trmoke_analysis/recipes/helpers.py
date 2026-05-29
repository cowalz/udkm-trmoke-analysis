# ------------------
# Generic helpers for recipes
# ------------------

import numpy as np
import pandas as pd
import xarray as xr
from astropy.stats import sigma_clip  # advanced sigmaclipping with axis support


def calculate_pump_unpump_difference(
    data_unchop: xr.Dataset, data_chop: xr.Dataset, detector_idx: int = 0
) -> list[dict]:
    """
    Calculate pumped - unpumped signal.

    For each unique combination of all parameters,
    find the corresponding chopped and unchopped measurements.
    """
    group_vars = ["fluence", "field", "delay", "loop"]
    pumped = pd.DataFrame(
        {
            "fluence": data_unchop.fluence.values,
            "field": data_unchop.field.values,
            "delay": data_unchop.delay.values,
            "loop": data_unchop.loop.values,
            "signal": data_unchop.detector_data.isel(detector=detector_idx).values,
        }
    )
    unpumped = pd.DataFrame(
        {
            "fluence": data_chop.fluence.values,
            "field": data_chop.field.values,
            "delay": data_chop.delay.values,
            "loop": data_chop.loop.values,
            "signal": data_chop.detector_data.isel(detector=detector_idx).values,
        }
    )

    # Match previous behavior: if duplicate keys exist, use the first frame per key.
    pumped = pumped.drop_duplicates(subset=group_vars, keep="first")
    unpumped = unpumped.drop_duplicates(subset=group_vars, keep="first")

    merged = pumped.merge(unpumped, on=group_vars, how="inner", suffixes=("_pump", "_unpump"))
    merged["signal"] = merged["signal_pump"] - merged["signal_unpump"]

    return merged.sort_values(group_vars)[group_vars + ["signal"]].to_dict("records")


def average_with_clip(data: np.ndarray | xr.Dataset, sigma: float = -1, axis: int = -1) -> np.ndarray:
    """
    Average data with optional sigma clipping.

    If sigma > 0, apply sigma clipping and then average along the specified axis.
    If sigma <= 0, simply average without clipping along the specified axis.
    """
    return_xarray = False
    if isinstance(data, xr.Dataset):
        array = data.detector_data.values  # Convert xarray Dataset to numpy array for processing
        axis = data.detector_data.get_axis_num("loop")
        return_xarray = True
    else:
        array = data
    if sigma > 0:
        array = sigma_clip(array, sigma=sigma, axis=axis, masked=False)
    if return_xarray:
        data["detector_data"] = (("detector", "fluence", "field", "delay", "loop"), array)
        return data.mean(dim="loop", skipna=True, keep_attrs=True)  # Return xarray Dataset with averaged data
    else:
        return np.nanmean(array, axis=axis)  # Return numpy array with averaged data
