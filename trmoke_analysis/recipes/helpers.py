# ------------------
# Generic helpers for recipes
# ------------------

import pandas as pd
import xarray as xr


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
