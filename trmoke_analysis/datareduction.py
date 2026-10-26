# %%
import numbers
import os
import re
import shutil
import time

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# Plotting import
from trmoke_analysis import plotting

# Recipes import
from trmoke_analysis.recipes.hysteresis import process_hysteresis
from trmoke_analysis.recipes.moke_pp import process_moke_pp

# -----------------------------
#  Data Evaluation
# -----------------------------

# Class based data analysis approach

# Proposal class:
# read raw data (e.g. on a server or from local files)
# allow reading data and display overview of scans and parameters
# then copy the files to the local folder and continue processing there (to avoid changing the original data)
# Data analysis using the spice concept from nx5d: t0 and sigmaclipping as possible "spices"
# Visualize the reuced data, and maybe tune the spice again until the results look good.
# %%
RECIPE_DICT = {
    "delay": process_moke_pp,
    "fluence": process_moke_pp,
    "field": process_moke_pp,
    "hysteresis": process_hysteresis,
    # add more recipes as needed
}

# Scan file kinds. Every scan file in a proposal folder is named "<scan_id><postfix>.nc" and is reached
# via prop.data.<prefix><scan_id> (e.g. prop.data.p0034). The prefix also labels the column of the
# proposal overview that marks whether the file exists. New kinds of files (e.g. new calibrations)
# only need to be added here.
SCAN_KINDS = {
    # kind: (filename postfix, accessor prefix)
    "raw": ("", "r"),
    "processed": ("_processed", "p"),
    "fluence_calibration": ("_fluence_calibration", "f"),
    "field_calibration": ("_field_calibration", "b"),
    "diode_calibration": ("_diode_calibration", "d"),
}
_KIND_BY_POSTFIX = {postfix: kind for kind, (postfix, _) in SCAN_KINDS.items()}
_KIND_BY_PREFIX = {prefix: kind for kind, (_, prefix) in SCAN_KINDS.items()}


def _format_scan_id(scan_id: str | int) -> str:
    """Return the scan ID as a zero-padded 4-digit string (e.g. 34 -> "0034"); strings are passed through."""
    if isinstance(scan_id, numbers.Integral):
        return f"{scan_id:04d}"
    return scan_id


def _scan_stem(scan_id: str | int, kind: str) -> str:
    """File stem (name without ".nc") of a scan file, e.g. ("0034", "processed") -> "0034_processed"."""
    return f"{_format_scan_id(scan_id)}{SCAN_KINDS[kind][0]}"


def _split_scan_stem(stem: str) -> tuple[str, str] | None:
    """Inverse of _scan_stem: "0034_processed" -> ("0034", "processed"); None if the stem is not a known kind."""
    match = re.fullmatch(r"(\d{4})(.*)", stem)
    if match is None or match.group(2) not in _KIND_BY_POSTFIX:
        return None
    return match.group(1), _KIND_BY_POSTFIX[match.group(2)]


class DataProposal:
    def __init__(
        self,
        proposal_id: str,
        proposal_path: str,
        spice_path: str = None,
        autofetch_spice: bool | str = True,
        overwrite: bool = False,
    ):
        """
        Initialize DataProposal with proposal ID and path. Check first if the complete data is available locally
        (where the python script is running), otherwise read from the server and copy to local folder.
        If overwrite is True, it will re-copy the data even if it already exists locally.

        Parameters
        ----------
        proposal_id : str
            Unique identifier for the proposal (e.g., "2026_04_FeAu_SL")
        proposal_path : str
            Path to the proposal folder on the server (e.g., "/server/data/")
        spice_path : str, optional
            Path to the spice folder on the server (if different from proposal_path + "/spice"), default is None
        autofetch_spice : bool or str, optional
            If True, automatically check for existing latest spice configuration and load it.
            Alternatively, provide a valid hash ID to load a specific spice file (default True)
        overwrite : bool, optional
            If True, re-copy data from server even if it exists locally (default False)
        """
        self.proposal_overview = None
        self._registry = None

        self.proposal_id = proposal_id
        self.proposal_path = proposal_path
        self.local_path = os.path.join(os.getcwd(), self.proposal_id)
        self.server_path = os.path.join(self.proposal_path, self.proposal_id)

        self._sync_from_server(overwrite=overwrite)

        # ScanAccessor is instantiated here and assigned to self.data
        self.data = ScanAccessor(self)

        # Build spice registry
        if spice_path is None:
            self.spice_path = self.server_path
        else:
            self.spice_path = os.path.join(spice_path, self.proposal_id)
        self.spice = Spice(self, server_path=self.spice_path)

        # Populate local spice cache from the remote repository on first setup.
        self._sync_spice_from_server()

        # Check for existing spice
        spice_folder_local = os.path.join(self.local_path, "spice")
        if os.path.exists(spice_folder_local):
            existing_files = os.listdir(spice_folder_local)
            if existing_files:  # get timestamps from saved attribute in the nc file
                timestamps = []
                for f in existing_files:
                    try:
                        data = xr.load_dataset(os.path.join(spice_folder_local, f))
                        timestamp = data.attrs.get("creation_timestamp", None)
                        if timestamp is not None:
                            timestamps.append((f, timestamp))
                    except Exception as e:
                        print(f"Warning: Could not read spice file {f}: {e}")
                if timestamps:
                    latest_file, latest_timestamp = max(timestamps, key=lambda x: x[1])
                    latest_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest_timestamp))
                    if autofetch_spice is True:
                        print(f"Existing spice configuration found: {latest_file} (created on {latest_date}).")
                        self.spice.recall_spice_data(latest_file.split(".")[0])  # Load the latest spice configuration
                    elif isinstance(autofetch_spice, str):
                        try:
                            self.spice.recall_spice_data(autofetch_spice)
                        except FileNotFoundError:
                            print(
                                f"Spice configuration with hash '{autofetch_spice}' not found. \
                                Starting with a new spice configuration."
                            )
                    else:
                        print("Starting with a new spice configuration.")
                else:
                    print("No valid spice files found. Starting with a new spice configuration.")

        # Build recipe dictionary: how should scans be processed
        self.recipe_dict = RECIPE_DICT

    # string and html based class representation for jupyter notebooks

    def __repr__(self):
        return self.proposal_overview.__repr__()

    def _repr_html_(self):
        header = f"<h4>DataProposal: {self.proposal_id}</h4>"
        return header + self.proposal_overview.to_html()

    def _sync_from_server(self, overwrite: bool = False) -> None:
        """
        Sync data from server to local cache.
        Only copies files that are missing or have changed (by size).
        """
        try:
            if not os.path.exists(self.server_path):
                raise FileNotFoundError(f"Server path not found: {self.server_path}")

            os.makedirs(self.local_path, exist_ok=True)

            server_files = {
                f: os.path.getsize(os.path.join(self.server_path, f))
                for f in os.listdir(self.server_path)
                if os.path.isfile(os.path.join(self.server_path, f))
            }

            if overwrite:
                files_to_copy = set(server_files)
            else:
                local_files = {
                    f: os.path.getsize(os.path.join(self.local_path, f))
                    for f in os.listdir(self.local_path)
                    if os.path.isfile(os.path.join(self.local_path, f))
                }
                files_to_copy = {
                    f for f, size in server_files.items() if f not in local_files or local_files[f] != size
                }

            if not files_to_copy:
                print(f"Local cache is up to date ({len(server_files)} files).")
            else:
                print(f"Syncing {len(files_to_copy)}/{len(server_files)} files from server...")
                for f in tqdm(sorted(files_to_copy)):
                    shutil.copy2(os.path.join(self.server_path, f), os.path.join(self.local_path, f))
                print("Sync complete.")

        except FileNotFoundError as e:
            print(e)
            if not os.path.exists(self.local_path):
                raise FileNotFoundError(
                    f"Server unreachable and no local cache found for '{self.proposal_id}'."
                ) from None
            print("Loading from local cache.")

        finally:
            if os.path.exists(self.local_path):
                self.proposal_overview = self._build_proposal_overview()
                self._registry = {f[:-3] for f in os.listdir(self.local_path) if f.endswith(".nc")}

    def _sync_spice_from_server(self) -> None:
        """
        Sync spice files from the remote repository to the local cache.

        This mirrors the proposal file sync behavior and backfills any new or changed
        spice files even when the local spice folder already contains entries.
        """
        spice_folder_local = os.path.join(self.local_path, "spice")
        spice_folder_server = os.path.join(self.spice._server_path, "spice")

        if not os.path.exists(spice_folder_server):
            print(f"Remote spice folder not found: {spice_folder_server}.")
            return

        os.makedirs(spice_folder_local, exist_ok=True)

        server_files = {
            f: os.path.getsize(os.path.join(spice_folder_server, f))
            for f in os.listdir(spice_folder_server)
            if os.path.isfile(os.path.join(spice_folder_server, f))
        }
        local_files = {
            f: os.path.getsize(os.path.join(spice_folder_local, f))
            for f in os.listdir(spice_folder_local)
            if os.path.isfile(os.path.join(spice_folder_local, f))
        }

        files_to_copy = {f for f, size in server_files.items() if f not in local_files or local_files[f] != size}

        if not files_to_copy:
            print(f"Local spice cache is up to date ({len(server_files)} files).")
            return

        print(f"Syncing {len(files_to_copy)}/{len(server_files)} spice files from remote repository...")
        for filename in tqdm(sorted(files_to_copy)):
            shutil.copy2(os.path.join(spice_folder_server, filename), os.path.join(spice_folder_local, filename))
        print("Spice sync complete.")

    def _build_proposal_overview(self):
        """
        Build a pandas DataFrame overview of all scans in the proposal.

        Reads all NetCDF files in the local proposal folder, extracts metadata and parameters,
        and compiles them into a structured DataFrame for easy reference.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            scan_id, one checkmark column per non-raw kind in SCAN_KINDS (e.g. "p" = processed),
            sample_name, scan_parameter, number of frames, delay, fluence and field values
        """
        all_nc = {filename[:-3] for filename in os.listdir(self.local_path) if filename.endswith(".nc")}
        raw_ids = sorted(s for s in all_nc if re.match(r"^\d{4}$", s))

        overview_data = []
        for scan_id in raw_ids:
            filepath = os.path.join(self.local_path, f"{scan_id}.nc")
            with xr.open_dataset(filepath) as data:
                sample_name = data.attrs.get("sample_name", "Unknown")
                scan_parameter = data.attrs.get("scan_parameter", "Unknown")
                frames = data.sizes["frames"]
                try:
                    loops = int(data.attrs["nr_loops"])
                except Exception:
                    loops = "Unknown"

                notes_list = []
                for param in ["delay", "fluence", "field"]:
                    try:
                        unique_values = np.unique(data.attrs[f"{param}_values"])
                        if len(unique_values) > 1:
                            notes_list.append(
                                f"{unique_values[0]:.2f} ... {unique_values[-1]:.2f} "
                                f"{data.attrs[f'{param}_unit']} ({len(unique_values)})"
                            )
                        else:
                            notes_list.append(f"{unique_values[0]:.2f} {data.attrs[f'{param}_unit']}")
                    except Exception:
                        notes_list.append("Unknown")

            row = {"scan_id": int(scan_id)}
            for kind, (_, prefix) in SCAN_KINDS.items():
                if kind != "raw":
                    row[prefix] = "✓" if _scan_stem(scan_id, kind) in all_nc else ""
            row.update(
                {
                    "sample": sample_name,
                    "scan_parameter": scan_parameter,
                    "frames": frames,
                    "loops": loops,
                    "delay_values": notes_list[0] if len(notes_list) > 0 else "Unknown",
                    "fluence_values": notes_list[1] if len(notes_list) > 1 else "Unknown",
                    "field_values": notes_list[2] if len(notes_list) > 2 else "Unknown",
                }
            )
            overview_data.append(row)

        overview_df = pd.DataFrame(overview_data).set_index("scan_id")
        return overview_df

    def _load(self, scan_id: str | int, kind: str = "raw"):
        """
        Load data for a specific scan ID and kind (raw or reduced).

        Parameters
        ----------
        scan_id : str or int
            Identifier for the scan to load (e.g., "0034" or 34)
        kind : str, optional
            Type of data to load, one of the keys of SCAN_KINDS (default is "raw")

        Returns
        -------
        xr.Dataset
            The loaded dataset for the specified scan and kind
        """
        if kind not in SCAN_KINDS:
            raise ValueError(f"Invalid kind: {kind}. Must be one of {', '.join(repr(k) for k in SCAN_KINDS)}.")

        filename = f"{_scan_stem(scan_id, kind)}.nc"
        file_path = os.path.join(self.local_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        return xr.load_dataset(file_path)

    def process(self, scan_id: str | int | list, overwrite: bool = False) -> None:
        """
        Process raw data for the specified scan(s) and save the processed dataset.

        Parameters
        ----------
        scan_id : str, int, or list
            Identifier(s) for the scan(s) to process (e.g., "0034", 34, or ["0034", "0035"])
        overwrite : bool, optional
            If True, overwrite existing processed files (default False)
        """
        self.spice.save_spice_data()  # Save current spice configuration before processing

        if isinstance(scan_id, (str, numbers.Integral)):
            self._process_single(scan_id, overwrite=overwrite)

        elif isinstance(scan_id, (list, np.ndarray)):
            # batch processing for a list of IDs
            # use try-except to skip processing if errors occur during singular scans
            for sid in scan_id:
                if isinstance(sid, (str, numbers.Integral)):
                    try:
                        self._process_single(sid, overwrite=overwrite)
                    except Exception as e:
                        print(f"Error processing scan {sid}: {e}. Skipping this scan.")
                else:
                    raise ValueError(f"List entry '{sid}' is not a string or integer.")
        else:
            raise ValueError("scan_id must be a string, integer, or list of strings/integers.")

    def _process_single(self, scan_id: str | int, overwrite: bool = False) -> None:
        """
        Process a single scan and save the processed dataset.

        Parameters
        ----------
        scan_id : str or int
            Identifier for the scan to process (e.g., "0034" or 34)
        overwrite : bool, optional
            If True, overwrite existing processed file (default False)
        """
        # Check if processed file already exists
        scan_id_str = _format_scan_id(scan_id)
        processed_filename = f"{_scan_stem(scan_id_str, 'processed')}.nc"
        processed_file_path = os.path.join(self.local_path, processed_filename)
        if os.path.exists(processed_file_path) and not overwrite:
            print(f"Processed file already exists for scan {scan_id_str}. Use overwrite=True to re-process.")
            return
        else:
            # Load data and load spice and determine processing recipe based on scan_parameter
            print(f"Processing scan {scan_id_str}...")
            data = self._load(scan_id, kind="raw")
            spice = self.spice.get(int(scan_id_str))
            # determine recipe to use for processing (based on scan_parameter or other metadata)
            scan_param = data.attrs.get("scan_parameter", "Unknown")
            if scan_param == "Unknown":  # scan_param is assumed to have most unique values
                fluences = np.sort(np.unique(data.fluence.values))
                fields_abs = np.sort(np.unique(np.abs(data.field.values)))  # without sign
                delays = np.sort(np.unique(data.delay.values))
                loops = np.int64(np.sort(np.unique(data.loop.values)))
                print("No scan_parameter attribute found. Determining scan parameter based on number unique values...")
                print(f"Unique fluences: {len(fluences)}")
                print(f"Unique fields: {len(fields_abs)}")
                print(f"Unique delays: {len(delays)}")
                print(f"Unique loops: {len(loops)}")
                params = ["fluence", "field", "delay"]
                params_nr = np.array([len(fluences), len(fields_abs), len(delays)])
                sorted = np.argsort(params_nr)[::-1]
                scan_param = params[sorted[0]]
            print(f"This is a {scan_param} scan.")
            recipe = self.recipe_dict.get(scan_param, None)
            if recipe is None:
                print(f"""No processing recipe found for scan type '{scan_param}'.
                      Skipping processing for scan {scan_id_str}.""")
                return
            processed_data = recipe(data, spice)
            # Save processed data to new NetCDF file
            processed_data.to_netcdf(processed_file_path)
            print(f"Processed data saved to {processed_file_path}.")

    def plot_loopwise(self, scan_id: str | int, detector_idx: int = 0):
        try:
            dataset = self._load(scan_id, kind="processed")
        except FileNotFoundError:
            print(f"Processed data not found for scan {scan_id}. Maybe the data is not processed yet.")
            return
        scan_type = dataset.attrs.get("scan_parameter", "Unknown")
        if scan_type in ["delay", "fluence", "field"]:
            plotting.plot_loopwise_moke_pp(dataset, detector_idx=detector_idx)
        elif scan_type == "hysteresis":
            plotting.plot_loopwise_hysteresis(dataset, detector_idx=detector_idx)
        else:
            print(f"Quick plotting not implemented for scan type '{scan_type}'.")

    def plot_overview(self, scan_id: str | int, detector_idx: int = 0):
        try:
            dataset = self._load(scan_id, kind="processed")
        except FileNotFoundError:
            print(f"Processed data not found for scan {scan_id}. Maybe the data is not processed yet.")
            return
        scan_type = dataset.attrs.get("scan_parameter", "Unknown")
        if scan_type in ["delay", "fluence", "field"]:
            plotting.plot_overview_moke_pp(dataset, detector_idx=detector_idx)
        elif scan_type == "hysteresis":
            plotting.plot_overview_hysteresis(dataset, detector_idx=detector_idx)
        else:
            print(f"Overview plotting not implemented for scan type '{scan_type}'.")


class ScanAccessor:
    """
    Helper class to access scans as attributes of the proposal, e.g. prop.data.r0001 for raw data
    and prop.data.p0001 for processed data. The one-letter prefixes are defined in SCAN_KINDS.
    """

    def __init__(self, proposal):
        self._proposal = proposal

    @staticmethod
    def _usage() -> str:
        return ", ".join(f"{prefix}0001={kind}" for kind, (_, prefix) in SCAN_KINDS.items())

    def __getattr__(self, name: str):
        """
        Access scans as attributes: a one-letter kind prefix from SCAN_KINDS followed by the scan number,
        e.g. r0001 (raw), p0001 (processed), f0001 (fluence calibration), b0001 (field calibration),
        d0001 (diode calibration).
        """
        kind = _KIND_BY_PREFIX.get(name[:1])
        scan_id = name[1:]
        if kind is None or not scan_id.isdigit():
            raise AttributeError(f"Invalid accessor '{name}'. Use <prefix><scan number>: {self._usage()}.")

        scan_id = f"{int(scan_id):04d}"
        if _scan_stem(scan_id, kind) not in self._proposal._registry:
            raise AttributeError(f"No {kind} data found for scan {scan_id} in proposal.")

        return self._proposal._load(scan_id, kind=kind)

    def __dir__(self):
        """
        List available scan accessors based on the proposal registry.
        """
        names = []
        for stem in self._proposal._registry:
            split = _split_scan_stem(stem)
            if split is not None:
                scan_id, kind = split
                names.append(f"{SCAN_KINDS[kind][1]}{scan_id}")
        return sorted(names)

    def __repr__(self):
        nr_files = len(self._proposal._registry)
        return f"ScanAccessor: {nr_files} scan files found. Use <prefix><scan number>: {self._usage()}."


class Spice:
    """
    Class to represent a nx5d-like "spice" for data reduction, e.g. t0 or sigmaclipping.
    """

    def __init__(self, proposal: DataProposal, server_path: str = None):
        scan_numbers = sorted(int(s) for s in proposal._registry if re.match(r"^\d{4}$", s))
        if not scan_numbers:
            raise ValueError("Registry contains no bare scan numbers (e.g. '0001').")

        self._scan_numbers = scan_numbers
        self._table = pd.DataFrame(index=scan_numbers, dtype=float)
        self._table.index.name = "scan_id"

        self._local_path = proposal.local_path
        if server_path is None:
            self._server_path = proposal.server_path
        else:
            self._server_path = server_path
        self._seeded = False

    def seed(self, params: dict) -> None:
        """
        Declare parameters and set their default starting values.
        Sets values at the first scan number so forward-fill works
        correctly from the beginning.

        Parameters
        ----------
        params : dict
            Parameter names and default values, e.g.
            {"temperature": 300.0, "field": 0.0, "fluence": 1.5}
        """
        # reset table columns so re-running in Jupyter works cleanly
        self._table = self._table[[]]
        first_scan = self._table.index[0]
        for param, value in params.items():
            self._table[param] = np.nan
            self._table.loc[first_scan, param] = float(value)
        self._seeded = True

    def update(self, scan_id: int, params: dict) -> None:
        """
        Set parameter values at a specific scan number.

        Only keys present in params are updated; all others remain
        unchanged and inherit the last explicitly set value via forward-fill.

        Parameters
        ----------
        scan_id : int
            Scan number where the change occurred (e.g. 50)
        params : dict
            Parameter names and their new values, e.g. {"temperature": 350.0}
        """
        if not self._seeded:
            raise RuntimeError("Spice not seeded. Call seed() first.")
        if scan_id not in self._table.index:
            raise IndexError(f"Scan {scan_id:04d} does not exist in this proposal.")
        unknown = set(params) - set(self._table.columns)
        if unknown:
            raise KeyError(f"Unknown parameter(s): {unknown}. Valid parameters are: {self.parameters}.")
        for param, value in params.items():
            self._table.loc[scan_id, param] = float(value)

    def get(self, scan_id: int) -> dict:
        """
        Get effective parameters at a specific scan (forward-filled).

        Parameters
        ----------
        scan_id : int
            Scan number to query.
        """
        if scan_id not in self._table.index:
            raise IndexError(f"Scan {scan_id:04d} does not exist in this proposal.")
        return self._table.ffill().loc[scan_id].to_dict()

    def save_spice_data(self) -> None:
        """
        Update the spice file in the local and remote proposal folder after changes have been made to the spice table.
        When called first on a new proposal, it will create a spice folder which contains the reduced spice table
        as a NetCDF file with a hashed filename to ensure only unique spice configurations are saved. When called on a
        proposal with existing spice data, it checks first that the current spice configuration is not already saved
        (by comparing the hash) and only saves a new file if the configuration has changed.
        """
        if not self._seeded:
            raise RuntimeError("Spice not seeded. Call seed() first.")
        spice_folder_local = os.path.join(self._local_path, "spice")
        spice_folder_server = os.path.join(self._server_path, "spice")
        os.makedirs(spice_folder_local, exist_ok=True)
        existing_files_local = os.listdir(spice_folder_local)

        server_available = False
        existing_files_server = []
        try:
            os.makedirs(spice_folder_server, exist_ok=True)
            existing_files_server = os.listdir(spice_folder_server)
            server_available = True
        except Exception:
            print(f"Warning: Could not reach server spice folder at {spice_folder_server}. Saving locally only.")

        current_hash = self._generate_hash()
        filename = f"{current_hash}.nc"

        if any(current_hash in fl for fl in existing_files_local):
            print("Current spice configuration already saved locally.")
            if server_available and not any(current_hash in fs for fs in existing_files_server):
                print("Remote spice repository outdated. Uploading latest spice...")
                shutil.copy2(os.path.join(spice_folder_local, filename), os.path.join(spice_folder_server, filename))
            return

        # Store only the explicit anchor entries; recall_spice_data rebuilds the full
        # scan index of the proposal and forward-fill restores the effective values
        data = xr.Dataset.from_dataframe(self._canonicalize_table(self.changelog))
        creation_time = time.time()
        data.attrs["creation_timestamp"] = creation_time
        data.attrs["creation_date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(creation_time))

        data.to_netcdf(os.path.join(spice_folder_local, filename))
        if server_available:
            shutil.copy2(os.path.join(spice_folder_local, filename), os.path.join(spice_folder_server, filename))
            print(f"Spice configuration saved as {filename} in local and server spice folders.")
        else:
            print(f"Spice configuration saved as {filename} locally only (server unreachable).")

    def recall_spice_data(self, hash_str: str) -> None:
        """
        Load a specific spice configuration from the local spice folder based on the provided hash string.
        The spice table is updated with the loaded configuration, allowing for easy recall of previous settings.

        Parameters
        ----------
        hash_str : str
            Hash string corresponding to the desired spice configuration (e.g., "abc123def456").
        """
        spice_folder_local = os.path.join(self._local_path, "spice")
        filename = f"{hash_str}.nc"
        file_path = os.path.join(spice_folder_local, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Spice configuration file not found: {file_path}")
        data = xr.load_dataset(file_path)
        self._table = data.to_dataframe()
        # Add empty rows for scans measured after this spice configuration was saved,
        # so forward-fill propagates the last set parameters onto the new scans
        self._table = self._table.reindex(self._table.index.union(self._scan_numbers))
        self._table = self._canonicalize_table(self._table)
        self._seeded = True
        print(f"Spice configuration '{hash_str}' loaded successfully.")

    def _generate_hash(self) -> str:
        """
        Generate a hash string based on the current state of the spice table.
        This can be used for caching or tracking changes.

        Returns
        -------
        str
            A hash string representing the current spice configuration.
        """
        import hashlib

        # Hash only the explicit anchor entries (changelog): the effective values of all
        # scans follow from these via forward-fill, so adding new scans to the proposal
        # does not change the configuration and must not create a new spice file.
        # float_format rounds to 4 decimals so that float noise (e.g. 0.1 + 0.2) does not
        # change the hash; lineterminator is pinned as to_csv defaults to the OS line ending.
        table_str = self._canonicalize_table(self.changelog).to_csv(
            float_format="%.4f", index_label="scan_id", lineterminator="\n"
        )
        return hashlib.sha256(table_str.encode()).hexdigest()

    def _canonicalize_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """Return a normalized table representation for stable hashing and loading."""
        canonical_table = table.sort_index().copy()
        canonical_table = canonical_table.reindex(sorted(canonical_table.columns), axis=1)
        canonical_table = canonical_table + 0.0  # normalize -0.0 to 0.0 (would format differently in the hash)
        canonical_table.index.name = "scan_id"
        return canonical_table

    @property
    def parameters(self) -> list:
        """List of declared parameter names."""
        return list(self._table.columns)

    @property
    def table(self) -> pd.DataFrame:
        """Forward-filled view of the full spice table (effective values)."""
        return self._table.ffill()

    @property
    def changelog(self) -> pd.DataFrame:
        """Raw table showing only rows with at least one explicit entry."""
        mask = self._table.notna().any(axis=1)
        return self._table[mask]

    # String and HTML representation for Jupyter notebooks
    def __repr__(self) -> str:
        if not self._seeded:
            return "Spice | not seeded. Call seed(['param1', 'param2', ...]) to initialize."
        n_changes = self._table.notna().any(axis=1).sum()
        header = (
            f"Spice | {len(self.parameters)} parameters | "
            f"{len(self._table)} scans | {n_changes} explicit entries\n"
            f"Parameters: {self.parameters}\n"
        )
        return header + self.changelog.to_string()

    def _repr_html_(self) -> str:
        if not self._seeded:
            return "<h4>Spice | not seeded</h4><p>Call <code>seed(['param1', 'param2', ...])</code> to initialize.</p>"
        n_changes = self._table.notna().any(axis=1).sum()
        header = (
            f"<h4>Spice &nbsp;|&nbsp; {len(self.parameters)} parameters &nbsp;|&nbsp; "
            f"{len(self._table)} scans &nbsp;|&nbsp; {n_changes} explicit entries</h4>"
        )
        return header + self.changelog.to_html()
