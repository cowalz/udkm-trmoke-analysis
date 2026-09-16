import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_ID = "2026_04_PtCo_SL"
RAW_DATA_PATH = REPO_ROOT / "Examples" / "Remote" / "Data"
SPICE_PATH = REPO_ROOT / "Examples" / "Remote" / "Spice"


def main() -> None:
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    from trmoke_analysis.datareduction import DataProposal

    # Notebook flow without the plotting cells.
    proposal = DataProposal(
        PROPOSAL_ID,
        str(RAW_DATA_PATH),
        spice_path=str(SPICE_PATH),
        overwrite=True,
    )

    print(proposal.data.r0016)
    print(proposal.data.r0017)
    print(proposal.data.r0018)

    proposal.spice.seed({"t0": 0})
    proposal.process([16, 17, 18], overwrite=True)

    print(proposal.data.p0016)
    print(proposal.data.p0017)
    print(proposal.data.p0018)

    proposal.spice.update(11, {"t0": 0.5})
    proposal.process([16, 17, 18], overwrite=True)

    print(proposal.data.p0016)
    print(proposal.data.p0017)
    print(proposal.data.p0018)

    # Scan 18 is a rotating-magnet measurement: field pairing must run in rotation mode,
    # while the electromagnet scan 17 keeps the +/- field sign pairing
    assert proposal.data.p0018.attrs["rotation_magnet"] == 1
    assert proposal.data.p0017.attrs["rotation_magnet"] == 0

    print("CI moke test completed successfully.")


if __name__ == "__main__":
    main()
