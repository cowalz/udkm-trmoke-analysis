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

    from datareduction import DataProposal

    # Notebook flow without the plotting cells.
    proposal = DataProposal(
        PROPOSAL_ID,
        str(RAW_DATA_PATH),
        spice_path=str(SPICE_PATH),
        overwrite=True,
    )

    print(proposal.data.r0011)

    proposal.spice.seed({"t0": 0})
    proposal.process([6, 11, 12], overwrite=True)

    print(proposal.p0011)

    proposal.spice.update(11, {"t0": 0.5})
    proposal.process([6, 11, 12], overwrite=True)

    print(proposal.p0011)

    print("CI moke test completed successfully.")


if __name__ == "__main__":
    main()
