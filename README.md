# trMOKE Analysis

Time-resolved MOKE data analysis tools for data from the Femtomag lab (0.044) in the UDKM group at the University of Potsdam.
Intended for usage with data generated from the [2nd generation trMOKE measurement software](https://gitup.uni-potsdam.de/udkm/lab-software/trmoke).

## Installation

### Requirements

- Python 3.12 or higher

### Setup

Clone the repository and install with pip in editable mode:

```bash
git clone git@gitup.uni-potsdam.de:udkm/lab-software/trmoke.git
cd trmoke-analysis
pip install -e .
```

This installs the `trmoke_analysis` package along with all runtime dependencies:
- astropy
- ipykernel
- matplotlib
- netCDF4
- tqdm
- xarray

Update the package by pulling the latest changes from the git repo using `git pull` should automatically update the package when installed in editable mode. In case of issues, you can also run `pip install -e .` again to ensure all dependencies are up to date.
## Usage

See [Examples/example.ipynb](Examples/example.ipynb) for a complete workflow example.


## Contributing
Please report issues and improvements via the Git repository.
To contribute/push to the project, you need a GitUP account and be a member of the UDKM group. Contact the maintainers or senior group members for access.

Please talk to the maintainers before making your first contribution to get an introduction into the structure and ideas of this project

## Continuous Integration

A minimal GitLab CI test lives in [.scripts/ci_moke_test.py](.scripts/ci_moke_test.py). It runs the notebook's data-reduction path without plotting, processes the example scans, and checks that the expected processed NetCDF files are written.

## Authors

- Constantin Walz (cwalz@uni-potsdam.de)
- UDKM group (udkm-group@uni-potsdam.de)

## License
This project is licensed under the MIT License - see the LICENSE file for details.
