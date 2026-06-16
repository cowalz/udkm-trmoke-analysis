# trMOKE Analysis 1.1.5

Time-resolved MOKE data analysis tools for data from the Femtomag lab (0.044) in the UDKM group at the University of Potsdam.
Intended for usage with data generated from the [2nd generation trMOKE measurement software](https://gitup.uni-potsdam.de/udkm/lab-software/trmoke).

## Installation

### Requirements

- Python 3.11 or higher

### Setup

In your project venv, install the latest release with pip (requires an SSH key with access to GitUP):
```bash
pip install git+https://gitup.uni-potsdam.de/udkm/tools/trmoke-analysis@main
```
Or use the public GitHub mirror:
```bash
pip install git+https://github.com/cowalz/udkm-trmoke-analysis@main
```
Replace `main` with `develop` tag if you want to install the potentially unstable development version.
For a previous version, replace `main` with the desired version tag, e.g. `1.0.0`.

This installs the `trmoke_analysis` package along with all runtime dependencies:
- astropy
- ipykernel
- matplotlib
- netCDF4
- scipy
- tqdm
- xarray

## Usage

See [Examples/example.ipynb](Examples/example.ipynb) for a complete workflow example.

An overview of the available spices (metadata/parameters for the data evaluation) can be found in [spice.md](spice.md).


## Contributing
Please report issues and improvements via the Git repository.
To contribute/push to the project, you need a GitUP account and be a member of the UDKM group. Contact the maintainers or senior group members for access.
To change the source code, first clone the repository somewhere on your local machine:

```bash
git clone git@gitup.uni-potsdam.de:udkm/tools/trmoke-analysis.git
```
and switch to the develop branch:
```bash
git checkout develop
```
This makes sure the stable version of the code remains unchanged until the updates are merged into the main branch for the next release. After making your changes, push them to the develop branch and create a merge request to the main branch for review.
Please talk to the maintainers before making your first contribution to get an introduction into the structure and ideas of this project

## Continuous Integration

A minimal GitLab CI test lives in [.scripts/ci_moke_test.py](.scripts/ci_moke_test.py). It runs the notebook's data-reduction path without plotting, processes the example scans, and checks that the expected processed NetCDF files are written.
Errors during the execution of this pipeline may indicate issues with code formatting (ruff) or bugs in the code itself. Please investigate and fix or discuss any issues before creating a new merge request.

## Authors

- Constantin Walz
- UDKM group

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
