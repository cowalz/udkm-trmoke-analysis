# Spice overview
Spices represent metadata or parameters for the data evaluation that usually need manual adjustment in an iterative 
project. A common example in pump-probe experiments is the time-zero (t0) which is usually manually set to either the
beginning or the largest slope of the experimental signal.

## Using spice

Spice needs to be added to a proposal object as an attribute. Creating a new spice is called "seeding":
```python
init_spice = {'t0': 0}
DataProposal.spice.seed(init_spice)
```
The interface to spice is via dictionary structures. Seeding sets a new default value for any given spice for the entire proposal.
Usually, these spice parameters are modied throughout the propsal if for example the measurement geometry changes.
This is reflected in the data processing by "updating" the spice at a specific anchor point.
This anchor point is simply the scan ID (i.e. 0042) at which the spice parameter(s) have changed.
```python
DataProposal.spice.update(42, {'t0': 10})
```
In this example, time-zero is shifted by 10ps for all scans with a number >= 42.
Only expliciitly specified spice parameters are updated.
Spices can be updated multiple times at different anchor points.
To keep track of all the spice parameters and their updates, we can simply call the spice attribute of the proposal object to gat an overview table when and which spice parameters are updated:
```python
print(DataProposal.spice)
```
**Spice  |  1 parameters  |  16 scans  |  2 explicit entries**

| scan_id | t0 |
|---------|----|
| 0       | 0  |
| 42      | 10 |

## Remote spice repository

Spice data is automatically synchronized to a remote spice repository to keep a global and public record of the used parameters to help share data with others.
The `autofetch_spice` option when creating the proposal object can be used to automatically fetch the latest spice data from the remote repository.
Whenever a scan is processed, the current spice will be uploaded to the remote repository.

## Overview Table

This overview table is meant to provide a quick reference for the different spices available and to which recipe they
can apply.

| Spice name | Description | Recipe(s) | Values |
|------------|-------------|-----------|--------|
| `t0` | Time-zero for time axis, usually set to the beginning or largest slope of the signal | moke_pp, hysteresis | float (default: 0) |
| `sigma` | Sigma value for sigma-clipping during averaging, higher values lead to more aggressive clipping of outliers | moke_pp, hysteresis | float > 0 (default: -1, no clipping) |
| `drift_correct` | Whether to apply drift correction to the signal, which can help remove slow drifts in the data | hysteresis | bool (default: False) |
| `shift_hysteresis` | Whether to shift the hysteresis loops along the y-axis (along the magnetization axis) | hysteresis | bool (default: False) |
| `shift_hysteresis_x` | Whether to shift the hysteresis loops along the x-axis (along the field axis). Requires `shift_hysteresis` == True | hysteresis | bool (default: False) |
|