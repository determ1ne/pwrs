# Getting started

## Installation

We recommend create an isolated environment with `conda-forge` compatible software first:

```bash
conda create -c conda-forge -n pwrsenv python=3.13
conda activate pwrsenv
pip install pwrs[all]
# or install the dev version
# pip install git+https://github.com/determ1ne/pwrs@dev
```

For OPF setup with **external optimizers** (for example, Ipopt), see
{ref}`ext-optimizers` and the
[PowerModels-compatible OPF](optimal-power-flow/power-models.md) guide.

## Load a built-in case

Built-in MATPOWER cases are available from the top-level package for convenience.
Additionaly, [Power Grid Lib - Optimal Power Flow](https://github.com/power-grid-lib/pglib-opf)
cases are available from `pwrs.data.pglibopf`.

```python
import pwrs as mp

# MATPOWER case
mpc = mp.case9()
mpc = mp.data.matpower.case14()
# pglib-opf case
mpc = mp.data.pglibopf.pglib_opf_case14_ieee()
```

## Run a power flow

```python
import pwrs as mp

mpc = mp.case9()
result = mp.runpf(mpc)

print(result.success)
print(result.bus)
```

See the [Power flow](power-flow.md) guide for the workflows.

## Run an optimal power flow

```python
import pwrs as mp

result = mp.runopf(mp.case30())
print(result.success)
```

See the [Optimal power flow](optimal-power-flow.md) guide for the workflows.

(ext-optimizers)=
## Setup with external optimizers

### IPOPT

You can install prebuilt binaries from conda-forge:

```bash
conda install -c conda-forge ipopt cyipopt
```

#### Coin-HSL support

You may obtain or purchase the source code from the official store of Coin-HSL.

First, install the dependencies needed to build the HSL library:

```bash
# activate conda environment
conda activate pwrsenv
# install compiler and dependencies
conda install -c conda-forge gfortran pkg-config meson metis openblas pkg-config
```

Then compile the HSL library with prefix of the conda environment root:

```bash
cd coinhsl
# on Linux
meson setup builddir --buildtype=release --prefix=/YOUR_CONDA_ROOT/envs/pwrsenv
# on Windows
# meson setup builddir --buildtype=release --prefix=C:\YOUR_CONDA_ROOT\envs\pwrsenv --Dlibblas=openblas -Dliblapack=openblas
meson compile -C builddir
meson install -C builddir
```

Verify the installation by running an OPF with an HSL linear solver. The
`POWER_MODELS` backend is implemented with Pyomo/CVXPY in Python; it does not
start Julia or call PowerModels.jl.

```python
import pwrs as mp
from pwrs.data.pglibopf import pglib_opf_case1354_pegase

mpc = pglib_opf_case1354_pegase()
mpopt = mp.MatpowerConfig()
mpopt.opf.backend = "POWER_MODELS"
mpopt.opf.power_models.formulation = "ACP"
mpopt.opf.power_models.solver = "IPOPT"
mpopt.ipopt.opts.update(
    {
        "linear_solver": "ma27",
        "hsllib": "libcoinhsl.so",  # or the path to your platform-specific library
        "print_level": 5,
    }
)

result = mp.runopf(mpc, mpopt)
```

The output should be similar to the following:

```text
This is Ipopt version 3.14.19, running with linear solver ma27.
```
