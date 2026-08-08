# pwrs

[![DOI](https://zenodo.org/badge/1187810140.svg)](https://doi.org/10.5281/zenodo.19145297)

> **⚠️ This project is still under heavy development, the functionality is not yet guaranteed to be stable or fully available.**

Pwrs is a Python package for power system simulation. It ports the well-known
[MATPOWER](https://github.com/MATPOWER/matpower) and 
[PowerModels.jl](https://github.com/lanl-ansi/powermodels.jl) to Python,
providing functions for solving power flow, continuation power flow, and optimal power
flow problems.

The project aims to add type annotations to MATPOWER-style functions and to
refactor MATLAB functions with multiple input/output parameters into clearer,
more structured Python interfaces.

This project is **NOT** affiliated with or endorsed by the MATPOWER or PowerModels.jl authors.

While the source code preserves the original functions as much as possible,
deprecated or compatibility-related functions may be omitted.

## Quick Start

You can install pwrs by pip:

```bash
pip install pwrs
# or install the development version by
pip install git+https://github.com/determ1ne/pwrs@dev
# install all optional dependencies by
pip install pwrs[all]
# or install the package for development by uv:
# uv sync --dev
# or by pip:
# pip install -e .
```

Run an OPF by:

```python
import pwrs as mp

mpc = mp.case9241pegase()
opt = mp.mpoption()
# opt.mips.linsolver = 'LU'
result = mp.runopf(mpc, opt)
```

## License and Terms of Use

Pwrs is distributed under the 3-Clause BSD License.

This software is derived from:
- MATPOWER, licensed under the 3-Clause BSD License.
- PowerModels.jl, licensed under the BSD license.

The case files distributed with pwrs are **NOT** covered by the BSD License.
In most cases, these data files have either been included with permission or
converted from data available from public sources.

Copyright 2026, Liangyu Zhang.  

See the [LICENSE](./LICENSE) file distributed with this software for details.

### Citation Guidelines

Users of this project are encouraged to cite the software:

> Liangyu Zhang (\<YEAR\>). pwrs [Software]. 
> Available: https://github.com/determ1ne/pwrs 
> doi: [10.5281/zenodo.19145297]

Publications derived from the use of functions
originated from the MATPOWER software should cite the 2011 MATPOWER paper:

>   R. D. Zimmerman, C. E. Murillo-Sanchez, and R. J. Thomas, "MATPOWER:
>   Steady-State Operations, Planning and Analysis Tools for Power Systems
>   Research and Education," *Power Systems, IEEE Transactions on*, vol. 26,
>   no. 1, pp. 12-19, Feb. 2011.  
>   doi: [10.1109/TPWRS.2010.2051168][13]

Work making specific reference to the Pwrs Interior Point Solver
(PIPS) should also cite:

>   H. Wang, C. E. Murillo-Sánchez, R. D. Zimmerman, R. J. Thomas, "On
>   Computational Issues of Market-Based Optimal Power Flow," *Power Systems,
>   IEEE Transactions on*, vol. 22, no. 3, pp. 1185-1193, Aug. 2007.  
>   doi: [10.1109/TPWRS.2007.901301][17]

Publications derived from the use of functions
originated from the PowerModels.jl software should cite with:

```bib
@inproceedings{8442948,
  author = {Carleton Coffrin and Russell Bent and Kaarthik Sundar and Yeesian Ng and Miles Lubin},
  title = {PowerModels.jl: An Open-Source Framework for Exploring Power Flow Formulations},
  booktitle = {2018 Power Systems Computation Conference (PSCC)},
  year = {2018},
  month = {June},
  pages = {1-8},
  doi = {10.23919/PSCC.2018.8442948}
}
```

NOTE: Some of the case files included  request the citation
of additional publications. This includes the ACTIVSg, PEGASE, and RTE
cases. Details are available in the help text at the top of the
corresponding case files.
