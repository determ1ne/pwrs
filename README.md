# pwrs

[![DOI](https://zenodo.org/badge/1187810140.svg)](https://doi.org/10.5281/zenodo.19145297)

> **⚠️ This project is still under heavy development, the functionality is not yet guaranteed to be stable or fully available.**

Pwrs is a Python package for power system simulation. It ports the well-known
[MATPOWER](https://github.com/MATPOWER/matpower) to Python and provides
functions for solving power flow, continuation power flow, and optimal power
flow problems.

The project aims to add type annotations to MATPOWER-style functions and to
refactor MATLAB functions with multiple input/output parameters into clearer,
more structured Python interfaces.
The PF, CPF, and OPF implementations are checked
via oracle-based tests against MATPOWER.

This project is **NOT** affiliated with or endorsed by the MATPOWER authors.

While the source code preserves the original functions as much as possible,
deprecated or compatibility-related functions may be omitted.

This version of pwrs targets MATPOWER version **7.1**.

## Quick Start

You can install pwrs by pip:

```bash
pip install pwrs
# or install the development version by
pip install git+https://github.com/determ1ne/pwrs
# install all optional dependencies by
pip install pwrs[all]
# or install the package for development by uv:
# uv sync --dev
# or by pip:
# pip install -e .
```

> **⚠️ Currently we recommend to install pwrs by `pip install git+https://github.com/determ1ne/pwrs`. **

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

This software is derived from MATPOWER, which is also licensed under the
3-Clause BSD License.

The case files distributed with pwrs are **NOT** covered by the BSD License.
In most cases, these data files have either been included with permission or
converted from data available from public sources.

Copyright 2026, Liangyu Zhang.  
Copyright (c) 1996–2020, Power Systems Engineering Research Center (PSERC)  
and individual contributors to MATPOWER.

See the [LICENSE](./LICENSE) file distributed with this software for details.

We, together with the MATPOWER authors, kindly request that publications based
on the use of pwrs or its included data files explicitly acknowledge this by
citing the appropriate paper(s) and the software itself.

### Papers

All publications derived from the use of pwrs, or the included data
files, should cite the 2011 MATPOWER paper:

>   R. D. Zimmerman, C. E. Murillo-Sanchez, and R. J. Thomas, "MATPOWER:
    Steady-State Operations, Planning and Analysis Tools for Power Systems
    Research and Education," *Power Systems, IEEE Transactions on*, vol. 26,
    no. 1, pp. 12-19, Feb. 2011.  
    doi: [10.1109/TPWRS.2010.2051168][13]

Publications derived from the use of the Pwrs Optimal Scheduling
Tool (POST) should cite the 2013 MOST paper, in addition to the
2011 MATPOWER paper above.

>   C. E. Murillo-Sanchez, R. D. Zimmerman, C. L. Anderson, and R. J. Thomas,
    "Secure Planning and Operations of Systems with Stochastic Sources,
    Energy Storage and Active Demand," *Smart Grid, IEEE Transactions on*,
    vol. 4, no. 4, pp. 2220-2229, Dec. 2013.  
    doi: [10.1109/TSG.2013.2281001][18]

Work making specific reference to the Pwrs Interior Point Solver
(PIPS) should also cite:

>   H. Wang, C. E. Murillo-Sánchez, R. D. Zimmerman, R. J. Thomas, "On
    Computational Issues of Market-Based Optimal Power Flow," *Power Systems,
    IEEE Transactions on*, vol. 22, no. 3, pp. 1185-1193, Aug. 2007.  
    doi: [10.1109/TPWRS.2007.901301][17]

NOTE: Some of the case files included with MATPOWER request the citation
of additional publications. This includes the ACTIVSg, PEGASE, and RTE
cases. Details are available in the help text at the top of the
corresponding case files.
