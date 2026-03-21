# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import shutil


def have_feature_ipopt(nargout=1):
    import pyomo.environ as pyo

    pyomo_ipopt = pyo.SolverFactory("ipopt").available(exception_flag=False)
    cyipopt = importlib.util.find_spec("cyipopt") is not None
    torf = bool(pyomo_ipopt or cyipopt)
    vstr = ""
    rdate = ""
    if pyomo_ipopt:
        ipopt_bin = shutil.which("ipopt")
        if ipopt_bin:
            rdate = ipopt_bin
    outputs = (torf, vstr, rdate)
    return outputs[:nargout] if nargout > 1 else torf
