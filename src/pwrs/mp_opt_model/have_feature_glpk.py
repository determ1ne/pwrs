# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import importlib.metadata
import shutil


def have_feature_glpk(nargout=1):
    import pyomo.environ as pyo

    solver = pyo.SolverFactory("glpk")
    torf = bool(solver.available(exception_flag=False))
    vstr = ""
    rdate = ""
    if torf:
        try:
            vstr = importlib.metadata.version("pyomo")
        except importlib.metadata.PackageNotFoundError:
            vstr = ""
        glpsol = shutil.which("glpsol")
        if glpsol:
            rdate = glpsol
    outputs = (torf, vstr, rdate)
    return outputs[:nargout] if nargout > 1 else torf
