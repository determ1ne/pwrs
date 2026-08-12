# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, cast

import numpy as np

from ..corex import MatpowerConfig
from ..mp_opt_model import mpopt2qpopt
from .idx_brch import MU_SF, MU_ST, PF, PT, QF, QT, RATE_A
from .idx_bus import BUS_TYPE, LAM_P, LAM_Q, MU_VMAX, MU_VMIN, REF, VA, VM
from .idx_cost import COST, MODEL, NCOST, PW_LINEAR
from .idx_gen import MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PG
from .mpoption import mpoption


def dcopf_solver(om, mpopt: MatpowerConfig, nargout=1):
    """Solve a DC optimal power flow.

    Parameters
    ----------
    om : OPFModel
        OPF model object for the DC OPF problem.
    mpopt : dict
        MATPOWER options dict used to configure the QP solver and DC OPF
        formulation.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    dict or tuple
        ``(results, success, raw)`` in MATLAB-compatible form. ``results``
        is a MATPOWER case dict containing solved bus, gen, branch, shadow
        price, and optimization fields.
    """
    if not isinstance(mpopt, MatpowerConfig):
        mpopt = mpoption(mpopt)

    mpc = om.get_mpc()
    baseMVA = mpc["baseMVA"]
    bus = mpc["bus"].copy()
    gen = mpc["gen"].copy()
    branch = mpc["branch"].copy()
    gencost = mpc["gencost"]
    vv, ll = om.get_idx("var", "lin")
    ny = om.getN("var", "y")
    opt = cast(dict[str, Any], mpopt2qpopt(mpopt, om.problem_type()))
    if int(mpopt.opf.start) < 2 and str(opt.get("alg", "MIPS")).upper() == "MIPS":
        x0, xmin, xmax, _ = om.params_var()
        lb = xmin.copy()
        ub = xmax.copy()
        lb[np.isneginf(lb)] = -1e10
        ub[np.isposinf(ub)] = 1e10
        x0 = (lb + ub) / 2
        k = np.flatnonzero(np.isneginf(xmin) & np.isfinite(xmax))
        x0[k] = xmax[k] - 1
        k = np.flatnonzero(np.isfinite(xmin) & np.isposinf(xmax))
        x0[k] = xmin[k] + 1
        Varefs = bus[bus[:, BUS_TYPE] == REF, VA] * np.pi / 180.0
        x0[vv.i1["Va"] - 1 : vv.iN["Va"]] = Varefs[0]
        if ny > 0:
            ipwl = np.flatnonzero(gencost[:, MODEL] == PW_LINEAR)
            c = gencost[ipwl, NCOST].astype(int)
            ymax = [gencost[row, COST + 2 * ncost - 1] for row, ncost in zip(ipwl, c)]
            x0[vv.i1["y"] - 1 : vv.iN["y"]] = max(ymax) + 0.1 * abs(max(ymax))
        opt["x0"] = x0
    x, f, eflag, output, lambda_ = om.solve(opt)
    success = int(eflag == 1)
    if not np.any(np.isnan(x)):
        Va = x[vv.i1["Va"] - 1 : vv.iN["Va"]]
        Pg = x[vv.i1["Pg"] - 1 : vv.iN["Pg"]]
        bus[:, VM] = 1.0
        bus[:, VA] = Va * 180 / np.pi
        gen[:, PG] = Pg * baseMVA
        branch[:, [QF, QT]] = 0
        Bf = om.get_userdata("Bf")
        Pfinj = om.get_userdata("Pfinj")
        branch[:, PF] = (Bf @ Va + Pfinj) * baseMVA
        branch[:, PT] = -branch[:, PF]
    mu_l = lambda_["mu_l"]
    mu_u = lambda_["mu_u"]
    muLB = lambda_["lower"]
    muUB = lambda_["upper"]
    il = np.flatnonzero((branch[:, RATE_A] != 0) & (branch[:, RATE_A] < 1e10))
    bus[:, [LAM_P, LAM_Q, MU_VMIN, MU_VMAX]] = 0
    gen[:, [MU_PMIN, MU_PMAX, MU_QMIN, MU_QMAX]] = 0
    branch[:, [MU_SF, MU_ST]] = 0
    bus[:, LAM_P] = (mu_u[ll.i1["Pmis"] - 1 : ll.iN["Pmis"]] - mu_l[ll.i1["Pmis"] - 1 : ll.iN["Pmis"]]) / baseMVA
    if len(il):
        branch[il, MU_SF] = mu_u[ll.i1["Pf"] - 1 : ll.iN["Pf"]] / baseMVA
        branch[il, MU_ST] = mu_l[ll.i1["Pf"] - 1 : ll.iN["Pf"]] / baseMVA
    gen[:, MU_PMIN] = muLB[vv.i1["Pg"] - 1 : vv.iN["Pg"]] / baseMVA
    gen[:, MU_PMAX] = muUB[vv.i1["Pg"] - 1 : vv.iN["Pg"]] / baseMVA
    pimul = np.r_[mu_l - mu_u, -np.ones(1 if ny > 0 else 0), muLB - muUB]
    mu = {"var": {"l": muLB, "u": muUB}, "lin": {"l": mu_l, "u": mu_u}}
    results = dict(mpc)
    results["bus"] = bus
    results["branch"] = branch
    results["gen"] = gen
    results["om"] = om
    results["x"] = x
    results["mu"] = mu
    results["f"] = f
    raw = {"xr": x, "pimul": pimul, "info": eflag, "output": output}
    outputs = (results, success, raw)
    return outputs[:nargout] if nargout > 1 else results


def dcopf_solver_full(om, mpopt: MatpowerConfig) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Return DC OPF results, success flag and raw solver data."""
    return cast(tuple[dict[str, Any], float, dict[str, Any]], dcopf_solver(om, mpopt, nargout=3))
