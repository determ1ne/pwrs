# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import cast

import numpy as np

from ..corex import MatpowerConfig
from ..mp_opt_model import mpopt2nlpopt
from .idx_brch import F_BUS, MU_SF, MU_ST, PF, PT, QF, QT, RATE_A, T_BUS
from .idx_bus import BUS_TYPE, LAM_P, LAM_Q, MU_VMAX, MU_VMIN, REF, VA, VM, VMAX, VMIN
from .idx_cost import MODEL, NCOST, PW_LINEAR
from .idx_gen import GEN_BUS, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PG, QG, VG
from .makeYbus import makeYbus_full
from .mpoption import mpoption


def nlpopf_solver(om, mpopt: MatpowerConfig, nargout=1):
    """Solve AC optimal power flow using MP-Opt-Model.

    Parameters
    ----------
    om : OPFModel
        OPF model object for the AC OPF problem.
    mpopt : dict
        MATPOWER options dict used to configure the NLP solver and AC OPF
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
    bus = np.array(mpc["bus"], copy=True)
    gen = np.array(mpc["gen"], copy=True)
    branch = np.array(mpc["branch"], copy=True)
    gencost = mpc["gencost"]
    vv, ll, nne, nni = om.get_idx("var", "lin", "nle", "nli")

    nb = bus.shape[0]
    nl = branch.shape[0]
    ny = om.getN("var", "y")

    model = om.problem_type()
    opt = mpopt2nlpopt(mpopt, model)

    if float(mpopt.opf.start) < 2:
        x0, xmin, xmax, _ = om.params_var()
        s = 1.0
        lb = xmin.copy()
        ub = xmax.copy()
        lb[np.isneginf(xmin)] = -1e10
        ub[np.isposinf(xmax)] = 1e10
        x0 = (lb + ub) / 2
        k = np.flatnonzero(np.isneginf(xmin) & np.isfinite(xmax))
        x0[k] = xmax[k] - s
        k = np.flatnonzero(np.isfinite(xmin) & np.isposinf(xmax))
        x0[k] = xmin[k] + s
        Varefs = bus[bus[:, BUS_TYPE - 1] == REF, VA - 1] * (np.pi / 180.0)
        Vmax0 = np.minimum(bus[:, VMAX - 1], 1.5)
        Vmin0 = np.maximum(bus[:, VMIN - 1], 0.5)
        Vm0 = (Vmax0 + Vmin0) / 2
        if float(mpopt.opf.v_cartesian):
            V0 = Vm0 * np.exp(1j * Varefs[0])
            x0[vv.i1["Vr"] - 1 : vv.iN["Vr"]] = np.real(V0)
            x0[vv.i1["Vi"] - 1 : vv.iN["Vi"]] = np.imag(V0)
        else:
            x0[vv.i1["Va"] - 1 : vv.iN["Va"]] = Varefs[0]
            x0[vv.i1["Vm"] - 1 : vv.iN["Vm"]] = Vm0
            if ny > 0:
                ipwl = np.flatnonzero(gencost[:, MODEL - 1] == PW_LINEAR)
                c = gencost[ipwl, NCOST - 1].astype(int)
                ymax = [gencost[row, NCOST + 2 * ncost - 1] for row, ncost in zip(ipwl, c)]
                x0[vv.i1["y"] - 1 : vv.iN["y"]] = max(ymax) + 0.1 * abs(max(ymax))
        opt["x0"] = x0

    il = np.flatnonzero((branch[:, RATE_A - 1] != 0) & (branch[:, RATE_A - 1] < 1e10))

    x, f, eflag, output, lambda_ = om.solve(opt)
    success = int(eflag > 0)

    if float(mpopt.opf.v_cartesian):
        Vi = x[vv.i1["Vi"] - 1 : vv.iN["Vi"]]
        Vr = x[vv.i1["Vr"] - 1 : vv.iN["Vr"]]
        V = Vr + 1j * Vi
        Va = np.angle(V)
        Vm = np.abs(V)
    else:
        Va = x[vv.i1["Va"] - 1 : vv.iN["Va"]]
        Vm = x[vv.i1["Vm"] - 1 : vv.iN["Vm"]]
        V = Vm * np.exp(1j * Va)
    Pg = x[vv.i1["Pg"] - 1 : vv.iN["Pg"]]
    Qg = x[vv.i1["Qg"] - 1 : vv.iN["Qg"]]

    bus[:, VA - 1] = Va * 180 / np.pi
    bus[:, VM - 1] = Vm
    gen[:, PG - 1] = Pg * baseMVA
    gen[:, QG - 1] = Qg * baseMVA
    gen[:, VG - 1] = Vm[gen[:, GEN_BUS - 1].astype(int) - 1]

    Ybus, Yf, Yt = makeYbus_full(baseMVA, bus, branch)
    Sf = V[branch[:, F_BUS - 1].astype(int) - 1] * np.conjugate(Yf @ V)
    St = V[branch[:, T_BUS - 1].astype(int) - 1] * np.conjugate(Yt @ V)
    branch[:, PF - 1] = np.real(Sf) * baseMVA
    branch[:, QF - 1] = np.imag(Sf) * baseMVA
    branch[:, PT - 1] = np.real(St) * baseMVA
    branch[:, QT - 1] = np.imag(St) * baseMVA

    muSf = np.zeros(nl)
    muSt = np.zeros(nl)
    if il.size:
        if str(mpopt.opf.flow_lim)[0].upper() == "P":
            muSf[il] = lambda_["ineqnonlin"][nni.i1["Sf"] - 1 : nni.iN["Sf"]]
            muSt[il] = lambda_["ineqnonlin"][nni.i1["St"] - 1 : nni.iN["St"]]
        else:
            muSf[il] = 2 * lambda_["ineqnonlin"][nni.i1["Sf"] - 1 : nni.iN["Sf"]] * branch[il, RATE_A - 1] / baseMVA
            muSt[il] = 2 * lambda_["ineqnonlin"][nni.i1["St"] - 1 : nni.iN["St"]] * branch[il, RATE_A - 1] / baseMVA

    if float(mpopt.opf.v_cartesian):
        veq = np.asarray(om.userdata.get("veq", np.array([]))).reshape(-1).astype(int)
        if veq.size:
            lam = lambda_["eqnonlin"][nne.i1["Veq"] - 1 : nne.iN["Veq"]]
            mu_Vmax = np.zeros_like(lam)
            mu_Vmin = np.zeros_like(lam)
            mu_Vmax[lam > 0] = lam[lam > 0]
            mu_Vmin[lam < 0] = -lam[lam < 0]
            bus[veq - 1, MU_VMAX - 1] = mu_Vmax
            bus[veq - 1, MU_VMIN - 1] = mu_Vmin
        viq = np.asarray(om.userdata.get("viq", np.array([]))).reshape(-1).astype(int)
        if viq.size:
            bus[viq - 1, MU_VMAX - 1] = lambda_["ineqnonlin"][nni.i1["Vmax"] - 1 : nni.iN["Vmax"]]
            bus[viq - 1, MU_VMIN - 1] = lambda_["ineqnonlin"][nni.i1["Vmin"] - 1 : nni.iN["Vmin"]]
    else:
        bus[:, MU_VMAX - 1] = lambda_["upper"][vv.i1["Vm"] - 1 : vv.iN["Vm"]]
        bus[:, MU_VMIN - 1] = lambda_["lower"][vv.i1["Vm"] - 1 : vv.iN["Vm"]]

    gen[:, MU_PMAX - 1] = lambda_["upper"][vv.i1["Pg"] - 1 : vv.iN["Pg"]] / baseMVA
    gen[:, MU_PMIN - 1] = lambda_["lower"][vv.i1["Pg"] - 1 : vv.iN["Pg"]] / baseMVA
    gen[:, MU_QMAX - 1] = lambda_["upper"][vv.i1["Qg"] - 1 : vv.iN["Qg"]] / baseMVA
    gen[:, MU_QMIN - 1] = lambda_["lower"][vv.i1["Qg"] - 1 : vv.iN["Qg"]] / baseMVA

    if float(mpopt.opf.current_balance):
        VV = V / (V * np.conjugate(V))
        VVr = np.real(VV)
        VVi = np.imag(VV)
        lamM = lambda_["eqnonlin"][nne.i1["rImis"] - 1 : nne.iN["rImis"]]
        lamN = lambda_["eqnonlin"][nne.i1["iImis"] - 1 : nne.iN["iImis"]]
        bus[:, LAM_P - 1] = (VVr * lamM + VVi * lamN) / baseMVA
        bus[:, LAM_Q - 1] = (VVi * lamM - VVr * lamN) / baseMVA
    else:
        bus[:, LAM_P - 1] = lambda_["eqnonlin"][nne.i1["Pmis"] - 1 : nne.iN["Pmis"]] / baseMVA
        bus[:, LAM_Q - 1] = lambda_["eqnonlin"][nne.i1["Qmis"] - 1 : nne.iN["Qmis"]] / baseMVA
    branch[:, MU_SF - 1] = muSf / baseMVA
    branch[:, MU_ST - 1] = muSt / baseMVA

    nlnN = 2 * nb + 2 * nl
    kl = np.flatnonzero(lambda_["eqnonlin"][: 2 * nb] < 0)
    ku = np.flatnonzero(lambda_["eqnonlin"][: 2 * nb] > 0)
    nl_mu_l = np.zeros(nlnN)
    nl_mu_u = np.r_[np.zeros(2 * nb), muSf, muSt]
    nl_mu_l[kl] = -lambda_["eqnonlin"][kl]
    nl_mu_u[ku] = lambda_["eqnonlin"][ku]
    lam_nli = lambda_.get("ineqnonlin", np.array([]))

    mu = {
        "var": {"l": lambda_["lower"], "u": lambda_["upper"]},
        "nln": {"l": nl_mu_l, "u": nl_mu_u},
        "nle": lambda_["eqnonlin"],
        "nli": lam_nli,
        "lin": {"l": lambda_["mu_l"], "u": lambda_["mu_u"]},
    }

    results = dict(mpc)
    results["bus"] = bus
    results["branch"] = branch
    results["gen"] = gen
    results["om"] = om
    results["x"] = x
    results["mu"] = mu
    results["f"] = f

    pimul = np.r_[
        results["mu"]["nln"]["l"] - results["mu"]["nln"]["u"],
        results["mu"]["lin"]["l"] - results["mu"]["lin"]["u"],
        -np.ones(1 if ny > 0 else 0),
        results["mu"]["var"]["l"] - results["mu"]["var"]["u"],
    ]
    raw = {"xr": x, "pimul": pimul, "info": eflag, "output": output}
    outputs = (results, success, raw)
    return outputs[:nargout] if nargout > 1 else results


def nlpopf_solver_full(om, mpopt: MatpowerConfig) -> tuple[dict[str, object], float, dict[str, object]]:
    """Return nonlinear OPF results, success flag and raw solver data."""
    return cast(tuple[dict[str, object], float, dict[str, object]], nlpopf_solver(om, mpopt, nargout=3))
