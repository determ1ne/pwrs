# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from ..corex import MatpowerConfig
from ..mp_opt_model import OPFModel
from .idx_brch import F_BUS, RATE_A, T_BUS
from .idx_bus import BUS_TYPE, GS, PD, REF, VA, VM, VMAX, VMIN
from .idx_cost import COST, MODEL, NCOST, POLYNOMIAL, PW_LINEAR
from .idx_gen import GEN_BUS, GEN_STATUS, PG, PMAX, PMIN, QG, QMAX, QMIN, VG
from .makeAang import makeAang
from .makeApq import makeApq
from .makeAvl import makeAvl
from .makeAy import makeAy
from .makeBdc import makeBdc
from .makeYbus import makeYbus, makeYbus_full
from .opf_branch_ang_fcn import opf_branch_ang_fcn
from .opf_branch_ang_hess import opf_branch_ang_hess
from .opf_branch_flow_fcn import opf_branch_flow_fcn
from .opf_branch_flow_hess import opf_branch_flow_hess
from .opf_current_balance_fcn import opf_current_balance_fcn
from .opf_current_balance_hess import opf_current_balance_hess
from .opf_gen_cost_fcn import opf_gen_cost_fcn
from .opf_legacy_user_cost_fcn import opf_legacy_user_cost_fcn
from .opf_power_balance_fcn import opf_power_balance_fcn
from .opf_power_balance_hess import opf_power_balance_hess
from .opf_veq_fcn import opf_veq_fcn
from .opf_veq_hess import opf_veq_hess
from .opf_vlim_fcn import opf_vlim_fcn
from .opf_vlim_hess import opf_vlim_hess
from .opf_vref_fcn import opf_vref_fcn
from .opf_vref_hess import opf_vref_hess
from .pqcost import pqcost
from .run_userfcn import run_userfcn
from .mpoption import mpoption


def opf_setup(mpc, mpopt, nargout=1):
    """Construct an OPF model object from a MATPOWER case dict.

    Parameters
    ----------
    mpc : dict
        MATPOWER case dict with internal indexing and in-service equipment,
        typically produced by :func:`ext2int`.
    mpopt : dict
        MATPOWER options dict controlling AC/DC model selection, solver
        family, and OPF formulation options.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    OPFModel
        OPF model object ready to be passed to :func:`opf_execute`.
    """

    if not isinstance(mpopt, MatpowerConfig):
        mpopt = mpoption(mpopt)

    def _split_blocks(x, sizes):
        x = np.asarray(x).reshape(-1)
        out = []
        k = 0
        for n in sizes:
            out.append(x[k : k + n])
            k += n
        return out

    dc = mpopt.model.upper() == "DC"
    alg = mpopt.opf.ac.solver.upper()
    if not dc and alg == "DEFAULT":
        alg = "MIPS"
    use_vg = float(mpopt.opf.use_vg)
    vcart = (not dc) and bool(float(mpopt.opf.v_cartesian))
    current_balance = (not dc) and bool(float(mpopt.opf.current_balance))
    mpc = dict(mpc)
    baseMVA = mpc["baseMVA"]
    bus = np.array(mpc["bus"], copy=True)
    gen = np.array(mpc["gen"], copy=True)
    branch = np.array(mpc["branch"], copy=True)
    gencost = np.array(mpc["gencost"], copy=True)

    nb = bus.shape[0]
    ng = gen.shape[0]
    x_v_sizes = [nb, nb]
    x_full_sizes = [nb, nb, ng, ng]
    nlin = mpc.get("A", sparse.csc_matrix((0, 0))).shape[0] if "A" in mpc else 0
    nw = mpc.get("N", sparse.csc_matrix((0, 0))).shape[0] if "N" in mpc else 0
    nnle = 0
    nnli = 0

    if (not dc) and use_vg:
        Cg = sparse.csc_matrix(
            (gen[:, GEN_STATUS - 1] > 0, (gen[:, GEN_BUS - 1].astype(int) - 1, np.arange(ng))), shape=(nb, ng)
        )
        Vbg = Cg @ sparse.diags(gen[:, VG - 1], format="csc")
        Vmax_g = np.asarray(Vbg.max(axis=1).toarray()).reshape(-1)
        ib = np.flatnonzero(Vmax_g)
        Vmin_g = np.asarray(np.maximum((2 * Cg - Vbg).toarray(), 0).max(axis=1)).reshape(-1)
        Vmin_g[ib] = 2 - Vmin_g[ib]
        if use_vg == 1:
            bus[ib, VMAX - 1] = Vmax_g[ib]
            bus[ib, VMIN - 1] = Vmin_g[ib]
            bus[ib, VM - 1] = bus[ib, VMAX - 1]
        elif 0 < use_vg < 1:
            bus[ib, VMAX - 1] = (1 - use_vg) * bus[ib, VMAX - 1] + use_vg * Vmax_g[ib]
            bus[ib, VMIN - 1] = (1 - use_vg) * bus[ib, VMIN - 1] + use_vg * Vmin_g[ib]
        else:
            raise ValueError(f"opf_setup: option 'opf.use_vg' (= {use_vg:g}) cannot be negative or greater than 1")

    if (not dc) and "user_constraints" in mpc:
        if "nle" in mpc["user_constraints"]:
            for item in mpc["user_constraints"]["nle"]:
                nnle += int(np.sum(np.asarray(item[1]).reshape(-1)))
        if "nli" in mpc["user_constraints"]:
            for item in mpc["user_constraints"]["nli"]:
                nnli += int(np.sum(np.asarray(item[1]).reshape(-1)))

    pwl1 = np.flatnonzero((gencost[:, MODEL - 1] == PW_LINEAR) & (gencost[:, NCOST - 1] == 2))
    if pwl1.size:
        x0 = gencost[pwl1, COST - 1]
        y0 = gencost[pwl1, COST]
        x1 = gencost[pwl1, COST + 1]
        y1 = gencost[pwl1, COST + 2]
        m = (y1 - y0) / (x1 - x0)
        b = y0 - m * x0
        gencost[pwl1, MODEL - 1] = POLYNOMIAL
        gencost[pwl1, NCOST - 1] = 2
        gencost[pwl1, COST - 1 : COST + 1] = np.c_[m, b]

    pcost, qcost = pqcost(gencost, ng, nargout=2)
    ip0 = np.flatnonzero((pcost[:, MODEL - 1] == POLYNOMIAL) & (pcost[:, NCOST - 1] == 1))
    ip1 = np.flatnonzero((pcost[:, MODEL - 1] == POLYNOMIAL) & (pcost[:, NCOST - 1] == 2))
    ip2 = np.flatnonzero((pcost[:, MODEL - 1] == POLYNOMIAL) & (pcost[:, NCOST - 1] == 3))
    ip3 = np.flatnonzero((pcost[:, MODEL - 1] == POLYNOMIAL) & (pcost[:, NCOST - 1] > 3))
    if dc and ip3.size:
        raise ValueError("opf_setup: DC OPF cannot handle polynomial costs with higher than quadratic order.")

    kpg = np.zeros(ng)
    cpg = np.zeros(ng)
    if ip2.size:
        Qpg = np.zeros(ng)
        Qpg[ip2] = 2 * pcost[ip2, COST - 1] * baseMVA**2
        cpg[ip2] = cpg[ip2] + pcost[ip2, COST] * baseMVA
        kpg[ip2] = kpg[ip2] + pcost[ip2, COST + 1]
    else:
        Qpg = []
    if ip1.size:
        cpg[ip1] = cpg[ip1] + pcost[ip1, COST - 1] * baseMVA
        kpg[ip1] = kpg[ip1] + pcost[ip1, COST]
    if ip0.size:
        kpg[ip0] = kpg[ip0] + pcost[ip0, COST - 1]

    cqg = []
    iq3 = np.array([], dtype=int)
    if np.size(qcost):
        iq0 = np.flatnonzero((qcost[:, MODEL - 1] == POLYNOMIAL) & (qcost[:, NCOST - 1] == 1))
        iq1 = np.flatnonzero((qcost[:, MODEL - 1] == POLYNOMIAL) & (qcost[:, NCOST - 1] == 2))
        iq2 = np.flatnonzero((qcost[:, MODEL - 1] == POLYNOMIAL) & (qcost[:, NCOST - 1] == 3))
        iq3 = np.flatnonzero((qcost[:, MODEL - 1] == POLYNOMIAL) & (qcost[:, NCOST - 1] > 3))
        kqg = np.zeros(ng)
        cqg = np.zeros(ng)
        if iq2.size:
            Qqg = np.zeros(ng)
            Qqg[iq2] = 2 * qcost[iq2, COST - 1] * baseMVA**2
            cqg[iq2] = cqg[iq2] + qcost[iq2, COST] * baseMVA
            kqg[iq2] = kqg[iq2] + qcost[iq2, COST + 1]
        else:
            Qqg = []
        if iq1.size:
            cqg[iq1] = cqg[iq1] + qcost[iq1, COST - 1] * baseMVA
            kqg[iq1] = kqg[iq1] + qcost[iq1, COST]
        if iq0.size:
            kqg[iq0] = kqg[iq0] + qcost[iq0, COST - 1]
    else:
        Qqg = []
        kqg = np.array([])

    Aang, lang, uang, iang = makeAang(baseMVA, branch, nb, mpopt, nargout=4)

    Va = bus[:, VA - 1] * np.pi / 180.0
    refs = np.flatnonzero(bus[:, BUS_TYPE - 1] == REF)
    Vau = np.full(nb, np.inf)
    Val = -Vau.copy()
    Vau[refs] = Va[refs]
    Val[refs] = Va[refs]
    Pg = gen[:, PG - 1] / baseMVA
    Pmin = gen[:, PMIN - 1] / baseMVA
    Pmax = gen[:, PMAX - 1] / baseMVA

    if dc:
        ny = int(np.sum(gencost[:, MODEL - 1] == PW_LINEAR))
        Ay, by = (
            makeAy(baseMVA, ng, gencost, 1, [], 1 + ng, nargout=2) if ny else (sparse.csc_matrix((0, ng)), np.array([]))
        )
        nx = nb + ng
        user_vars = ["Va", "Pg"]
        ycon_vars = ["Pg", "y"]
    else:
        Vm = bus[:, VM - 1]
        Qg = gen[:, QG - 1] / baseMVA
        Qmin = gen[:, QMIN - 1] / baseMVA
        Qmax = gen[:, QMAX - 1] / baseMVA
        if vcart:
            V = Vm * np.exp(1j * Va)
            Vr = np.real(V)
            Vi = np.imag(V)

        il = np.flatnonzero((branch[:, RATE_A - 1] != 0) & (branch[:, RATE_A - 1] < 1e10)) + 1
        Ybus, Yf, Yt = makeYbus_full(baseMVA, bus, branch)
        branch_il = branch[il - 1, :] if il.size else branch[:0, :]
        mpc_internal = mpc | {
            "bus": bus,
            "gen": gen,
            "branch": branch,
            "gencost": gencost,
            "_opf_branch_il": branch_il,
            "_opf_flow_f_idx": branch_il[:, F_BUS - 1].astype(int) - 1 if il.size else np.zeros(0, dtype=int),
            "_opf_flow_t_idx": branch_il[:, T_BUS - 1].astype(int) - 1 if il.size else np.zeros(0, dtype=int),
            "_opf_flow_max": branch_il[:, RATE_A - 1] / baseMVA if il.size else np.zeros(0),
        }
        split_v = lambda x: _split_blocks(x, x_v_sizes)
        split_full = lambda x: _split_blocks(x, x_full_sizes)
        Avl, lvl, uvl = makeAvl(mpc_internal, nargout=3)
        Apqh, ubpqh, Apql, ubpql, Apqdata = makeApq(baseMVA, gen, nargout=5)

        if vcart:
            user_vars = ["Vr", "Vi", "Pg", "Qg"]
            nodal_balance_vars = ["Vr", "Vi", "Pg", "Qg"]
            flow_lim_vars = ["Vr", "Vi"]
        else:
            user_vars = ["Va", "Vm", "Pg", "Qg"]
            nodal_balance_vars = ["Va", "Vm", "Pg", "Qg"]
            flow_lim_vars = ["Va", "Vm"]
        ycon_vars = ["Pg", "Qg", "y"]

        if current_balance:
            mis_cons = ["rImis", "iImis"]
            fcn_mis = lambda x: opf_current_balance_fcn(
                split_full(x),
                mpc_internal,
                Ybus,
                mpopt,
                nargout=2,
            )
            hess_mis = lambda x, lam: opf_current_balance_hess(
                split_full(x),
                lam,
                mpc_internal,
                Ybus,
                mpopt,
                nargout=1,
            )
        else:
            mis_cons = ["Pmis", "Qmis"]
            fcn_mis = lambda x: opf_power_balance_fcn(
                split_full(x),
                mpc_internal,
                Ybus,
                mpopt,
                nargout=2,
            )
            hess_mis = lambda x, lam: opf_power_balance_hess(
                split_full(x),
                lam,
                mpc_internal,
                Ybus,
                mpopt,
                nargout=1,
            )
        Yf_lim = Yf[il - 1, :] if il.size else Yf[:0, :]
        Yt_lim = Yt[il - 1, :] if il.size else Yt[:0, :]
        fcn_flow = lambda x: opf_branch_flow_fcn(
            split_v(x),
            mpc_internal,
            Yf_lim,
            Yt_lim,
            il,
            mpopt,
            nargout=2,
        )
        hess_flow = lambda x, lam: opf_branch_flow_hess(
            split_v(x),
            lam,
            mpc_internal,
            Yf_lim,
            Yt_lim,
            il,
            mpopt,
            nargout=1,
        )
        if vcart:
            fcn_vref = lambda x: opf_vref_fcn(
                split_v(x),
                mpc_internal,
                refs + 1,
                mpopt,
                nargout=2,
            )
            hess_vref = lambda x, lam: opf_vref_hess(
                split_v(x),
                lam,
                mpc_internal,
                refs + 1,
                mpopt,
                nargout=1,
            )
            veq = np.flatnonzero(bus[:, VMIN - 1] == bus[:, VMAX - 1]) + 1
            viq = np.flatnonzero(bus[:, VMIN - 1] != bus[:, VMAX - 1]) + 1
            nveq = len(veq)
            nvlims = len(viq)
            if nveq:
                fcn_veq = lambda x: opf_veq_fcn(
                    split_v(x),
                    mpc_internal,
                    veq,
                    mpopt,
                    nargout=2,
                )
                hess_veq = lambda x, lam: opf_veq_hess(
                    split_v(x),
                    lam,
                    mpc_internal,
                    veq,
                    mpopt,
                    nargout=1,
                )
            fcn_vlim = lambda x: opf_vlim_fcn(
                split_v(x),
                mpc_internal,
                viq,
                mpopt,
                nargout=2,
            )
            hess_vlim = lambda x, lam: opf_vlim_hess(
                split_v(x),
                lam,
                mpc_internal,
                viq,
                mpopt,
                nargout=1,
            )
            fcn_ang = lambda x: opf_branch_ang_fcn(split_v(x), Aang, lang, uang, nargout=2)
            hess_ang = lambda x, lam: opf_branch_ang_hess(split_v(x), lam, Aang, lang, uang, nargout=1)
        if ip3.size:
            cost_Pg = lambda x: opf_gen_cost_fcn([np.asarray(x).reshape(-1)], baseMVA, pcost, ip3, mpopt, nargout=3)
        if np.size(qcost) and iq3.size:
            cost_Qg = lambda x: opf_gen_cost_fcn([np.asarray(x).reshape(-1)], baseMVA, qcost, iq3, mpopt, nargout=3)

        ny = int(np.sum(gencost[:, MODEL - 1] == PW_LINEAR))
        Ay, by = (
            makeAy(baseMVA, ng, gencost, 1, 1 + ng, 1 + ng + ng, nargout=2)
            if ny
            else (sparse.csc_matrix((0, 2 * ng)), np.array([]))
        )
        nx = 2 * nb + 2 * ng

    nz = mpc["A"].shape[1] - nx if nlin else 0
    if nz < 0:
        raise ValueError(f"opf_setup: user supplied A matrix must have at least {nx} columns.")
    if nw and not nlin and mpc["N"].shape[1] != nx:
        raise ValueError(f"opf_setup: user supplied N matrix must have {nx} columns.")

    om = OPFModel(mpc | {"bus": bus, "gen": gen, "branch": branch, "gencost": gencost})
    if pwl1.size:
        om.userdata["pwl1"] = pwl1 + 1
    om.userdata["iang"] = iang

    if dc:
        B, Bf, Pbusinj, Pfinj = makeBdc(baseMVA, bus, branch)
        Pbusinj = np.asarray(Pbusinj).reshape(-1)
        Pfinj = np.asarray(Pfinj).reshape(-1)
        neg_Cg = sparse.csc_matrix((-np.ones(ng), (gen[:, GEN_BUS - 1].astype(int) - 1, np.arange(ng))), shape=(nb, ng))
        Amis = sparse.hstack([B, neg_Cg], format="csc")
        bmis = (-(bus[:, PD - 1] + bus[:, GS - 1]) / baseMVA - Pbusinj).reshape(-1)
        il = np.flatnonzero((branch[:, RATE_A - 1] != 0) & (branch[:, RATE_A - 1] < 1e10))
        upf = branch[il, RATE_A - 1] / baseMVA - Pfinj[il] if il.size else np.array([])
        upt = branch[il, RATE_A - 1] / baseMVA + Pfinj[il] if il.size else np.array([])
        om.userdata["Bf"] = Bf
        om.userdata["Pfinj"] = Pfinj
        om.add_var("Va", nb, Va, Val, Vau)
        om.add_var("Pg", ng, Pg, Pmin, Pmax)
        om.add_lin_constraint("Pmis", Amis, bmis, bmis, ["Va", "Pg"])
        om.add_lin_constraint("Pf", Bf[il, :], -upt, upf, ["Va"])
        om.add_lin_constraint("ang", Aang, lang, uang, ["Va"])
        if np.size(cpg):
            om.add_quad_cost("polPg", sparse.diags(Qpg, format="csc"), cpg, kpg, ["Pg"])
    else:
        om.userdata["Apqdata"] = Apqdata
        if vcart:
            Vclim = 1.1 * bus[:, VMAX - 1]
            om.add_var("Vr", nb, Vr, -Vclim, Vclim)
            om.add_var("Vi", nb, Vi, -Vclim, Vclim)
        else:
            om.add_var("Va", nb, Va, Val, Vau)
            om.add_var("Vm", nb, Vm, bus[:, VMIN - 1], bus[:, VMAX - 1])
        om.add_var("Pg", ng, Pg, Pmin, Pmax)
        om.add_var("Qg", ng, Qg, Qmin, Qmax)
        om.add_nln_constraint(mis_cons, np.array([nb, nb]), 1, fcn_mis, hess_mis, nodal_balance_vars)
        om.add_nln_constraint(["Sf", "St"], np.array([len(il), len(il)]), 0, fcn_flow, hess_flow, flow_lim_vars)
        if vcart:
            om.userdata["veq"] = veq
            om.userdata["viq"] = viq
            om.add_nln_constraint("Vref", len(refs), 1, fcn_vref, hess_vref, ["Vr", "Vi"])
            if nveq:
                om.add_nln_constraint("Veq", nveq, 1, fcn_veq, hess_veq, ["Vr", "Vi"])
            om.add_nln_constraint(["Vmin", "Vmax"], np.array([nvlims, nvlims]), 0, fcn_vlim, hess_vlim, ["Vr", "Vi"])
            om.add_nln_constraint(
                ["angL", "angU"], np.array([len(iang), len(iang)]), 0, fcn_ang, hess_ang, ["Vr", "Vi"]
            )
        om.add_lin_constraint("PQh", Apqh, [], ubpqh, ["Pg", "Qg"])
        om.add_lin_constraint("PQl", Apql, [], ubpql, ["Pg", "Qg"])
        om.add_lin_constraint("vl", Avl, lvl, uvl, ["Pg", "Qg"])
        if not vcart:
            om.add_lin_constraint("ang", Aang, lang, uang, ["Va"])
        if np.size(cpg):
            om.add_quad_cost("polPg", sparse.diags(Qpg, format="csc"), cpg, kpg, ["Pg"])
        if isinstance(cqg, np.ndarray) and cqg.size:
            om.add_quad_cost("polQg", sparse.diags(Qqg, format="csc"), cqg, kqg, ["Qg"])
        if ip3.size:
            om.add_nln_cost("polPg", 1, cost_Pg, ["Pg"])
        if np.size(qcost) and iq3.size:
            om.add_nln_cost("polQg", 1, cost_Qg, ["Qg"])

    if ny > 0:
        om.add_var("y", ny)
        om.add_lin_constraint("ycon", Ay, [], by, ycon_vars)
        om.add_quad_cost("pwl", sparse.csc_matrix((0, 0)), np.ones(ny), 0, ["y"])

    if nz > 0:
        z0 = np.asarray(mpc.get("z0", np.zeros(nz))).reshape(-1)
        zl = np.asarray(mpc.get("zl", -np.inf * np.ones(nz))).reshape(-1)
        zu = np.asarray(mpc.get("zu", np.inf * np.ones(nz))).reshape(-1)
        om.add_var("z", nz, z0, zl, zu)
        user_vars = user_vars + ["z"]
    if nlin:
        om.add_lin_constraint("usr", mpc["A"], mpc["l"], mpc["u"], user_vars)
    if nnle:
        raise NotImplementedError("opf_setup: user-defined nonlinear equality constraints not yet implemented")
    if nnli:
        raise NotImplementedError("opf_setup: user-defined nonlinear inequality constraints not yet implemented")
    if nw:
        user_cost = {"N": mpc["N"], "Cw": mpc["Cw"]}
        if "fparm" in mpc and len(mpc["fparm"]):
            user_cost["dd"] = mpc["fparm"][:, 0]
            user_cost["rh"] = mpc["fparm"][:, 1]
            user_cost["kk"] = mpc["fparm"][:, 2]
            user_cost["mm"] = mpc["fparm"][:, 3]
        if "H" in mpc and np.size(mpc["H"]):
            user_cost["H"] = mpc["H"]
        om.add_legacy_cost("usr", user_cost, user_vars)

    om = run_userfcn(mpc.get("userfcn", []), "formulation", om, mpopt, nargout=1)

    cp = om.params_legacy_cost()[0]
    N = cp["N"]
    H = cp["H"]
    Cw = cp["Cw"]
    rh = cp["rh"]
    mm = cp["mm"]
    nw2, _ = N.shape
    if nw2:
        if np.any(cp["dd"] != 1) or np.any(cp["kk"]):
            if dc:
                if np.any(cp["dd"] != 1):
                    raise ValueError("opf_setup: DC OPF can only handle legacy user-defined costs with d = 1")
                if np.any(cp["kk"]):
                    raise ValueError(
                        'opf_setup: DC OPF can only handle legacy user-defined costs with no "dead zone", i.e. k = 0'
                    )
            else:
                legacy_cost_fcn = lambda x: opf_legacy_user_cost_fcn(x, cp, nargout=3)
                om.add_nln_cost("usr", 1, legacy_cost_fcn)
        else:
            M = sparse.diags(mm, format="csc")
            MN = M @ N
            MR = M @ rh
            HMR = H @ MR
            HtMR = H.T @ MR
            Q = MN.T @ H @ MN
            c = np.asarray(MN.T @ (Cw - 0.5 * (HMR + HtMR))).reshape(-1)
            k = float(((0.5 * HtMR - Cw).T @ MR))
            om.add_quad_cost("usr", Q, c, k)

    return om if nargout == 1 else (om,)
