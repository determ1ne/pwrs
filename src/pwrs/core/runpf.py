# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy
import io
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..corex import MatpowerCase, MatpowerConfig
from ..utils import get_nested
from .bustypes import bustypes
from .dcpf import dcpf
from .ext2int import ext2int
from .fdpf import fdpf
from .gausspf import gausspf
from .idx_brch import PF, PT, QF, QT
from .idx_bus import BUS_TYPE, GS, PQ, PV, REF, VA, VM
from .idx_gen import GEN_BUS, GEN_STATUS, PG, QG, QMAX, QMIN, VG
from .int2ext import int2ext
from .loadcase import loadcase
from .makeB import makeB
from .makeBdc import makeBdc
from .makeSbus import makeSbus
from .makeYbus import makeYbus
from .mpoption import mpoption
from .newtonpf import newtonpf
from .newtonpf_I_cart import newtonpf_I_cart
from .newtonpf_I_hybrid import newtonpf_I_hybrid
from .newtonpf_I_polar import newtonpf_I_polar
from .newtonpf_S_cart import newtonpf_S_cart
from .newtonpf_S_hybrid import newtonpf_S_hybrid
from .pfsoln import pfsoln
from .printpf import printpf
from .radial_pf import radial_pf
from .savecase import savecase


def _as_array(value: Any, *, dtype=None) -> np.ndarray:
    return np.atleast_2d(np.array(value, dtype=dtype, copy=True))


def _have_zip_loads(mpopt_value: dict[str, Any]) -> bool:
    pw = np.asarray(get_nested(mpopt_value, ["exp", "sys_wide_zip_loads", "pw"], np.array([]))).reshape(-1)
    qw = np.asarray(get_nested(mpopt_value, ["exp", "sys_wide_zip_loads", "qw"], np.array([]))).reshape(-1)
    return (pw.size and np.any(pw[1:])) or (qw.size and np.any(qw[1:]))


def _select_newton_solver(mpopt_value) -> Callable[..., Any]:
    current_balance = mpopt_value.pf.current_balance
    v_cartesian = mpopt_value.pf.v_cartesian
    if current_balance:
        if v_cartesian == 0:
            return newtonpf_I_polar
        if v_cartesian == 1:
            return newtonpf_I_cart
        return newtonpf_I_hybrid
    if v_cartesian == 0:
        return newtonpf
    if v_cartesian == 1:
        return newtonpf_S_cart
    return newtonpf_S_hybrid


def _capture_printpf(results: dict[str, Any], mpopt_value: dict[str, Any]) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        printpf(results, 1, mpopt_value, nargout=0)
    return buf.getvalue()


def _normalize_case(mpc: dict[str, Any]) -> dict[str, Any]:
    mpc = copy.deepcopy(mpc)
    mpc["baseMVA"] = float(mpc["baseMVA"])
    mpc["bus"] = _as_array(mpc["bus"], dtype=float)
    mpc["gen"] = _as_array(mpc["gen"], dtype=float)
    mpc["branch"] = _as_array(mpc["branch"], dtype=float)
    if "gencost" in mpc:
        mpc["gencost"] = _as_array(mpc["gencost"], dtype=float)
    if "areas" in mpc:
        mpc["areas"] = _as_array(mpc["areas"], dtype=float)
    return mpc


def _apply_pf_alg_overrides(mpopt_value: dict[str, Any], alg: str) -> dict[str, Any]:
    if alg == "NR-SP":
        return mpoption(mpopt_value, "pf.current_balance", 0, "pf.v_cartesian", 0)
    if alg == "NR-SC":
        return mpoption(mpopt_value, "pf.current_balance", 0, "pf.v_cartesian", 1)
    if alg == "NR-SH":
        return mpoption(mpopt_value, "pf.current_balance", 0, "pf.v_cartesian", 2)
    if alg == "NR-IP":
        return mpoption(mpopt_value, "pf.current_balance", 1, "pf.v_cartesian", 0)
    if alg == "NR-IC":
        return mpoption(mpopt_value, "pf.current_balance", 1, "pf.v_cartesian", 1)
    if alg == "NR-IH":
        return mpoption(mpopt_value, "pf.current_balance", 1, "pf.v_cartesian", 2)
    return mpopt_value


def runpf(casedata: MatpowerCase | dict, mpopt: MatpowerConfig | None = None, fname="", solvedcase="", *, nargout=None):
    """Run a power flow.

    Parameters
    ----------
    casedata : MatpowerCase or dict
        MATPOWER case struct or dictionary.
    mpopt_value : MatpowerConfig, optional
        MATPOWER options dict controlling the PF algorithm, tolerances,
        output options, and related settings.
    fname : str, optional
        File name to which pretty-printed output is appended.
    solvedcase : str, optional
        File name where the solved case is saved in MATPOWER case format.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    dict or tuple
        With one or two outputs, returns the solved results dict and
        optional success flag. With more outputs, returns the MATLAB-style
        tuple for the solved case data and summary outputs.
    """
    if not isinstance(casedata, dict) and not isinstance(casedata, MatpowerCase):
        raise TypeError("runpf: Python port currently supports MATPOWER case structs and dictionaries only")

    # default arguments
    if mpopt is None:
        mpopt = mpoption()
    fname = "" if fname is None else str(fname)
    solvedcase = "" if solvedcase is None else str(solvedcase)

    # options
    qlim = mpopt.pf.enforce_q_lims  # enforce Q limits on gens?
    dc = mpopt.model.upper() == "DC"  # use DC formulation?

    # read data
    # TODO
    mpc = _normalize_case(loadcase(casedata, nargout=1))

    # add zero columns to branch for flows if needed
    if mpc["branch"].shape[1] < QT:
        mpc["branch"] = np.concatenate(
            [mpc["branch"], np.zeros((mpc["branch"].shape[0], QT - mpc["branch"].shape[1]))],
            axis=1,
        )

    mpc = ext2int(mpc, mpopt)
    baseMVA = float(mpc["baseMVA"])
    bus = _as_array(mpc["bus"], dtype=float)
    gen = _as_array(mpc["gen"], dtype=float)
    branch = _as_array(mpc["branch"], dtype=float)

    if bus.size > 0:
        ref, pv, pq = bustypes(bus, gen)
        on = np.flatnonzero(gen[:, GEN_STATUS - 1] > 0)
        gbus = gen[on, GEN_BUS - 1].astype(int)

        t0 = time.perf_counter()
        its = 0.0
        alg = str(get_nested(mpopt, ["pf", "alg"], "NR")).upper()

        if dc:
            Va0 = bus[:, VA - 1] * (np.pi / 180.0)
            B, Bf, Pbusinj, Pfinj = makeBdc(baseMVA, bus, branch)
            Pbus = (
                np.real(np.asarray(makeSbus(baseMVA, bus, gen, nargout=1)).reshape(-1))
                - np.asarray(Pbusinj).reshape(-1)
                - bus[:, GS - 1] / baseMVA
            )
            Va, success = dcpf(B, Pbus, Va0, ref, pv, pq, nargout=2)
            Va = np.asarray(Va).reshape(-1)
            its = 1.0
            branch[:, [QF - 1, QT - 1]] = 0.0
            branch[:, PF - 1] = np.asarray(Bf @ Va).reshape(-1) + np.asarray(Pfinj).reshape(-1)
            branch[:, PF - 1] *= baseMVA
            branch[:, PT - 1] = -branch[:, PF - 1]
            bus[:, VM - 1] = 1.0
            bus[:, VA - 1] = Va * (180.0 / np.pi)
            ref = np.asarray(ref, dtype=int).reshape(-1)
            refgen = np.zeros(ref.size, dtype=int)
            for k, ref_bus in enumerate(ref):
                temp = np.flatnonzero(gbus == ref_bus)
                refgen[k] = on[temp[0]]
            gen[refgen, PG - 1] = (
                gen[refgen, PG - 1] + np.asarray(B[ref - 1, :] @ Va).reshape(-1) * baseMVA - Pbus[ref - 1] * baseMVA
            )
        else:
            mpopt_value = _apply_pf_alg_overrides(mpopt, alg)
            if alg not in {"NR", "NR-SP", "NR-SC", "NR-SH", "NR-IP", "NR-IC", "NR-IH"}:
                if mpopt_value.pf.current_balance or mpopt_value.pf.v_cartesian:
                    raise ValueError(
                        f"runpf: power flow algorithm '{alg}' only supports power balance, polar version\nI.e. both 'pf.current_balance' and 'pf.v_cartesian' must be set to 0."
                    )
            if _have_zip_loads(mpopt_value):
                warnstr = ""
                if mpopt_value.pf.current_balance or mpopt_value.pf.v_cartesian:
                    warnstr = "Newton algorithm (current or cartesian/hybrid versions) do"
                elif alg == "GS":
                    warnstr = "Gauss-Seidel algorithm does"
                if warnstr:
                    print(f"warning: runpf: {warnstr} not support ZIP load model. Converting to constant power loads.")
                    mpopt_value = mpoption(
                        mpopt_value, "exp.sys_wide_zip_loads", {"pw": np.array([]), "qw": np.array([])}
                    )

            V0 = bus[:, VM - 1] * np.exp(1j * np.pi / 180.0 * bus[:, VA - 1])
            vcb = np.ones(V0.size)
            pq_idx = np.asarray(pq, dtype=int).reshape(-1) - 1
            vcb[pq_idx] = 0.0
            gbus_idx = gbus - 1
            k = np.flatnonzero(vcb[gbus_idx])
            if k.size:
                V0[gbus_idx[k]] = gen[on[k], VG - 1] / np.abs(V0[gbus_idx[k]]) * V0[gbus_idx[k]]

            if qlim:
                ref0 = np.asarray(ref, dtype=int).reshape(-1)
                Varef0 = bus[ref0 - 1, VA - 1].copy()
                limited = np.array([], dtype=int)
                fixedQg = np.zeros(gen.shape[0])

            Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch, nargout=3)
            repeat = True
            success = 0.0
            iterations = 0.0
            while repeat:
                Sbus_callable = lambda Vm, nargout=1: makeSbus(baseMVA, bus, gen, mpopt_value, Vm, nargout=nargout)
                if alg in {"NR", "NR-SP", "NR-SC", "NR-SH", "NR-IP", "NR-IC", "NR-IH"}:
                    newtonpf_fcn = _select_newton_solver(mpopt_value)
                    V, success, iterations = newtonpf_fcn(Ybus, Sbus_callable, V0, ref, pv, pq, mpopt_value)
                elif alg in {"FDXB", "FDBX"}:
                    Bp, Bpp = makeB(baseMVA, bus, branch, alg, nargout=2)
                    V, success, iterations = fdpf(Ybus, Sbus_callable, V0, Bp, Bpp, ref, pv, pq, mpopt_value, nargout=3)
                elif alg == "GS":
                    Sbus0 = makeSbus(baseMVA, bus, gen, nargout=1)
                    V, success, iterations = gausspf(Ybus, Sbus0, V0, ref, pv, pq, mpopt_value, nargout=3)
                elif alg in {"PQSUM", "ISUM", "YSUM"}:
                    mpc["bus"] = bus
                    mpc["gen"] = gen
                    mpc["branch"] = branch
                    mpc, success, iterations = radial_pf(mpc, mpopt_value, nargout=3)
                else:
                    raise ValueError(
                        f"runpf: '{alg}' is not a valid power flow algorithm. See 'pf.alg' details in MPOPTION help."
                    )
                its += float(iterations)

                if alg in {"NR", "NR-SP", "NR-SC", "NR-SH", "NR-IP", "NR-IC", "NR-IH", "FDXB", "FDBX", "GS"}:
                    bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, pv, pq, mpopt_value)
                else:
                    bus = _as_array(mpc["bus"], dtype=float)
                    gen = _as_array(mpc["gen"], dtype=float)
                    branch = _as_array(mpc["branch"], dtype=float)

                if success and qlim:
                    mx = np.flatnonzero(
                        (gen[:, GEN_STATUS - 1] > 0)
                        & (
                            gen[:, QG - 1]
                            > gen[:, QMAX - 1] + mpopt_value.opf.violation
                        )
                    )
                    mn = np.flatnonzero(
                        (gen[:, GEN_STATUS - 1] > 0)
                        & (
                            gen[:, QG - 1]
                            < gen[:, QMIN - 1] - mpopt_value.opf.violation
                        )
                    )
                    if mx.size or mn.size:
                        infeas = np.union1d(mx, mn)
                        bus_types = bus[gen[:, GEN_BUS - 1].astype(int) - 1, BUS_TYPE - 1]
                        remaining = np.flatnonzero(
                            (gen[:, GEN_STATUS - 1] > 0) & ((bus_types == PV) | (bus_types == REF))
                        )
                        if (
                            infeas.size == remaining.size
                            and np.array_equal(infeas, remaining)
                            and (mx.size == 0 or mn.size == 0)
                        ):
                            success = 0.0
                            break
                        if qlim == 2:
                            violations = np.r_[gen[mx, QG - 1] - gen[mx, QMAX - 1], gen[mn, QMIN - 1] - gen[mn, QG - 1]]
                            kmax = int(np.argmax(violations))
                            if kmax >= mx.size:
                                mn = np.array([mn[kmax - mx.size]])
                                mx = np.array([], dtype=int)
                            else:
                                mx = np.array([mx[kmax]])
                                mn = np.array([], dtype=int)
                        fixedQg[mx] = gen[mx, QMAX - 1]
                        fixedQg[mn] = gen[mn, QMIN - 1]
                        hit = np.r_[mx, mn]
                        gen[hit, QG - 1] = fixedQg[hit]
                        if np.asarray(ref, dtype=int).size > 1 and np.any(
                            bus[gen[hit, GEN_BUS - 1].astype(int) - 1, BUS_TYPE - 1] == REF
                        ):
                            raise ValueError(
                                "runpf: Sorry, MATPOWER cannot enforce Q limits for slack buses in systems with multiple slacks."
                            )
                        bus[gen[hit, GEN_BUS - 1].astype(int) - 1, BUS_TYPE - 1] = PQ
                        ref_temp = np.asarray(ref, dtype=int).reshape(-1)
                        ref, pv, pq = bustypes(bus, gen)
                        ref_arr = np.asarray(ref, dtype=int).reshape(-1)
                        pv_arr = np.asarray(pv, dtype=int).reshape(-1)
                        if ref_arr.size and not np.array_equal(ref_arr, ref_temp):
                            bus[ref_arr - 1, BUS_TYPE - 1] = REF
                            if pv_arr.size:
                                bus[pv_arr - 1, BUS_TYPE - 1] = PV
                        limited = np.r_[limited, hit]
                    else:
                        repeat = False
                else:
                    repeat = False
            if qlim and np.asarray(limited).size:
                ref_arr = np.asarray(ref, dtype=int).reshape(-1)
                if ref_arr.size and not np.array_equal(ref_arr, ref0):
                    bus[:, VA - 1] = bus[:, VA - 1] - bus[ref0 - 1, VA - 1] + Varef0

        mpc["et"] = time.perf_counter() - t0
        mpc["success"] = float(success)
        mpc["iterations"] = float(its)
        mpc["bus"] = bus
        mpc["gen"] = gen
        mpc["branch"] = branch
    else:
        t0 = time.perf_counter()
        success = 0.0
        its = 0.0
        if mpopt.verbose:
            print("Power flow not valid : MATPOWER case contains no connected buses")
        mpc["et"] = time.perf_counter() - t0
        mpc["success"] = success
        mpc["iterations"] = its
        mpc["bus"] = bus
        mpc["gen"] = gen
        mpc["branch"] = branch

    results = int2ext(mpc, nargout=1)
    off_gen = np.asarray(get_nested(results, ["order", "gen", "status", "off"], np.array([]))).reshape(-1).astype(int)
    if off_gen.size:
        results["gen"][np.ix_(off_gen - 1, [PG - 1, QG - 1])] = 0.0
    off_branch = (
        np.asarray(get_nested(results, ["order", "branch", "status", "off"], np.array([]))).reshape(-1).astype(int)
    )
    if off_branch.size:
        results["branch"][np.ix_(off_branch - 1, [PF - 1, QF - 1, PT - 1, QT - 1])] = 0.0

    if fname:
        text = _capture_printpf(
            results,
            mpoption(mpopt, "out.all", -1)
            if mpopt.out.all == 0
            else mpopt,
        )
        with Path(fname).open("a", encoding="utf-8") as fh:
            fh.write(text)
    printpf(results, 1, mpopt, nargout=0)
    if solvedcase:
        savecase(solvedcase, results, nargout=0)

    if nargout in (None, 1):
        return results
    if nargout == 2:
        return results, float(results["success"])
    if nargout > 2:
        return (
            float(results["baseMVA"]),
            _as_array(results["bus"], dtype=float),
            _as_array(results["gen"], dtype=float),
            _as_array(results["branch"], dtype=float),
            float(results["success"]),
            float(results["et"]),
        )
    return None
