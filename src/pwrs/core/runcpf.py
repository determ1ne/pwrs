# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy
import io
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .bustypes import bustypes
from .cpf_corrector import cpf_corrector
from .cpf_current_mpc import cpf_current_mpc
from .cpf_detect_events import cpf_detect_events
from .cpf_predictor import cpf_predictor
from .cpf_register_callback import cpf_register_callback
from .cpf_register_event import cpf_register_event
from .cpf_tangent import cpf_tangent
from .ext2int import ext2int
from .i2e_data import i2e_data
from .idx_brch import PF, PT, QF, QT
from .idx_bus import BUS_TYPE, PQ, VA, VM
from .idx_gen import GEN_BUS, GEN_STATUS, PG, PMAX, QG
from .int2ext import int2ext
from .loadcase import loadcase
from .makeJac import makeJac, makeJac_matrix
from .makeSbus import makeSbus, makeSbus_dV, makeSbus_value
from .makeYbus import makeYbus, makeYbus_full
from .mpoption import mpoption
from .mpver import mpver
from .printpf import printpf
from .runpf import runpf
from .savecase import savecase


def _event_list(value):
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [value.flat[k] for k in range(value.size)]
    return [value]


def _isempty(value):
    if value is None:
        return True
    arr = np.asarray(value)
    return arr.size == 0


def _capture_printpf(results, mpopt_value):
    buf = io.StringIO()
    with redirect_stdout(buf):
        printpf(results, 1, mpopt_value, nargout=0)
    return buf.getvalue()


def _normalize_stop_at(stop_at):
    if isinstance(stop_at, np.ndarray) and stop_at.size == 1:
        stop_at = stop_at.reshape(-1)[0].item()
    return stop_at


def _get_off_status(results, table):
    order = results.get("order", {})
    status = order.get(table, {}).get("status", {})
    return np.asarray(status.get("off", np.array([]))).reshape(-1).astype(int)


def _min_real_eig_sr(J, nb):
    opts_tol = 1e-3
    opts_it = 2 * nb
    if sparse.issparse(J):
        vals = spla.eigs(J, k=1, which="SR", tol=opts_tol, maxiter=opts_it, return_eigenvectors=False)
        return np.real(vals[0])
    vals = np.linalg.eigvals(np.asarray(J))
    return np.min(np.real(vals))


def runcpf(basecasedata=None, targetcasedata=None, mpopt_value=None, fname="", solvedcase="", *, nargout=None):
    """Run a full AC continuation power flow.

    Parameters
    ----------
    basecasedata : dict or str, optional
        MATPOWER case dict or case-file name defining the base loading and
        generation.
    targetcasedata : dict or str, optional
        MATPOWER case dict or case-file name defining the target loading
        and generation.
    mpopt_value : dict, optional
        MATPOWER options dict controlling parameterization, adaptive step
        sizing, limits enforcement, and output behavior.
    fname : str, optional
        File name to which pretty-printed output is appended.
    solvedcase : str, optional
        File name where the solved case is saved in MATPOWER case format.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    dict or tuple
        With one or two outputs, returns the CPF results dict and optional
        success flag.
    """
    if basecasedata is None and targetcasedata is None:
        raise TypeError("runcpf: Python port currently requires base and target MATPOWER case structs")
    if targetcasedata is None:
        raise TypeError(
            "runcpf: Two input case files, a base and target case with different load/generation patterns are required for RUNCPF."
        )
    if not isinstance(basecasedata, dict) or not isinstance(targetcasedata, dict):
        raise TypeError("runcpf: Python port currently supports MATPOWER case structs only")

    if mpopt_value is None:
        mpopt_value = mpoption()
    else:
        mpopt_value = mpoption(mpopt_value)

    fname = "" if fname is None else str(fname)
    solvedcase = "" if solvedcase is None else str(solvedcase)

    step = mpopt_value.cpf.step
    parm = mpopt_value.cpf.parameterization
    adapt_step = mpopt_value.cpf.adapt_step
    qlim = mpopt_value.cpf.enforce_q_lims
    plim = mpopt_value.cpf.enforce_p_lims
    vlim = mpopt_value.cpf.enforce_v_lims
    flim = mpopt_value.cpf.enforce_flow_lims

    cpf_events = []
    cpf_callbacks = []
    stop_at = _normalize_stop_at(mpopt_value.cpf.stop_at)
    if isinstance(stop_at, str) and stop_at == "NOSE":
        cpf_events = cpf_register_event(cpf_events, "NOSE", "cpf_nose_event", mpopt_value.cpf.nose_tol, 1, nargout=1)
        cpf_callbacks = cpf_register_callback(cpf_callbacks, "cpf_nose_event_cb", 51, nargout=1)
    else:
        cpf_events = cpf_register_event(
            cpf_events,
            "TARGET_LAM",
            "cpf_target_lam_event",
            mpopt_value.cpf.target_lam_tol,
            1,
            nargout=1,
        )
        cpf_callbacks = cpf_register_callback(cpf_callbacks, "cpf_target_lam_event_cb", 50, nargout=1)
    if flim:
        cpf_events = cpf_register_event(
            cpf_events, "FLIM", "cpf_flim_event", mpopt_value.cpf.flow_lims_tol, 1, nargout=1
        )
        cpf_callbacks = cpf_register_callback(cpf_callbacks, "cpf_flim_event_cb", 53, nargout=1)
    if vlim:
        cpf_events = cpf_register_event(cpf_events, "VLIM", "cpf_vlim_event", mpopt_value.cpf.v_lims_tol, 1, nargout=1)
        cpf_callbacks = cpf_register_callback(cpf_callbacks, "cpf_vlim_event_cb", 52, nargout=1)
    if qlim:
        cpf_events = cpf_register_event(cpf_events, "QLIM", "cpf_qlim_event", mpopt_value.cpf.q_lims_tol, 1, nargout=1)
        cpf_callbacks = cpf_register_callback(cpf_callbacks, "cpf_qlim_event_cb", 41, nargout=1)
    if plim:
        cpf_events = cpf_register_event(cpf_events, "PLIM", "cpf_plim_event", mpopt_value.cpf.p_lims_tol, 1, nargout=1)
        cpf_callbacks = cpf_register_callback(cpf_callbacks, "cpf_plim_event_cb", 40, nargout=1)
    cpf_callbacks = cpf_register_callback(cpf_callbacks, "cpf_default_callback", 0, nargout=1)

    user_callbacks = mpopt_value.cpf.user_callback
    if not _isempty(user_callbacks):
        if isinstance(user_callbacks, list):
            callbacks = user_callbacks
        elif isinstance(user_callbacks, np.ndarray) and user_callbacks.dtype == object:
            callbacks = [user_callbacks.flat[k] for k in range(user_callbacks.size)]
        else:
            callbacks = [user_callbacks]
        for user_callback in callbacks:
            if isinstance(user_callback, dict):
                ucb = user_callback
            else:
                ucb = {"fcn": user_callback}
            if "priority" not in ucb:
                ucb["priority"] = []
            if "args" not in ucb:
                ucb["args"] = []
            cpf_callbacks = cpf_register_callback(cpf_callbacks, ucb["fcn"], ucb["priority"], ucb["args"], nargout=1)

    cpf_events = _event_list(cpf_events)
    cpf_callbacks = _event_list(cpf_callbacks)
    nef = len(cpf_events)
    ncb = len(cpf_callbacks)

    if mpopt_value.verbose > 4:
        mpopt_pf = mpoption(mpopt_value, "verbose", 2)
    else:
        mpopt_pf = mpoption(mpopt_value, "verbose", 0)
    mpopt_pf = mpoption(mpopt_pf, "pf.enforce_q_lims", qlim)

    mpcb = loadcase(basecasedata, nargout=1)
    idx_pmax = np.array([], dtype=int)
    if plim:
        idx_pmax = np.flatnonzero(
            (mpcb["gen"][:, GEN_STATUS - 1] > 0)
            & (mpcb["gen"][:, PG - 1] - mpcb["gen"][:, PMAX - 1] > -mpopt_value.cpf.p_lims_tol)
        )
        mpcb["gen"][idx_pmax, PG - 1] = mpcb["gen"][idx_pmax, PMAX - 1]

    rb, success = runpf(mpcb, mpopt_pf, nargout=2)
    if success:
        done = {"flag": 0, "msg": ""}
        mpcb = rb
    else:
        done = {"flag": 1, "msg": "Base case power flow did not converge."}
        results = rb
        results["cpf"] = {}

    if done["flag"] == 0:
        mpct = loadcase(targetcasedata, nargout=1)
        if mpct["branch"].shape[1] < QT:
            mpct["branch"] = np.concatenate(
                [mpct["branch"], np.zeros((mpct["branch"].shape[0], QT - mpct["branch"].shape[1]))], axis=1
            )

        mpcb_e = copy.deepcopy(mpcb)
        mpcb_e.pop("order", None)
        mpcb = ext2int(mpcb_e, mpopt_value)
        mpct = ext2int(mpct, mpopt_value)
        nb = mpcb["bus"].shape[0]

        ref, pv, pq = bustypes(mpcb["bus"], mpcb["gen"])
        ong = np.flatnonzero(
            (mpcb["gen"][:, GEN_STATUS - 1] > 0)
            & (mpcb["bus"][mpcb["gen"][:, GEN_BUS - 1].astype(int) - 1, BUS_TYPE - 1] != PQ)
        )
        gbus = mpcb["gen"][ong, GEN_BUS - 1].astype(int)

        if np.any(mpcb["bus"][:, BUS_TYPE - 1] != mpct["bus"][:, BUS_TYPE - 1]):
            raise ValueError("runcpf: BUS_TYPE of all buses must be the same in base and target cases")
        if np.any(mpcb["gen"][:, GEN_STATUS - 1] != mpct["gen"][:, GEN_STATUS - 1]):
            raise ValueError("runcpf: GEN_STATUS of all generators must be the same in base and target cases")
        mpct["gen"][ong, QG - 1] = mpcb["gen"][ong, QG - 1]
        for k in np.asarray(ref, dtype=int).reshape(-1):
            refgen = np.flatnonzero(gbus == k)
            mpct["gen"][ong[refgen], PG - 1] = mpcb["gen"][ong[refgen], PG - 1]

        t0 = time.perf_counter()
        verbose = mpopt_value.verbose
        if verbose:
            v = mpver("all", nargout=1)
            print(f"\npwrs Version {v['Version']}, {v['Date']} -- AC Continuation Power Flow")
            if verbose > 1:
                print(
                    f"step {0:3d}  :                      lambda = {0:6.3f}, {mpcb.get('iterations', 0):2d} Newton steps"
                )

        Ybus, Yf, Yt = makeYbus_full(mpcb["baseMVA"], mpcb["bus"], mpcb["branch"])
        Sbusb = lambda Vm: (
            makeSbus_value(mpcb["baseMVA"], mpcb["bus"], mpcb["gen"], mpopt_value, Vm),
            makeSbus_dV(mpcb["baseMVA"], mpcb["bus"], mpcb["gen"], mpopt_value, Vm)[1],
        )
        Sbust = lambda Vm: (
            makeSbus_value(mpct["baseMVA"], mpct["bus"], mpct["gen"], mpopt_value, Vm),
            makeSbus_dV(mpct["baseMVA"], mpct["bus"], mpct["gen"], mpopt_value, Vm)[1],
        )

        cont_steps = 0
        lam = 0.0
        V = mpcb["bus"][:, VM - 1] * np.exp(1j * np.pi / 180.0 * mpcb["bus"][:, VA - 1])
        rollback = 0
        locating = 0
        rb_cnt_ef = 0
        rb_cnt_cb = 0
        z = np.r_[np.zeros(2 * nb), 1.0].reshape(-1, 1)
        direction = 1.0
        z = cpf_tangent(V, lam, Ybus, Sbusb, Sbust, pv, pq, z, V, lam, parm, direction)

        cx = {
            "lam_hat": lam,
            "V_hat": V.reshape(-1, 1),
            "lam": lam,
            "V": V.reshape(-1, 1),
            "z": z,
            "default_step": step,
            "default_parm": parm,
            "this_step": np.array([]),
            "this_parm": np.array([]),
            "step": step,
            "parm": parm,
            "events": [],
            "cb": {},
            "ef": [None] * nef,
        }

        cb_data = {
            "mpc_base": mpcb,
            "mpc_target": mpct,
            "Sbusb": Sbusb,
            "Sbust": Sbust,
            "Ybus": Ybus,
            "Yf": Yf,
            "Yt": Yt,
            "ref": ref,
            "pv": pv,
            "pq": pq,
            "idx_pmax": idx_pmax,
            "mpopt": mpopt_value,
        }

        for k in range(nef):
            cx["ef"][k] = cpf_events[k]["fcn"](cb_data, cx)

        nx = cx
        evnts = []
        for k in range(ncb):
            nx, cx, done, rollback, evnts, cb_data, _ = cpf_callbacks[k]["fcn"](
                cont_steps, cx, cx, cx, done, 0, [], cb_data, cpf_callbacks[k]["args"], {}
            )

        if (
            np.linalg.norm(
                np.asarray(Sbust(np.abs(np.asarray(cx["V"]).reshape(-1)))[0])
                - np.asarray(Sbusb(np.abs(np.asarray(cx["V"]).reshape(-1)))[0])
            )
            == 0
        ):
            done["flag"] = 1
            done["msg"] = "Base case and target case have identical load and generation"

        cont_steps += 1
        px = copy.deepcopy(cx)
        rx = None
        while not done["flag"]:
            nx = copy.deepcopy(cx)

            nx["V_hat"], nx["lam_hat"] = cpf_predictor(
                cx["V"], cx["lam"], cx["z"], cx["step"], cb_data["pv"], cb_data["pq"]
            )
            nx["V"], success, iters, nx["lam"] = cpf_corrector(
                Ybus,
                cb_data["Sbusb"],
                nx["V_hat"],
                cb_data["ref"],
                cb_data["pv"],
                cb_data["pq"],
                nx["lam_hat"],
                cb_data["Sbust"],
                cx["V"],
                cx["lam"],
                cx["z"],
                cx["step"],
                cx["parm"],
                mpopt_pf,
            )
            if not success:
                done["flag"] = 1
                done["msg"] = f"Corrector did not converge in {int(iters)} iterations."
                if verbose:
                    print(
                        f"step {cont_steps:3d}  : stepsize = {cx['step']:<-9.3g} lambda = {float(nx['lam']):6.3f}  corrector did not converge in {int(iters)} iterations"
                    )
                cont_steps = max(cont_steps - 1, 1)
                break

            tx = px if float(nx["step"]) == 0 else cx
            nx["z"] = cpf_tangent(
                nx["V"],
                nx["lam"],
                Ybus,
                cb_data["Sbusb"],
                cb_data["Sbust"],
                cb_data["pv"],
                cb_data["pq"],
                cx["z"],
                tx["V"],
                tx["lam"],
                nx["parm"],
                direction,
            )

            for k in range(nef):
                nx["ef"][k] = cpf_events[k]["fcn"](cb_data, nx)
            rollback, evnts, nx["ef"] = cpf_detect_events(
                cpf_events, nx["ef"], cx["ef"], nx["step"], verbose, nargout=3
            )
            evnts_list = _event_list(evnts)

            if rollback:
                rx = copy.deepcopy(nx)
                rx["evnts"] = copy.deepcopy(evnts)
                cx["this_step"] = float(evnts_list[0]["step_scale"]) * float(rx["step"])
                cx["this_parm"] = rx["parm"]
                locating = 1
                rb_cnt_ef += 1
                if rb_cnt_ef > 26:
                    done["flag"] = 1
                    done["msg"] = f"Could not locate {evnts_list[0]['name']} event!"
            elif locating:
                if evnts_list[0]["zero"]:
                    locating = 0
                    rb_cnt_ef = 0
                else:
                    rx_ef = np.asarray(rx["ef"][int(rx["evnts"]["eidx"]) - 1]).reshape(-1)[
                        int(np.asarray(rx["evnts"]["idx"]).reshape(-1)[0]) - 1
                    ]
                    cx_ef = np.asarray(nx["ef"][int(rx["evnts"]["eidx"]) - 1]).reshape(-1)[
                        int(np.asarray(rx["evnts"]["idx"]).reshape(-1)[0]) - 1
                    ]
                    step_scale = cx_ef / (cx_ef - rx_ef)
                    nx["this_step"] = float(step_scale) * (float(rx["step"]) - float(nx["step"]))
                    rb_cnt_ef = 0
            else:
                direction = 1

            rb = rollback
            for k in range(ncb):
                nx, cx, done, rollback, evnts, cb_data, _ = cpf_callbacks[k]["fcn"](
                    cont_steps, nx, cx, px, done, rollback, evnts, cb_data, cpf_callbacks[k]["args"], {}
                )
            evnts_list = _event_list(evnts)

            if (not rb) and rollback:
                rb_cnt_cb += 1
                if rb_cnt_cb > 26:
                    done["flag"] = 1
                    done["msg"] = "Too many rollback steps triggered by callbacks!"
            else:
                if (not done["flag"]) and evnts_list[0]["zero"]:
                    mpce = cpf_current_mpc(
                        cb_data["mpc_base"],
                        cb_data["mpc_target"],
                        Ybus,
                        Yf,
                        Yt,
                        cb_data["ref"],
                        cb_data["pv"],
                        cb_data["pq"],
                        nx["V"],
                        nx["lam"],
                        mpopt_value,
                    )
                    J = makeJac_matrix(mpce)
                    pq_arr = np.asarray(pq, dtype=int).reshape(-1)
                    if not np.any(np.asarray(nx["z"]).reshape(-1)[nb + pq_arr - 1] > 0):
                        direction = np.sign(float(np.asarray(nx["z"]).reshape(-1)[-1]) * float(_min_real_eig_sr(J, nb)))
                rb_cnt_cb = 0

            for ev in evnts_list:
                if ev["log"]:
                    e = {"k": cont_steps, "name": ev["name"], "idx": ev["idx"], "msg": ev["msg"]}
                    if not nx["events"]:
                        nx["events"] = [e]
                    else:
                        nx["events"].append(e)

            if (
                adapt_step
                and (not done["flag"])
                and (not locating)
                and (not evnts_list[0]["zero"])
                and float(nx["step"]) != 0
            ):
                pq_arr = np.asarray(cb_data["pq"], dtype=int).reshape(-1)
                pvpq = np.r_[np.asarray(cb_data["pv"], dtype=int).reshape(-1), pq_arr]
                cpf_error = np.linalg.norm(
                    np.r_[
                        np.angle(np.asarray(nx["V"]).reshape(-1)[pq_arr - 1]),
                        np.abs(np.asarray(nx["V"]).reshape(-1)[pvpq - 1]),
                        float(nx["lam"]),
                    ]
                    - np.r_[
                        np.angle(np.asarray(nx["V_hat"]).reshape(-1)[pq_arr - 1]),
                        np.abs(np.asarray(nx["V_hat"]).reshape(-1)[pvpq - 1]),
                        float(nx["lam_hat"]),
                    ],
                    np.inf,
                )
                step_scale = min(
                    2,
                    1 + mpopt_value.cpf.adapt_step_damping * (mpopt_value.cpf.adapt_step_tol / cpf_error - 1),
                )
                nx["default_step"] = float(nx["step"]) * step_scale
                nx["default_step"] = min(nx["default_step"], mpopt_value.cpf.step_max)
                nx["default_step"] = max(nx["default_step"], mpopt_value.cpf.step_min)

            if not rollback:
                px = copy.deepcopy(cx)
                cx = copy.deepcopy(nx)
                if not done["flag"]:
                    cont_steps += 1

            if _isempty(cx["this_step"]):
                cx["step"] = cx["default_step"]
            else:
                cx["step"] = cx["this_step"]
                cx["this_step"] = np.array([])
            if _isempty(cx["this_parm"]):
                cx["parm"] = cx["default_parm"]
            else:
                cx["parm"] = cx["this_parm"]
                cx["this_parm"] = np.array([])

        cpf_results = {}
        for k in range(ncb):
            nx, cx, done, rollback, evnts, cb_data, cpf_results = cpf_callbacks[k]["fcn"](
                -cont_steps,
                nx,
                cx,
                px,
                done,
                rollback,
                evnts,
                cb_data,
                cpf_callbacks[k]["args"],
                cpf_results,
            )
        cpf_results["events"] = cx["events"]

        mpct = cpf_current_mpc(
            cb_data["mpc_base"],
            cb_data["mpc_target"],
            Ybus,
            Yf,
            Yt,
            cb_data["ref"],
            cb_data["pv"],
            cb_data["pq"],
            cx["V"],
            cx["lam"],
            mpopt_value,
        )
        mpct["et"] = time.perf_counter() - t0
        mpct["success"] = success

        n = np.asarray(cpf_results["V"]).shape[1]
        cpf_results["V_hat"] = i2e_data(
            mpct, cpf_results["V_hat"], np.full((nb, n), np.nan, dtype=complex), "bus", 1, nargout=1
        )
        cpf_results["V"] = i2e_data(
            mpct, cpf_results["V"], np.full((nb, n), np.nan, dtype=complex), "bus", 1, nargout=1
        )
        results = int2ext(mpct, nargout=1)
        results["cpf"] = cpf_results

        off_gen = _get_off_status(results, "gen")
        if off_gen.size:
            results["gen"][np.ix_(off_gen - 1, [PG - 1, QG - 1])] = 0
        off_branch = _get_off_status(results, "branch")
        if off_branch.size:
            results["branch"][np.ix_(off_branch - 1, [PF - 1, QF - 1, PT - 1, QT - 1])] = 0
    results["cpf"]["done_msg"] = done["msg"]

    if mpopt_value.verbose:
        print(f"CPF TERMINATION: {done['msg']}")

    if fname:
        if mpopt_value.out.all == 0:
            text = _capture_printpf(results, mpoption(mpopt_value, "out.all", -1))
        else:
            text = _capture_printpf(results, mpopt_value)
        with Path(fname).open("a", encoding="utf-8") as fh:
            fh.write(text)
    printpf(results, 1, mpopt_value, nargout=0)

    if solvedcase:
        savecase(solvedcase, results, nargout=0)

    if nargout in (None, 1):
        return results
    if nargout == 2:
        return results, float(results["success"])
    return results
