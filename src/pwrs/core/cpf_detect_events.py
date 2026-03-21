# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy

import numpy as np


def _event_list(cpf_events):
    if isinstance(cpf_events, dict):
        return [cpf_events]
    return list(cpf_events)


def _value_list(values):
    if isinstance(values, np.ndarray) and values.dtype == object:
        return [values.flat[k] for k in range(values.size)]
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]


def _column_indices(idx):
    return np.asarray(idx, dtype=np.int64).reshape(-1, 1)


def _new_event():
    return {
        "eidx": 0,
        "name": "",
        "zero": 0,
        "idx": 0,
        "step_scale": 1,
        "log": 0,
        "msg": "",
    }


def cpf_detect_events(cpf_events, cef, pef, step, verbose, *, nargout=None):
    """Detect continuation power flow events from event-function values.

    Parameters
    ----------
    cpf_events : sequence or dict
        Registered CPF event definitions.
    cef : sequence
        Current event-function values.
    pef : sequence
        Previous event-function values.
    step : float
        Current step size.
    verbose : int
        Verbosity level.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    tuple
        ``(rollback, evnts, cef)`` containing the rollback flag, detected
        event information, and the possibly adjusted current event-function
        values.
    """
    rollback = 0
    evnts = _new_event()
    cef = copy.deepcopy(_value_list(cef))
    pef = _value_list(pef)

    i = 0
    cpf_events = _event_list(cpf_events)
    nef = len(cef)

    for eidx in range(nef):
        if not cpf_events[eidx]["locate"]:
            continue

        c_sign = np.sign(np.asarray(cef[eidx]).reshape(-1))
        p_sign = np.sign(np.asarray(pef[eidx]).reshape(-1))

        idx = np.flatnonzero(
            (np.abs(c_sign) == 1)
            & (c_sign == -p_sign)
            & (np.abs(np.asarray(cef[eidx]).reshape(-1)) > float(cpf_events[eidx]["tol"]))
        )
        if idx.size:
            if float(np.asarray(step).reshape(-1)[0]) == 0:
                evnts["eidx"] = eidx + 1
                evnts["name"] = cpf_events[eidx]["name"]
                evnts["zero"] = 1
                evnts["idx"] = _column_indices(idx + 1)
                evnts["step_scale"] = 1
                evnts["log"] = 1
                evnts["msg"] = "ZERO (BIFURCATION)"
                i += 1
                break
            step_scales = np.asarray(pef[eidx]).reshape(-1)[idx] / (
                np.asarray(pef[eidx]).reshape(-1)[idx] - np.asarray(cef[eidx]).reshape(-1)[idx]
            )
            j = int(np.argmin(step_scales))
            step_scale = float(step_scales[j])
            if step_scale < float(evnts["step_scale"]):
                evnts["eidx"] = eidx + 1
                evnts["name"] = cpf_events[eidx]["name"]
                evnts["zero"] = 0
                evnts["idx"] = int(idx[j] + 1)
                evnts["step_scale"] = step_scale
                evnts["log"] = 0
                evnts["msg"] = "INTERVAL"
                rollback = 1

    if rollback == 0:
        if isinstance(evnts, dict):
            evnts_list = [evnts]
        else:
            evnts_list = list(evnts)
        for eidx in range(nef):
            idx = np.flatnonzero(np.abs(np.asarray(cef[eidx]).reshape(-1)) <= float(cpf_events[eidx]["tol"]))
            if idx.size:
                cur = np.asarray(cef[eidx]).copy().reshape(-1)
                cur[idx] = 0
                cef[eidx] = cur.reshape(np.asarray(cef[eidx]).shape)

                ev = _new_event()
                ev["eidx"] = eidx + 1
                ev["name"] = cpf_events[eidx]["name"]
                ev["zero"] = 1
                ev["idx"] = _column_indices(idx + 1)
                ev["step_scale"] = 1
                ev["log"] = 1
                ev["msg"] = "ZERO"
                if i == 0 and evnts_list and evnts_list[0]["eidx"] == 0:
                    evnts_list[0] = ev
                else:
                    evnts_list.append(ev)
                i += 1

        if i == 0:
            for eidx in range(nef):
                c_sign = np.sign(np.asarray(cef[eidx]).reshape(-1))
                p_sign = np.sign(np.asarray(pef[eidx]).reshape(-1))
                idx = np.flatnonzero((np.abs(c_sign) == 1) & (c_sign == -p_sign))
                if idx.size:
                    step_scale = np.asarray(pef[eidx]).reshape(-1)[idx] / (
                        np.asarray(pef[eidx]).reshape(-1)[idx] - np.asarray(cef[eidx]).reshape(-1)[idx]
                    )
                    ev = _new_event()
                    ev["eidx"] = eidx + 1
                    ev["name"] = cpf_events[eidx]["name"]
                    ev["zero"] = 0
                    ev["idx"] = _column_indices(idx + 1)
                    ev["step_scale"] = step_scale.reshape(-1, 1)
                    ev["log"] = 0
                    ev["msg"] = "INTERVAL"
                    if i == 0 and evnts_list and evnts_list[0]["eidx"] == 0:
                        evnts_list[0] = ev
                    else:
                        evnts_list.append(ev)
                    i += 1
        evnts = evnts_list if len(evnts_list) > 1 else evnts_list[0]

    evnts_list = [evnts] if isinstance(evnts, dict) else evnts
    if evnts_list and evnts_list[0]["eidx"]:
        for ev in evnts_list:
            if np.asarray(cef[int(ev["eidx"]) - 1]).size > 1:
                idxv = np.asarray(ev["idx"]).reshape(-1)
                s1 = f"({int(idxv[0])})" if idxv.size == 1 else f"({','.join(str(int(v)) for v in idxv)})"
            else:
                s1 = ""
            if rollback:
                s2 = f"ROLLBACK by {float(np.asarray(ev['step_scale']).reshape(-1)[0]):g}"
            else:
                s2 = "CONTINUE"
            ev["msg"] = f"{ev['msg']} detected for {ev['name']}{s1} event : {s2}"

    if nargout == 1:
        return rollback
    if nargout == 2:
        return rollback, evnts
    return rollback, evnts, cef
