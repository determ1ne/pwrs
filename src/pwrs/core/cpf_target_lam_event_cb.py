# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from ..corex import MatpowerConfig
from .cpf_predictor import cpf_predictor
from .mpoption import mpoption


def _get_cb_mpopt(cb_data):
    mpopt = cb_data["mpopt"]
    if isinstance(mpopt, MatpowerConfig):
        return mpopt
    return mpoption(mpopt)


def cpf_target_lam_event_cb(k, nx, cx, px, done, rollback, evnts, cb_data, cb_args, results=None):
    """Handle CPF target-lambda events and step adjustments.

    Implements the MATPOWER callback logic for the ``TARGET_LAM`` event. It
    marks CPF completion when the target is reached exactly and otherwise
    adjusts the next predictor step to avoid overshooting the configured
    lambda target or full-curve stopping point.

    Parameters
    ----------
    k : int
        Current continuation step index.
    nx : dict
        Next-step CPF state struct.
    cx : dict
        Current-step CPF state struct.
    px : dict
        Previous-step CPF state struct.
    done : dict
        CPF completion flag/message struct.
    rollback : bool
        Whether the current step is being rolled back.
    evnts : dict or list of dict
        Detected CPF events for the current step.
    cb_data : dict
        CPF callback data struct.
    cb_args : dict
        Callback arguments struct.
    results : dict, optional
        CPF results accumulator.

    Returns
    -------
    tuple
        Updated ``(nx, cx, done, rollback, evnts, cb_data, results)``.
    """
    if int(np.asarray(k).reshape(-1)[0]) <= 0 or done["flag"]:
        return nx, cx, done, rollback, evnts, cb_data, results

    mpopt = _get_cb_mpopt(cb_data)
    stop_at = mpopt.cpf.stop_at
    verbose = float(mpopt.verbose)
    if isinstance(stop_at, np.ndarray) and stop_at.size == 1:
        stop_at = stop_at.reshape(-1)[0].item()
    if isinstance(stop_at, str):
        if stop_at == "FULL":
            stop_at = 0
        else:
            stop_at = -np.inf

    event_detected = 0
    evnts_list = [evnts] if isinstance(evnts, dict) else list(evnts)
    for i, ev in enumerate(evnts_list):
        if ev["name"] == "TARGET_LAM":
            event_detected = 1
            if ev["zero"]:
                done["flag"] = 1
                if stop_at == 0:
                    done["msg"] = f"Traced full continuation curve in {int(k)} continuation steps"
                else:
                    done["msg"] = f"Reached desired lambda {float(stop_at):g} in {int(k)} continuation steps"
            else:
                if stop_at == 0:
                    ev["msg"] = (
                        f"{ev['msg']}\n  step {int(k)} expected to overshoot full trace, "
                        "reduce step size and set natural param"
                    )
                else:
                    ev["msg"] = (
                        f"{ev['msg']}\n  step {int(k)} expected to overshoot target lambda, "
                        "reduce step size and set natural param"
                    )
                ev["log"] = 1
                if stop_at == 0:
                    cx["this_step"] = float(np.asarray(cx["lam"]).reshape(-1)[0])
                else:
                    cx["this_step"] = float(stop_at) - float(np.asarray(cx["lam"]).reshape(-1)[0])
                cx["this_parm"] = 1
            evnts = evnts_list if len(evnts_list) > 1 else evnts_list[0]
            break

    if not event_detected and not rollback:
        step = nx["this_step"] if not _isempty(nx.get("this_step")) else nx["default_step"]
        _, lam_hat = cpf_predictor(nx["V"], nx["lam"], nx["z"], step, cb_data["pv"], cb_data["pq"])
        if stop_at == 0:
            if lam_hat < -float(np.asarray(nx["lam"]).reshape(-1)[0]):
                nx["this_step"] = float(np.asarray(nx["lam"]).reshape(-1)[0])
                nx["this_parm"] = 1
                if verbose > 2:
                    print(
                        f"  step {int(k) + 1} expected to overshoot full trace, reduce step size and set natural param"
                    )
        elif stop_at > 0:
            if lam_hat > float(stop_at) + (float(stop_at) - float(np.asarray(nx["lam"]).reshape(-1)[0])):
                nx["this_step"] = float(stop_at) - float(np.asarray(nx["lam"]).reshape(-1)[0])
                nx["this_parm"] = 1
                if verbose > 2:
                    print(
                        f"  step {int(k) + 1} expected to overshoot target lambda, reduce step size and set natural param"
                    )

    return nx, cx, done, rollback, evnts, cb_data, results


def _isempty(value):
    if value is None:
        return True
    arr = np.asarray(value)
    return arr.size == 0
