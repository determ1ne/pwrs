# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from ..corex import MatpowerConfig
from .mpoption import mpoption


def _get_cb_mpopt(cb_data):
    mpopt = cb_data["mpopt"]
    if isinstance(mpopt, MatpowerConfig):
        return mpopt
    return mpoption(mpopt)


def cpf_nose_event_cb(k, nx, cx, px, done, rollback, evnts, cb_data, cb_args, results=None):
    """Handle CPF nose-point events.

    Implements the MATPOWER callback logic for the ``NOSE`` event, updating
    the event messages and marking the CPF run complete when the configured
    stop mode requests termination at the steady-state loading limit.

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

    stop_at = _get_cb_mpopt(cb_data).cpf.stop_at

    if (not rollback) or float(np.asarray(nx["step"]).reshape(-1)[0]) == 0:
        evnts_list = [evnts] if isinstance(evnts, dict) else list(evnts)
        for ev in evnts_list:
            if ev["name"] == "NOSE" and ev["zero"]:
                if float(np.asarray(nx["step"]).reshape(-1)[0]) == 0:
                    ev["msg"] = (
                        f"Nose point eliminated by limit induced bifurcation at {int(k)} continuation steps, "
                        f"lambda = {float(np.asarray(nx['lam']).reshape(-1)[0]):.4g}."
                    )
                else:
                    ev["msg"] = (
                        f"Reached steady state loading limit in {int(k)} continuation steps, "
                        f"lambda = {float(np.asarray(nx['lam']).reshape(-1)[0]):.4g}."
                    )
                if isinstance(stop_at, str) and stop_at == "NOSE":
                    done["flag"] = 1
                    done["msg"] = ev["msg"]
                evnts = evnts_list if len(evnts_list) > 1 else evnts_list[0]
                break

    return nx, cx, done, rollback, evnts, cb_data, results
