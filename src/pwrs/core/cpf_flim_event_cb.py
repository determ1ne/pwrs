# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .idx_brch import F_BUS, PF, PT, QF, QT, RATE_A, T_BUS


def cpf_flim_event_cb(k, nx, cx, px, done, rollback, evnts, cb_data, cb_args, results=None, *, nargout=None):
    """Handle CPF branch flow limit events.

    Implements MATPOWER's callback logic for ``FLIM`` events, including base
    case validation and CPF termination when a branch MVA limit is reached
    during continuation.

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
    mpc = cb_data["mpc_base"]
    i2e_bus = np.asarray(mpc["order"]["bus"]["i2e"]).reshape(-1)
    f = mpc["branch"][:, F_BUS - 1].astype(int)
    t = mpc["branch"][:, T_BUS - 1].astype(int)
    srate_a = mpc["branch"][:, RATE_A - 1]

    if int(np.asarray(k).reshape(-1)[0]) == 0:
        sf = np.sqrt(mpc["branch"][:, PF - 1] ** 2 + mpc["branch"][:, QF - 1] ** 2)
        st = np.sqrt(mpc["branch"][:, PT - 1] ** 2 + mpc["branch"][:, QT - 1] ** 2)
        idx = np.flatnonzero(np.maximum(sf, st) > srate_a)
        if idx.size:
            msg = ""
            for L in idx:
                msg = (
                    f"branch flow limit violated in base case: branch {int(i2e_bus[f[L] - 1])} -- "
                    f"{int(i2e_bus[t[L] - 1])} exceeds limit of {srate_a[L]:g} MVA\n"
                )
            done["flag"] = 1
            done["msg"] = msg

    if int(np.asarray(k).reshape(-1)[0]) < 0 or done["flag"]:
        return nx, cx, done, rollback, evnts, cb_data, results

    evnts_list = [evnts] if isinstance(evnts, dict) else list(evnts)
    for ev in evnts_list:
        if ev["name"] == "FLIM" and ev["zero"]:
            if cb_data["mpopt"]["verbose"] > 3:
                msg = f"{ev['msg']}\n    "
            else:
                msg = ""
            idx = np.asarray(ev["idx"], dtype=int).reshape(-1)
            for L in idx:
                LL = L - 1
                msg = (
                    f"{msg}branch flow limit reached\nbranch {int(i2e_bus[f[LL] - 1])} -- {int(i2e_bus[t[LL] - 1])} "
                    f"at limit of {srate_a[LL]:g} MVA @ lambda = {float(np.asarray(nx['lam']).reshape(-1)[0]):.4g}, "
                    f"in {int(np.asarray(k).reshape(-1)[0])} continuation steps"
                )
            done["flag"] = 1
            done["msg"] = msg

    return nx, cx, done, rollback, evnts, cb_data, results
