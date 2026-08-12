# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from ..corex import MatpowerConfig
from .idx_bus import VM, VMAX, VMIN
from .mpoption import mpoption


def cpf_vlim_event_cb(k, nx, cx, px, done, rollback, evnts, cb_data, cb_args, results=None):
    """Handle CPF bus voltage magnitude limit events.

    Implements MATPOWER's callback logic for ``VLIM`` events, including base
    case validation and CPF termination when a bus voltage magnitude limit is
    reached during continuation.

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
    mpopt = cb_data["mpopt"]
    if not isinstance(mpopt, MatpowerConfig):
        mpopt = mpoption(mpopt)
    mpc = cb_data["mpc_base"]
    nb = mpc["bus"].shape[0]
    i2e_bus = np.asarray(mpc["order"]["bus"]["i2e"]).reshape(-1)

    if int(np.asarray(k).reshape(-1)[0]) == 0:
        if np.any(mpc["bus"][:, VM - 1] < mpc["bus"][:, VMIN - 1]) or np.any(
            mpc["bus"][:, VM - 1] > mpc["bus"][:, VMAX - 1]
        ):
            ib = np.flatnonzero(
                np.r_[mpc["bus"][:, VM - 1] < mpc["bus"][:, VMIN - 1], mpc["bus"][:, VM - 1] > mpc["bus"][:, VMAX - 1]]
            )
            msg = ""
            for b in ib:
                if b >= nb:
                    bb = b - nb
                    msg = f"bus voltage magnitude limit violated in base case: bus {int(i2e_bus[bb])} exceeds Vmax limit {mpc['bus'][bb, VMAX - 1]:g} p.u."
                else:
                    msg = f"bus voltage magnitude limit violated in base case: bus {int(i2e_bus[b])} exceeds Vmin limit {mpc['bus'][b, VMIN - 1]:g} p.u."
            done["flag"] = 1
            done["msg"] = msg

    if int(np.asarray(k).reshape(-1)[0]) < 0 or done["flag"]:
        return nx, cx, done, rollback, evnts, cb_data, results

    evnts_list = [evnts] if isinstance(evnts, dict) else list(evnts)
    for ev in evnts_list:
        if ev["name"] == "VLIM" and ev["zero"]:
            if mpopt.verbose > 3:
                msg = f"{ev['msg']}\n    "
            else:
                msg = ""
            ib = np.asarray(ev["idx"], dtype=int).reshape(-1)
            for b in ib:
                if b > nb:
                    bb = b - nb
                    msg = (
                        f"{msg}bus voltage magnitude limit reached\nbus {int(i2e_bus[bb - 1])} at VMAX limit "
                        f"{mpc['bus'][bb - 1, VMAX - 1]:g} p.u. @ lambda = {float(np.asarray(nx['lam']).reshape(-1)[0]):.4g}, "
                        f"in {int(np.asarray(k).reshape(-1)[0])} continuation steps"
                    )
                else:
                    msg = (
                        f"{msg}bus voltage magnitude limit reached\nbus {int(i2e_bus[b - 1])} at Vmin limit "
                        f"{mpc['bus'][b - 1, VMIN - 1]:g} p.u. @ lambda = {float(np.asarray(nx['lam']).reshape(-1)[0]):.4g}, "
                        f"in {int(np.asarray(k).reshape(-1)[0])} continuation steps"
                    )
            done["flag"] = 1
            done["msg"] = msg

    return nx, cx, done, rollback, evnts, cb_data, results
