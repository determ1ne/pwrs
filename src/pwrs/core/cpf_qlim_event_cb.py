# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .bustypes import bustypes
from .cpf_current_mpc import cpf_current_mpc
from .idx_bus import BUS_TYPE, PQ, REF
from .idx_gen import GEN_BUS, GEN_STATUS, PG, QG, QMAX, QMIN
from .makeSbus import makeSbus


def cpf_qlim_event_cb(k, nx, cx, px, done, rollback, evnts, cb_data, cb_args, results=None, *, nargout=None):
    """Handle CPF reactive-power limit events.

    Implements MATPOWER's callback logic for ``QLIM`` events by fixing
    generators at their reactive limits, converting affected buses to PQ,
    updating the reference/PV/PQ partitions, and restarting the continuation
    step when necessary.

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

    mpc = []
    evnts_list = [evnts] if isinstance(evnts, dict) else list(evnts)

    for ev in evnts_list:
        if ev["name"] == "QLIM" and ev["zero"]:
            if mpc == []:
                d = cb_data
                ref = np.asarray(d["ref"], dtype=int).reshape(-1)
                if ref.size != 1:
                    raise ValueError(
                        "cpf_qlim_event_cb: 'cpf.enforce_qlims' option only valid for systems with exactly one REF bus"
                    )
                mpc = cpf_current_mpc(
                    d["mpc_base"],
                    d["mpc_target"],
                    d["Ybus"],
                    d["Yf"],
                    d["Yt"],
                    d["ref"],
                    d["pv"],
                    d["pq"],
                    nx["V"],
                    nx["lam"],
                    d["mpopt"],
                    nargout=1,
                )
                ng = mpc["gen"].shape[0]
                i2e_bus = np.asarray(mpc["order"]["bus"]["i2e"]).reshape(-1)
                i2e_gen = np.asarray(mpc["order"]["gen"]["i2e"]).reshape(-1)

            if cb_data["mpopt"]["verbose"] > 3:
                msg = f"{ev['msg']}\n    "
            else:
                msg = ""
            ig = np.asarray(ev["idx"], dtype=int).reshape(-1)
            for g in ig:
                maxlim = 1
                if g > ng:
                    g -= ng
                    maxlim = 0
                ib = int(mpc["gen"][g - 1, GEN_BUS - 1])
                if maxlim:
                    msg = (
                        f"{msg}gen {int(i2e_gen[g - 1])} @ bus {int(i2e_bus[ib - 1])} reached "
                        f"{mpc['gen'][g - 1, QMAX - 1]:g} MVAr Qmax lim @ lambda = {float(np.asarray(nx['lam']).reshape(-1)[0]):.4g} : "
                        f"bus {int(i2e_bus[ib - 1])} converted to PQ"
                    )
                    mpc["gen"][g - 1, QG - 1] = mpc["gen"][g - 1, QMAX - 1]
                else:
                    msg = (
                        f"{msg}gen {int(i2e_gen[g - 1])} @ bus {int(i2e_bus[ib - 1])} reached "
                        f"{mpc['gen'][g - 1, QMIN - 1]:g} MVAr Qmin lim @ lambda = {float(np.asarray(nx['lam']).reshape(-1)[0]):.4g} : "
                        f"bus {int(i2e_bus[ib - 1])} converted to PQ"
                    )
                    mpc["gen"][g - 1, QG - 1] = mpc["gen"][g - 1, QMIN - 1]

                mpc["bus"][ib - 1, BUS_TYPE - 1] = PQ
                on = np.flatnonzero(
                    (mpc["gen"][:, GEN_STATUS - 1] > 0)
                    & (mpc["bus"][mpc["gen"][:, GEN_BUS - 1].astype(int) - 1, BUS_TYPE - 1] != PQ)
                )
                if on.size == 0:
                    done["flag"] = 1
                    done["msg"] = "No REF or PV buses remaining."
                else:
                    oldref = int(np.asarray(cb_data["ref"]).reshape(-1)[0])
                    ref, pv, pq = bustypes(mpc["bus"], mpc["gen"])
                    ref_scalar = int(np.asarray(ref).reshape(-1)[0])
                    if oldref != ref_scalar:
                        mpc["bus"][ref_scalar - 1, BUS_TYPE - 1] = REF

                    cb_data["ref"] = ref
                    cb_data["pv"] = pv
                    cb_data["pq"] = pq
                    cb_data["mpc_base"]["bus"][:, BUS_TYPE - 1] = mpc["bus"][:, BUS_TYPE - 1]
                    cb_data["mpc_target"]["bus"][:, BUS_TYPE - 1] = mpc["bus"][:, BUS_TYPE - 1]
                    cb_data["mpc_base"]["gen"][g - 1, QG - 1] = mpc["gen"][g - 1, QG - 1]
                    cb_data["mpc_target"]["gen"][g - 1, QG - 1] = mpc["gen"][g - 1, QG - 1]

                    if oldref != ref_scalar:
                        cb_data["mpc_base"]["gen"][g - 1, PG - 1] = mpc["gen"][g - 1, PG - 1]
                        cb_data["mpc_target"]["gen"][g - 1, PG - 1] = mpc["gen"][g - 1, PG - 1]

                    b = cb_data["mpc_base"]
                    t = cb_data["mpc_target"]
                    cb_data["Sbusb"] = lambda Vm, b=b, mpopt=cb_data["mpopt"]: (
                        makeSbus(b["baseMVA"], b["bus"], b["gen"], mpopt, Vm, nargout=1),
                        makeSbus(b["baseMVA"], b["bus"], b["gen"], mpopt, Vm, nargout=2)[1],
                    )
                    cb_data["Sbust"] = lambda Vm, t=t, mpopt=cb_data["mpopt"]: (
                        makeSbus(t["baseMVA"], t["bus"], t["gen"], mpopt, Vm, nargout=1),
                        makeSbus(t["baseMVA"], t["bus"], t["gen"], mpopt, Vm, nargout=2)[1],
                    )
                    nx["this_step"] = 0
            ev["msg"] = msg

    evnts = evnts_list if len(evnts_list) > 1 else evnts_list[0]
    return nx, cx, done, rollback, evnts, cb_data, results
