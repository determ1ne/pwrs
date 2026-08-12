# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, cast

import numpy as np

from ..corex import MatpowerConfig
from .bustypes import bustypes
from .cpf_current_mpc import cpf_current_mpc
from .idx_bus import BUS_TYPE, PQ, REF
from .idx_gen import GEN_BUS, GEN_STATUS, PG, QG, QMAX, QMIN
from .makeSbus import makeSbus_dV, makeSbus_value
from .mpoption import mpoption


def cpf_qlim_event_cb(
    k: Any,
    nx: dict[str, Any],
    cx: dict[str, Any],
    px: dict[str, Any],
    done: dict[str, Any],
    rollback: Any,
    evnts: dict[str, Any] | list[dict[str, Any]],
    cb_data: dict[str, Any],
    cb_args: Any,
    results: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any, Any, dict[str, Any], dict[str, Any] | None]:
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
    mpopt = cb_data["mpopt"]
    if not isinstance(mpopt, MatpowerConfig):
        mpopt = mpoption(mpopt)
        cb_data["mpopt"] = mpopt

    if int(np.asarray(k).reshape(-1)[0]) <= 0 or done["flag"]:
        return nx, cx, done, rollback, evnts, cb_data, results

    mpc: dict[str, Any] | None = None
    ng = 0
    i2e_bus = np.array([], dtype=int)
    i2e_gen = np.array([], dtype=int)
    evnts_list = [evnts] if isinstance(evnts, dict) else list(evnts)

    for ev in evnts_list:
        if ev["name"] == "QLIM" and ev["zero"]:
            if mpc is None:
                d = cb_data
                ref = np.asarray(d["ref"], dtype=int).reshape(-1)
                if ref.size != 1:
                    raise ValueError(
                        "cpf_qlim_event_cb: 'cpf.enforce_qlims' option only valid for systems with exactly one REF bus"
                    )
                mpc = cast(dict[str, Any], cpf_current_mpc(
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
                ))
                ng = mpc["gen"].shape[0]
                i2e_bus = np.asarray(mpc["order"]["bus"]["i2e"]).reshape(-1)
                i2e_gen = np.asarray(mpc["order"]["gen"]["i2e"]).reshape(-1)

            if mpopt.verbose > 3:
                msg = f"{ev['msg']}\n    "
            else:
                msg = ""
            ig = np.asarray(ev["idx"], dtype=int).reshape(-1)
            for g in ig:
                maxlim = 1
                if g > ng:
                    g -= ng
                    maxlim = 0
                ib = int(mpc["gen"][g - 1, GEN_BUS])
                if maxlim:
                    msg = (
                        f"{msg}gen {int(i2e_gen[g - 1])} @ bus {int(i2e_bus[ib - 1])} reached "
                        f"{mpc['gen'][g - 1, QMAX]:g} MVAr Qmax lim @ lambda = {float(np.asarray(nx['lam']).reshape(-1)[0]):.4g} : "
                        f"bus {int(i2e_bus[ib - 1])} converted to PQ"
                    )
                    mpc["gen"][g - 1, QG] = mpc["gen"][g - 1, QMAX]
                else:
                    msg = (
                        f"{msg}gen {int(i2e_gen[g - 1])} @ bus {int(i2e_bus[ib - 1])} reached "
                        f"{mpc['gen'][g - 1, QMIN]:g} MVAr Qmin lim @ lambda = {float(np.asarray(nx['lam']).reshape(-1)[0]):.4g} : "
                        f"bus {int(i2e_bus[ib - 1])} converted to PQ"
                    )
                    mpc["gen"][g - 1, QG] = mpc["gen"][g - 1, QMIN]

                mpc["bus"][ib - 1, BUS_TYPE] = PQ
                on = np.flatnonzero(
                    (mpc["gen"][:, GEN_STATUS] > 0)
                    & (mpc["bus"][mpc["gen"][:, GEN_BUS].astype(int) - 1, BUS_TYPE] != PQ)
                )
                if on.size == 0:
                    done["flag"] = 1
                    done["msg"] = "No REF or PV buses remaining."
                else:
                    oldref = int(np.asarray(cb_data["ref"]).reshape(-1)[0])
                    ref, pv, pq = bustypes(mpc["bus"], mpc["gen"])
                    ref_scalar = int(np.asarray(ref).reshape(-1)[0])
                    if oldref != ref_scalar:
                        mpc["bus"][ref_scalar - 1, BUS_TYPE] = REF

                    cb_data["ref"] = ref
                    cb_data["pv"] = pv
                    cb_data["pq"] = pq
                    cb_data["mpc_base"]["bus"][:, BUS_TYPE] = mpc["bus"][:, BUS_TYPE]
                    cb_data["mpc_target"]["bus"][:, BUS_TYPE] = mpc["bus"][:, BUS_TYPE]
                    cb_data["mpc_base"]["gen"][g - 1, QG] = mpc["gen"][g - 1, QG]
                    cb_data["mpc_target"]["gen"][g - 1, QG] = mpc["gen"][g - 1, QG]

                    if oldref != ref_scalar:
                        cb_data["mpc_base"]["gen"][g - 1, PG] = mpc["gen"][g - 1, PG]
                        cb_data["mpc_target"]["gen"][g - 1, PG] = mpc["gen"][g - 1, PG]

                    b = cb_data["mpc_base"]
                    t = cb_data["mpc_target"]
                    cb_data["Sbusb"] = lambda Vm, b=b, mpopt=mpopt: (
                        makeSbus_value(b["baseMVA"], b["bus"], b["gen"], mpopt, Vm),
                        makeSbus_dV(b["baseMVA"], b["bus"], b["gen"], mpopt, Vm)[1],
                    )
                    cb_data["Sbust"] = lambda Vm, t=t, mpopt=mpopt: (
                        makeSbus_value(t["baseMVA"], t["bus"], t["gen"], mpopt, Vm),
                        makeSbus_dV(t["baseMVA"], t["bus"], t["gen"], mpopt, Vm)[1],
                    )
                    nx["this_step"] = 0
            ev["msg"] = msg

    evnts = evnts_list if len(evnts_list) > 1 else evnts_list[0]
    return nx, cx, done, rollback, evnts, cb_data, results
