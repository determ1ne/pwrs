# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, cast

import numpy as np

from ..corex import MatpowerConfig
from .bustypes import bustypes
from .cpf_current_mpc import cpf_current_mpc
from .idx_bus import BUS_TYPE, PV, REF
from .idx_gen import GEN_BUS, GEN_STATUS, PG, PMAX
from .makeSbus import makeSbus_dV, makeSbus_value
from .mpoption import mpoption


def cpf_plim_event_cb(
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
    """Handle CPF active-power limit events.

    Implements MATPOWER's callback logic for ``PLIM`` events by fixing
    generators at ``Pmax``, updating the base and target cases, optionally
    switching the reference bus, and forcing a zero-length continuation step
    restart when needed.

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
    i2e_bus = np.array([], dtype=int)
    i2e_gen = np.array([], dtype=int)
    ref = np.array([], dtype=int)
    pv = np.array([], dtype=int)
    pq = np.array([], dtype=int)
    evnts_list = [evnts] if isinstance(evnts, dict) else list(evnts)

    for ev in evnts_list:
        if ev["name"] == "PLIM" and ev["zero"]:
            if mpc is None:
                d = cb_data
                ref = np.asarray(d["ref"], dtype=int).reshape(-1)
                if ref.size != 1:
                    raise ValueError(
                        "cpf_plim_event_cb: 'cpf.enforce_plims' option only valid for systems with exactly one REF bus"
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
                i2e_bus = np.asarray(mpc["order"]["bus"]["i2e"]).reshape(-1)
                i2e_gen = np.asarray(mpc["order"]["gen"]["i2e"]).reshape(-1)

            if mpopt.verbose > 3:
                msg = f"{ev['msg']}\n    "
            else:
                msg = ""
            ig = np.asarray(ev["idx"], dtype=int).reshape(-1)
            for g in ig:
                ib = int(mpc["gen"][g - 1, GEN_BUS])
                msg = (
                    f"{msg}gen {int(i2e_gen[g - 1])} @ bus {int(i2e_bus[ib - 1])} reached "
                    f"{mpc['gen'][g - 1, PMAX]:g} MW Pmax lim @ lambda = {float(np.asarray(nx['lam']).reshape(-1)[0]):.4g}"
                )
                ref_scalar = int(np.asarray(cb_data["ref"]).reshape(-1)[0])
                new_ref = np.array([], dtype=int)
                if ib == ref_scalar:
                    idx_pmax = np.flatnonzero(
                        (mpc["gen"][:, GEN_STATUS] > 0)
                        & (
                            np.abs(mpc["gen"][:, PG] - mpc["gen"][:, PMAX])
                            < float(mpopt.cpf.p_lims_tol)
                        )
                    )
                    candidates = np.zeros(mpc["bus"].shape[0], dtype=int)
                    candidates[np.asarray(cb_data["pv"], dtype=int).reshape(-1) - 1] = 1
                    candidates[mpc["gen"][idx_pmax, GEN_BUS].astype(int) - 1] = 0
                    candidates[ib - 1] = 0
                    candidate_idx = np.flatnonzero(candidates)
                    if candidate_idx.size == 0:
                        done["flag"] = 1
                        done["msg"] = "All generators at Pmax"
                    else:
                        new_ref = np.array([candidate_idx[0] + 1], dtype=int)
                        mpc["bus"][ib - 1, BUS_TYPE] = PV
                        mpc["bus"][new_ref[0] - 1, BUS_TYPE] = REF
                        ref, pv, pq = bustypes(mpc["bus"], mpc["gen"])
                        msg = f"{msg} : ref changed from bus {int(i2e_bus[ib - 1])} to {int(i2e_bus[new_ref[0] - 1])}"

                mpc["gen"][g - 1, PG] = mpc["gen"][g - 1, PMAX]

                if ib == int(np.asarray(cb_data["ref"]).reshape(-1)[0]) and new_ref.size:
                    cb_data["ref"] = ref
                    cb_data["pv"] = pv
                    cb_data["pq"] = pq
                    cb_data["mpc_base"]["bus"][ib - 1, BUS_TYPE] = mpc["bus"][ib - 1, BUS_TYPE]
                    cb_data["mpc_target"]["bus"][ib - 1, BUS_TYPE] = mpc["bus"][ib - 1, BUS_TYPE]
                    cb_data["mpc_base"]["bus"][new_ref[0] - 1, BUS_TYPE] = mpc["bus"][new_ref[0] - 1, BUS_TYPE]
                    cb_data["mpc_target"]["bus"][new_ref[0] - 1, BUS_TYPE] = mpc["bus"][
                        new_ref[0] - 1, BUS_TYPE
                    ]

                cb_data["mpc_base"]["gen"][g - 1, PG] = mpc["gen"][g - 1, PG]
                cb_data["mpc_target"]["gen"][g - 1, PG] = mpc["gen"][g - 1, PG]
                cb_data["idx_pmax"] = np.r_[np.asarray(cb_data.get("idx_pmax", np.array([])), dtype=int).reshape(-1), g]

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
