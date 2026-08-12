# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .cpf_current_mpc import cpf_current_mpc
from .idx_gen import GEN_STATUS, PG, PMAX


def cpf_plim_event(cb_data, cx):
    """Evaluate CPF active-power limit event functions.

    Builds the current CPF case at the present continuation point and returns
    event function values for generator ``Pmax`` violations, with inactive or
    already-handled generators masked by ``NaN``.

    Parameters
    ----------
    cb_data : dict
        CPF callback data struct.
    cx : dict
        Current CPF state struct.

    Returns
    -------
    numpy.ndarray
        Column vector of ``PG - PMAX`` event function values.
    """
    d = cb_data
    mpc = cpf_current_mpc(
        d["mpc_base"],
        d["mpc_target"],
        d["Ybus"],
        d["Yf"],
        d["Yt"],
        d["ref"],
        d["pv"],
        d["pq"],
        cx["V"],
        cx["lam"],
        d["mpopt"],
    )

    ng = mpc["gen"].shape[0]
    v_pmax = np.full((ng, 1), np.nan, dtype=float)
    on = np.flatnonzero(mpc["gen"][:, GEN_STATUS] > 0)
    v_pmax[on, 0] = mpc["gen"][on, PG] - mpc["gen"][on, PMAX]
    idx_pmax = np.asarray(d.get("idx_pmax", np.array([])), dtype=int).reshape(-1)
    if idx_pmax.size:
        v_pmax[idx_pmax - 1, 0] = np.nan
    return v_pmax
