# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .cpf_current_mpc import cpf_current_mpc
from .idx_bus import BUS_TYPE, PQ
from .idx_gen import GEN_BUS, GEN_STATUS, QG, QMAX, QMIN


def cpf_qlim_event(cb_data, cx):
    """Evaluate CPF reactive-power limit event functions.

    Builds the current CPF case at the present continuation point and returns
    event function values for generator ``Qmax`` and ``Qmin`` violations for
    online generators that are not already at PQ buses.

    Parameters
    ----------
    cb_data : dict
        CPF callback data struct.
    cx : dict
        Current CPF state struct.

    Returns
    -------
    numpy.ndarray
        Stacked column vector of ``QG - QMAX`` and ``QMIN - QG`` event
        function values.
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

    nb = mpc["bus"].shape[0]
    ng = mpc["gen"].shape[0]
    on = np.flatnonzero(
        (mpc["gen"][:, GEN_STATUS - 1] > 0)
        & (mpc["bus"][mpc["gen"][:, GEN_BUS - 1].astype(int) - 1, BUS_TYPE - 1] != PQ)
    )
    gbus = mpc["gen"][on, GEN_BUS - 1].astype(int)
    ngon = on.size
    Cg = sparse.csc_matrix((np.ones(ngon), (np.arange(ngon), gbus - 1)), shape=(ngon, nb))
    C = Cg @ Cg.T

    v_qmax = np.full((ng, 1), np.nan, dtype=float)
    v_qmin = np.full((ng, 1), np.nan, dtype=float)
    if on.size:
        v_qmax[on, 0] = np.asarray(C @ (mpc["gen"][on, QG - 1] - mpc["gen"][on, QMAX - 1])).reshape(-1)
        v_qmin[on, 0] = np.asarray(C @ (mpc["gen"][on, QMIN - 1] - mpc["gen"][on, QG - 1])).reshape(-1)
    return np.vstack([v_qmax, v_qmin])
