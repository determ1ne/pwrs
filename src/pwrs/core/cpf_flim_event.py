# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .cpf_current_mpc import cpf_current_mpc
from .idx_brch import PF, PT, QF, QT, RATE_A


def cpf_flim_event(cb_data, cx):
    """Evaluate CPF branch flow limit event functions.

    Builds the current CPF case at the present continuation point and returns
    event function values for branch MVA limit violations, based on the
    larger of the from-end and to-end apparent power flows.

    Parameters
    ----------
    cb_data : dict
        CPF callback data struct.
    cx : dict
        Current CPF state struct.

    Returns
    -------
    numpy.ndarray
        Column vector of branch flow limit event function values.
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

    srate_a = d["mpc_base"]["branch"][:, RATE_A - 1]
    sf = np.sqrt(mpc["branch"][:, PF - 1] ** 2 + mpc["branch"][:, QF - 1] ** 2)
    st = np.sqrt(mpc["branch"][:, PT - 1] ** 2 + mpc["branch"][:, QT - 1] ** 2)
    return (np.maximum(sf, st) - srate_a).reshape(-1, 1)
