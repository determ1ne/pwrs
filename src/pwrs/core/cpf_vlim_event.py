# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .cpf_current_mpc import cpf_current_mpc
from .idx_bus import VM, VMAX, VMIN


def cpf_vlim_event(cb_data, cx):
    """Evaluate CPF bus voltage magnitude limit event functions.

    Builds the current CPF case at the present continuation point and returns
    event function values for lower and upper bus voltage magnitude
    violations.

    Parameters
    ----------
    cb_data : dict
        CPF callback data struct.
    cx : dict
        Current CPF state struct.

    Returns
    -------
    numpy.ndarray
        Stacked column vector of ``VMIN - VM`` and ``VM - VMAX`` event
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
    v_vmin = mpc["bus"][:, VMIN] - mpc["bus"][:, VM]
    v_vmax = mpc["bus"][:, VM] - mpc["bus"][:, VMAX]
    return np.r_[v_vmin, v_vmax].reshape(-1, 1)
