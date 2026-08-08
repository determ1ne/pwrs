# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from ..corex import MatpowerConfig
from .mpoption import mpoption


def cpf_target_lam_event(cb_data, cx):
    """Evaluate the CPF target-lambda event function.

    Returns the signed distance between the current continuation parameter and
    the configured CPF stopping target. For full-curve tracing modes the
    target is derived from the sign of the natural parameter state.

    Parameters
    ----------
    cb_data : dict
        CPF callback data struct.
    cx : dict
        Current CPF state struct.

    Returns
    -------
    float
        Event function value for target-lambda detection.
    """
    mpopt = cb_data["mpopt"]
    if not isinstance(mpopt, MatpowerConfig):
        mpopt = mpoption(mpopt)
    target = mpopt.cpf.stop_at
    if isinstance(target, np.ndarray) and target.size == 1:
        target = target.reshape(-1)[0].item()
    if isinstance(target, str):
        if float(np.asarray(cx["z"]).reshape(-1)[-1]) >= 0:
            target = -1
        else:
            target = 0
    return float(np.asarray(cx["lam"]).reshape(-1)[0]) - float(target)
