# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def cpf_nose_event(cb_data, cx, *, nargout=None):
    """Evaluate the CPF nose-point event function.

    Parameters
    ----------
    cb_data : dict
        CPF callback data struct.
    cx : dict
        Current CPF state struct.

    Returns
    -------
    float
        Event function value for nose-point detection.
    """
    return float(np.asarray(cx["z"]).reshape(-1)[-1])
