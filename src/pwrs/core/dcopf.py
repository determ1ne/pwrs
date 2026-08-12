# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from .mpoption import mpoption
from .opf import opf
from .opf_args import opf_args_case


def dcopf(*args, nargout=1):
    """Solve a DC optimal power flow.

    This is a thin wrapper around :func:`opf` that sets ``model`` to
    ``"DC"`` before dispatching.

    Parameters
    ----------
    *args
        Inputs accepted by :func:`opf`.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    dict or tuple
        Outputs from :func:`opf` for the DC model.
    """
    mpc, mpopt_value = opf_args_case(*args)
    mpopt_value = mpoption(mpopt_value, "model", "DC")
    return opf(mpc, mpopt_value, nargout=nargout)
