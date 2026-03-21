# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from .mpoption import mpoption
from .runpf import runpf


def rundcpf(casedata="case9", mpopt_value=None, fname="", solvedcase="", *, nargout=None):
    """Run a DC power flow.

    This is a thin wrapper around :func:`runpf` that forces the model to
    ``"DC"`` before dispatching.

    Parameters
    ----------
    casedata : dict or str, optional
        MATPOWER case dict or case-file name. Defaults to ``"case9"``.
    mpopt_value : dict, optional
        MATPOWER options dict to override defaults.
    fname : str, optional
        File name to which pretty-printed output is appended.
    solvedcase : str, optional
        File name where the solved case is saved in MATPOWER case format.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    dict or tuple
        Same return forms as :func:`runpf`, but with ``model="DC"``.
    """
    if mpopt_value is None:
        mpopt_value = mpoption()
    mpopt_value = mpoption(mpopt_value, "model", "DC")
    return runpf(casedata, mpopt_value, fname, solvedcase, nargout=nargout)
