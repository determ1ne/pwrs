# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from .mpoption import mpoption
from .runopf import runopf


def rundcopf(casedata="case9", mpopt=None, fname="", solvedcase="", nargout=1):
    """Run a DC optimal power flow.

    Parameters
    ----------
    casedata : dict or str, optional
        MATPOWER case dict or case-file name. Defaults to ``"case9"``.
    mpopt : dict, optional
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
        Same return forms as :func:`runopf`, but with ``model="DC"``.
    """
    if mpopt is None:
        mpopt = mpoption()
    mpopt = mpoption(mpopt, "model", "DC")
    return runopf(casedata, mpopt, fname, solvedcase, nargout=nargout)
