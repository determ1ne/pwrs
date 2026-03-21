# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from .mpoption import mpoption
from .opf import opf
from .printpf import printpf
from .savecase import savecase


def runopf(casedata="case9", mpopt=None, fname="", solvedcase="", nargout=1):
    """Run an optimal power flow.

    Parameters
    ----------
    casedata : dict or str, optional
        MATPOWER case dict or case-file name. Defaults to ``"case9"``.
    mpopt : dict, optional
        MATPOWER options dict used to select the OPF algorithm, output
        options, tolerances, and related settings.
    fname : str, optional
        File name to which pretty-printed output is appended.
    solvedcase : str, optional
        File name where the solved case is saved in MATPOWER case format.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    dict or tuple
        With one or two outputs, returns the solved results dict and
        optional success flag. With more outputs, returns
        ``(baseMVA, bus, gen, gencost, branch, f, success, et)``.
    """
    if mpopt is None:
        mpopt = mpoption()
    r, success = opf(casedata, mpopt, nargout=2)
    if fname:
        with open(fname, "a", encoding="utf-8") as fd:
            printpf(
                r,
                fd,
                mpoption(mpopt, "out.all", -1, nargout=1) if mpopt.get("out", {}).get("all", 0) == 0 else mpopt,
                nargout=0,
            )
    printpf(r, 1, mpopt, nargout=0)
    if solvedcase:
        savecase(solvedcase, r, nargout=0)
    if nargout <= 2:
        return (r, success)[:nargout] if nargout > 1 else r
    return r["baseMVA"], r["bus"], r["gen"], r["gencost"], r["branch"], r["f"], success, r["et"]
