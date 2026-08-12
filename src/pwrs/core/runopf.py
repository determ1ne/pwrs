# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping
from typing import Literal, overload

from ..corex import (
    MatpowerCase,
    MatpowerConfig,
    OptimalPowerFlowExpandedResult,
    OptimalPowerFlowResult,
)
from .mpoption import mpoption
from .opf import opf_with_success
from .printpf import printpf
from .savecase import savecase


@overload
def runopf(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
    nargout: Literal[1] = 1,
) -> OptimalPowerFlowResult: ...


@overload
def runopf(
    casedata: str | MatpowerCase | Mapping[str, object],
    mpopt: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
    *,
    nargout: Literal[2],
) -> tuple[OptimalPowerFlowResult, bool]: ...


@overload
def runopf(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
    nargout: int = 1,
) -> OptimalPowerFlowResult | tuple[OptimalPowerFlowResult, bool] | OptimalPowerFlowExpandedResult: ...


def runopf(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
    nargout: int = 1,
) -> OptimalPowerFlowResult | tuple[OptimalPowerFlowResult, bool] | OptimalPowerFlowExpandedResult:
    """Run an optimal power flow.

    Parameters
    ----------
    casedata : dict or str, optional
        MATPOWER case dict or case-file name. Defaults to ``"case9"``.
    mpopt : MatpowerConfig or dict, optional
        MATPOWER options configuration used to select the OPF algorithm, output
        options, tolerances, and related settings.
    fname : str, optional
        File name to which pretty-printed output is appended.
    solvedcase : str, optional
        File name where the solved case is saved in MATPOWER case format.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    OptimalPowerFlowResult or tuple
        With one or two outputs, returns the structured solved result and
        optional success flag. With more outputs, returns
        ``(baseMVA, bus, gen, gencost, branch, f, success, et)``.
    """
    if mpopt is None:
        mpopt = mpoption()
    elif not isinstance(mpopt, MatpowerConfig):
        mpopt = mpoption(mpopt)
    r, success = opf_with_success(casedata, mpopt)
    if fname:
        with open(fname, "a", encoding="utf-8") as fd:
            printpf(
                r.to_dict(),
                fd,
                mpoption(mpopt, "out.all", -1) if mpopt.out.all == 0 else mpopt,
            )
    printpf(r.to_dict(), 1, mpopt)
    if solvedcase:
        savecase(solvedcase, r.to_dict())
    if nargout == 1:
        return r
    if nargout == 2:
        return r, success
    return OptimalPowerFlowExpandedResult(
        r.baseMVA,
        r.bus,
        r.gen,
        r.gencost,
        r.branch,
        r.f,
        r.success,
        r.et,
    )


def runopf_with_success(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
) -> tuple[OptimalPowerFlowResult, bool]:
    """Run an OPF and return ``(result, success)``."""
    result = runopf(casedata, mpopt, fname, solvedcase)
    return result, result.success


def runopf_expanded(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
) -> OptimalPowerFlowExpandedResult:
    """Run an OPF and return the named MATLAB-style expanded result."""
    result = runopf(casedata, mpopt, fname, solvedcase)
    return OptimalPowerFlowExpandedResult(
        result.baseMVA,
        result.bus,
        result.gen,
        result.gencost,
        result.branch,
        result.f,
        result.success,
        result.et,
    )
