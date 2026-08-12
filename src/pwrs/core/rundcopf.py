# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping
from typing import Literal, overload

from ..corex import MatpowerCase, MatpowerConfig, OptimalPowerFlowExpandedResult, OptimalPowerFlowResult
from .mpoption import mpoption
from .runopf import runopf


@overload
def rundcopf(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
    nargout: Literal[1] = 1,
) -> OptimalPowerFlowResult: ...


@overload
def rundcopf(
    casedata: str | MatpowerCase | Mapping[str, object],
    mpopt: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
    *,
    nargout: Literal[2],
) -> tuple[OptimalPowerFlowResult, bool]: ...


@overload
def rundcopf(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
    nargout: int = 1,
) -> OptimalPowerFlowResult | tuple[OptimalPowerFlowResult, bool] | OptimalPowerFlowExpandedResult: ...


def rundcopf(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
    nargout: int = 1,
) -> OptimalPowerFlowResult | tuple[OptimalPowerFlowResult, bool] | OptimalPowerFlowExpandedResult:
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
    OptimalPowerFlowResult or tuple
        Same structured return forms as :func:`runopf`, but with
        ``model="DC"``.
    """
    if mpopt is None:
        mpopt = mpoption()
    mpopt = mpoption(mpopt, "model", "DC")
    result = runopf(casedata, mpopt, fname, solvedcase)
    if nargout <= 2:
        return (result, result.success) if nargout > 1 else result
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


def rundcopf_with_success(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
) -> tuple[OptimalPowerFlowResult, bool]:
    """Run a DC OPF and return ``(result, success)``."""
    result = rundcopf(casedata, mpopt, fname, solvedcase)
    return result, result.success


def rundcopf_expanded(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
) -> OptimalPowerFlowExpandedResult:
    """Run a DC OPF and return the named expanded result."""
    result = rundcopf(casedata, mpopt, fname, solvedcase)
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
