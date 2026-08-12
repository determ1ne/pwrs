# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping
from typing import Literal, overload

from ..corex import MatpowerCase, MatpowerConfig, PowerFlowExpandedResult, PowerFlowResult
from .mpoption import mpoption
from .runpf import runpf


@overload
def rundcpf(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt_value: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
    *,
    nargout: None | Literal[1] = None,
) -> PowerFlowResult: ...


@overload
def rundcpf(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt_value: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
    *,
    nargout: Literal[2],
) -> tuple[PowerFlowResult, bool]: ...


@overload
def rundcpf(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt_value: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
    *,
    nargout: int,
) -> PowerFlowResult | tuple[PowerFlowResult, bool] | PowerFlowExpandedResult | None: ...


def rundcpf(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt_value: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
    *,
    nargout: int | None = None,
) -> PowerFlowResult | tuple[PowerFlowResult, bool] | PowerFlowExpandedResult | None:
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
    PowerFlowResult or tuple
        Same structured return forms as :func:`runpf`, but with
        ``model="DC"``.
    """
    if mpopt_value is None:
        mpopt_value = mpoption()
    mpopt_value = mpoption(mpopt_value, "model", "DC")
    result = runpf(casedata, mpopt_value, fname, solvedcase)
    if nargout in (None, 1):
        return result
    if nargout == 2:
        return result, result.success
    if nargout > 2:
        return PowerFlowExpandedResult(
            baseMVA=result.baseMVA,
            bus=result.bus,
            gen=result.gen,
            branch=result.branch,
            success=result.success,
            et=result.et,
        )
    return None


def rundcpf_with_success(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt_value: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
) -> tuple[PowerFlowResult, bool]:
    """Run a DC power flow and return ``(result, success)``."""
    result = rundcpf(casedata, mpopt_value, fname, solvedcase)
    return result, result.success


def rundcpf_expanded(
    casedata: str | MatpowerCase | Mapping[str, object] = "case9",
    mpopt_value: MatpowerConfig | Mapping[str, object] | None = None,
    fname: str = "",
    solvedcase: str = "",
) -> PowerFlowExpandedResult:
    """Run a DC power flow and return the named expanded result."""
    result = rundcpf(casedata, mpopt_value, fname, solvedcase)
    return PowerFlowExpandedResult(
        result.baseMVA,
        result.bus,
        result.gen,
        result.branch,
        result.success,
        result.et,
    )
