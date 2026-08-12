# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import time
from typing import Any, cast

import numpy as np

from ..corex import CaseData, MatpowerCase, MatpowerConfig, OptimalPowerFlowResult
from ..power_models import solve_power_models_opf
from .ext2int import ext2int
from .idx_brch import MU_ANGMAX, MU_ANGMIN, MU_SF, MU_ST, PF, PT, QF, QT
from .idx_bus import MU_VMIN
from .idx_gen import MU_PMAX, MU_PMIN, MU_QMIN, PG, QG
from .int2ext import int2ext
from .mpoption import mpoption
from .opf_args import opf_args_case
from .opf_execute import opf_execute_full
from .opf_setup import opf_setup_model
from .runpf import runpf


def _solve_power_models_opf_full(mpc: CaseData | MatpowerCase, mpopt: MatpowerConfig) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Return PowerModels results, success flag and raw solver data."""
    return cast(tuple[dict[str, Any], float, dict[str, Any]], solve_power_models_opf(mpc, mpopt, nargout=3))


def opf(*args, nargout=1):
    """Solve an optimal power flow.

    This port preserves the MATLAB ``opf`` calling conventions, supporting
    case dicts, case names, and the extended argument forms handled by
    :func:`opf_args`.

    Parameters
    ----------
    *args
        MATPOWER OPF inputs. Supported forms include ``opf(mpc)``,
        ``opf(mpc, mpopt)``, user linear/cost extensions, and the expanded
        ``baseMVA, bus, gen, branch, ...`` signatures.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    OptimalPowerFlowResult or tuple
        With one or two outputs, returns the structured solved result and
        optional success flag. With more outputs, returns
        ``(bus, gen, branch, f, success, info, et, g, jac, xr, pimul)``
        with ``g`` and ``jac`` currently left as ``None``.
    """
    t0 = time.time()
    mpc, mpopt = cast(tuple[CaseData | MatpowerCase, MatpowerConfig], opf_args_case(*args))
    mpc_mapping = cast(CaseData, mpc.to_dict() if isinstance(mpc, MatpowerCase) else mpc)
    if str(mpopt.opf.backend).upper() == "POWER_MODELS":
        results, success, raw = _solve_power_models_opf_full(mpc, mpopt)
        et = time.time() - t0
        results["success"] = bool(success)
        results["et"] = et
        results["raw"] = raw
        if nargout <= 2:
            structured = OptimalPowerFlowResult.from_mapping(results)
            return (structured, structured.success) if nargout > 1 else structured
        outputs = (
            results["bus"],
            results["gen"],
            results["branch"],
            results["f"],
            success,
            raw["info"],
            et,
            None,
            None,
            raw["xr"],
            raw["pimul"],
        )
        return outputs[:nargout]
    if str(mpopt.opf.backend).upper() != "MATPOWER":
        raise ValueError(f"unsupported OPF backend: {mpopt.opf.backend}")
    if mpopt.opf.ac.solver.upper() == "DEFAULT":
        mpopt = mpoption(mpopt, "opf.ac.solver", "MIPS")
    if mpopt.opf.start == 3:
        mpopt_pf = mpoption(mpopt, "out.all", 0, "verbose", max(0, mpopt.verbose - 1))
        rpf_value = runpf(mpc_mapping, mpopt_pf)
        rpf = cast(CaseData, rpf_value.to_dict())
        if rpf.get("success", 0):
            mpc_mapping = cast(CaseData, {key: value for key, value in rpf.items() if value is not None})
            mpc_mapping.pop("order", None)
    nb = mpc_mapping["bus"].shape[0]
    nl = mpc_mapping["branch"].shape[0]
    ng = mpc_mapping["gen"].shape[0]
    if mpc_mapping["bus"].shape[1] < MU_VMIN:
        mpc_mapping["bus"] = np.c_[mpc_mapping["bus"], np.zeros((nb, MU_VMIN - mpc_mapping["bus"].shape[1]))]
    if mpc_mapping["gen"].shape[1] < MU_QMIN:
        mpc_mapping["gen"] = np.c_[mpc_mapping["gen"], np.zeros((ng, MU_QMIN - mpc_mapping["gen"].shape[1]))]
    if mpc_mapping["branch"].shape[1] < MU_ANGMAX:
        mpc_mapping["branch"] = np.c_[mpc_mapping["branch"], np.zeros((nl, MU_ANGMAX - mpc_mapping["branch"].shape[1]))]
    mpc = cast(CaseData, ext2int(mpc_mapping, mpopt))
    om = opf_setup_model(mpc, mpopt)
    results, success, raw = opf_execute_full(om, mpopt)
    results["success"] = success
    results = cast(dict[str, Any], int2ext(results))
    if "order" in results:
        gen_off = np.asarray(results["order"]["gen"]["status"]["off"]).astype(int).reshape(-1)
        if gen_off.size:
            results["gen"][
                np.ix_(gen_off - 1, np.array([PG - 1, QG - 1, MU_PMAX - 1, MU_PMIN - 1], dtype=int))
            ] = 0
        branch_off = np.asarray(results["order"]["branch"]["status"]["off"]).astype(int).reshape(-1)
        if branch_off.size:
            results["branch"][
                np.ix_(
                    branch_off - 1,
                    np.array(
                        [PF - 1, QF - 1, PT - 1, QT - 1, MU_SF - 1, MU_ST - 1, MU_ANGMIN - 1, MU_ANGMAX - 1],
                        dtype=int,
                    ),
                )
            ] = 0
    et = time.time() - t0
    if nargout <= 2:
        results["et"] = et
        results["raw"] = raw
        structured = OptimalPowerFlowResult.from_mapping(results)
        return (structured, structured.success) if nargout > 1 else structured
    busout = results["bus"]
    genout = results["gen"]
    branchout = results["branch"]
    f = results["f"]
    info = raw["info"]
    xr = raw["xr"]
    pimul = raw["pimul"]
    outputs = (busout, genout, branchout, f, success, info, et, None, None, xr, pimul)
    return outputs[:nargout]


def opf_with_success(*args: Any) -> tuple[OptimalPowerFlowResult, bool]:
    """Solve an OPF and return ``(results, success)``."""
    result = cast(OptimalPowerFlowResult, opf(*args))
    return result, result.success
