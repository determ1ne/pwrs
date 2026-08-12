# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy
import warnings
from os import PathLike
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..corex import MatpowerCase
from ..corex import from_matpower_mat as loadcase_matfile  # noqa: F401
from ..corex.io import load as load_case_file
from .idx_brch import BR_STATUS, MU_ST, PF, QT
from .idx_gen import APF, MU_PMAX, MU_QMIN, PMIN


def _copy_case(mpc: dict[str, Any] | MatpowerCase) -> dict[str, Any]:
    data = mpc.to_dict() if isinstance(mpc, MatpowerCase) else mpc
    return copy.deepcopy(data)


def _as_array(value: Any) -> np.ndarray:
    return np.atleast_2d(np.array(value, copy=True))


def _version_string(value: Any) -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    return str(arr.reshape(-1)[0])


def _mpc_1to2(gen: np.ndarray, branch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gen = np.array(gen, copy=True)
    branch = np.array(branch, copy=True)

    if gen.shape[1] > APF:
        raise ValueError("mpc_1to2: gen matrix appears to already be in version 2 format")

    gen_shift = MU_PMAX - PMIN - 1
    gen_tmp = np.zeros((gen.shape[0], gen_shift), dtype=gen.dtype)
    if gen.shape[1] >= MU_QMIN:
        gen = np.hstack((gen[:, :PMIN], gen_tmp, gen[:, MU_PMAX - 1 : MU_QMIN]))
    else:
        gen = np.hstack((gen[:, :PMIN], gen_tmp))

    branch_tmp = np.full((branch.shape[0], 2), [-360, 360], dtype=branch.dtype)
    branch_tmp2 = np.zeros((branch.shape[0], 2), dtype=branch.dtype)
    if branch.shape[1] >= MU_ST:
        branch = np.hstack((branch[:, :BR_STATUS], branch_tmp, branch[:, PF - 1 : MU_ST], branch_tmp2))
    elif branch.shape[1] >= QT:
        branch = np.hstack((branch[:, :BR_STATUS], branch_tmp, branch[:, PF - 1 : QT]))
    else:
        branch = np.hstack((branch[:, :BR_STATUS], branch_tmp))

    return gen, branch


def loadcase_embedded(case_name: str) -> MatpowerCase:
    """Load a built-in pwrs case struct.

    Parameters
    ----------
    case_name : str
        Name of the built-in case to load. Should correspond to a function in
        ``pwrs.cases`` that returns a case struct.

    Returns
    -------
    MatpowerCase
        The loaded case struct as a MatpowerCase instance.
    """
    import importlib

    try:
        cases_module = importlib.import_module("pwrs.data.matpower")
        case_func = getattr(cases_module, case_name)
        case_struct = case_func()
        if isinstance(case_struct, MatpowerCase):
            return case_struct
        return MatpowerCase.from_dict(case_struct)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"loadcase_embedded: MATPOWER case '{case_name}' not found") from e


def _loadcase_impl(casefile: Any, *, nargout: int | None = None) -> MatpowerCase | tuple:
    """Load a MATPOWER case from an in-memory case struct.

    .. deprecated:: 0.1.1
       Use :func:`pwrs.load` for file and mapping IO. This compatibility
       function remains available for MATPOWER-style ``nargout`` behavior.

    Mirrors the struct-handling branch of MATPOWER's ``loadcase``. It
    validates the required fields, normalizes matrices to 2-D arrays, and
    converts version 1 case data to version 2 format when needed.

    Parameters
    ----------
    casefile : dict or str
        MATPOWER case struct, or the name of a built-in case to load.
    nargout : int, optional
        MATLAB compatibility flag controlling whether the case is returned as
        a struct or as expanded ``baseMVA, bus, gen, branch, ...`` outputs.

    Returns
    -------
    dict or tuple
        Normalized MATPOWER case struct, or the expanded MATLAB-style output
        tuple when ``nargout >= 3``.
    """

    if isinstance(casefile, (str, PathLike)):
        path = Path(casefile)
        if path.suffix.lower() in {".json", ".mat", ".npz", ".xlsx", ".xls"}:
            return load_case_file(path)
        if not isinstance(casefile, str):
            raise ValueError(f"loadcase: cannot infer case format from path {path}")
        return loadcase_embedded(casefile)

    if not isinstance(casefile, dict) and not isinstance(casefile, MatpowerCase):
        raise TypeError("loadcase: input arg should be a struct containing MATPOWER case data")
    casefile = MatpowerCase.from_dict(casefile) if not isinstance(casefile, MatpowerCase) else casefile

    if nargout is None:
        nargout = 1

    return_as_struct = nargout < 3
    expect_gencost = nargout >= 5
    expect_areas = nargout > 5

    s = _copy_case(casefile)

    if expect_areas and "areas" not in s:
        s["areas"] = np.array([])

    missing_required = not all(field in s for field in ("baseMVA", "bus", "gen", "branch"))
    missing_gencost = expect_gencost and "gencost" not in s
    if missing_required or missing_gencost:
        raise ValueError("loadcase: syntax error or undefined data matrix(ices) in the file\nmissing data")

    if "areas" in s and np.asarray(s["areas"]).size == 0 and not expect_areas:
        del s["areas"]

    mpc = _copy_case(s)
    mpc["baseMVA"] = float(np.asarray(mpc["baseMVA"]).reshape(-1)[0])
    mpc["bus"] = _as_array(mpc["bus"])
    mpc["gen"] = _as_array(mpc["gen"])
    mpc["branch"] = _as_array(mpc["branch"])
    if "areas" in mpc:
        mpc["areas"] = _as_array(mpc["areas"])
    if "gencost" in mpc:
        mpc["gencost"] = _as_array(mpc["gencost"])

    if "version" not in mpc:
        mpc["version"] = "1" if mpc["gen"].shape[1] < 21 else "2"
    else:
        mpc["version"] = _version_string(mpc["version"])

    if mpc["version"] == "1":
        mpc["gen"], mpc["branch"] = _mpc_1to2(mpc["gen"], mpc["branch"])
        mpc["version"] = "2"

    if return_as_struct:
        return MatpowerCase.from_dict(mpc)

    outputs: list[Any] = [mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"]]
    if expect_gencost:
        if expect_areas:
            outputs.extend([mpc.get("areas", np.array([])), mpc["gencost"]])
        else:
            outputs.append(mpc["gencost"])
    if nargout >= 7:
        outputs.append(0)
    return tuple(outputs)


def loadcase(casefile: Any, *, nargout: int | None = None) -> MatpowerCase | tuple:
    """Deprecated MATPOWER-compatible loader.

    .. deprecated:: 0.1.1
       Use :func:`pwrs.load` for file and mapping IO.
    """
    warnings.warn(
        "loadcase() is deprecated for data IO; use pwrs.load() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return _loadcase_impl(casefile, nargout=nargout)


def loadcase_struct(casefile: Any) -> MatpowerCase:
    """Load and normalize a case as a MATPOWER case struct."""
    return cast(MatpowerCase, _loadcase_impl(casefile, nargout=1))


def loadcase_expanded(casefile: Any) -> tuple:
    """Load a case using the explicit expanded MATLAB output convention."""
    return cast(tuple, _loadcase_impl(casefile, nargout=4))
