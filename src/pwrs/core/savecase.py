# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import sparse

from ..corex import MatpowerCase, StructuredMapping
from ..corex import to_matpower_mat as savecase_matfile  # noqa: F401
from ..corex.io import save as save_case_file
from .idx_brch import ANGMAX, MU_ANGMAX, MU_ST, QT
from .idx_bus import MU_VMIN, VMIN
from .idx_cost import MODEL, NCOST, POLYNOMIAL, PW_LINEAR
from .idx_gen import APF, MU_QMIN
from .run_userfcn import run_userfcn


def _as_array(value: Any) -> np.ndarray[Any, Any]:
    arr = np.asarray(value)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return np.array(arr, copy=True)


def _cellstr(value: Any) -> list[str]:
    arr = np.asarray(value, dtype=object)
    if arr.size == 0:
        return []
    return [str(item) for item in arr.reshape(-1)]


def _matlab_comment(comment: Any) -> list[str]:
    if isinstance(comment, str):
        return [comment]
    arr = np.asarray(comment, dtype=object)
    return [str(item) for item in arr.reshape(-1)] if arr.size else [""]


def _print_sparse_lines(varname: str, A: Any) -> list[str]:
    if sparse.issparse(A):
        A = A.tocoo()
        rows = A.row + 1
        cols = A.col + 1
        vals = A.data
        shape = A.shape
    else:
        rows, cols = np.nonzero(A)
        vals = np.asarray(A)[rows, cols]
        rows = rows + 1
        cols = cols + 1
        shape = np.asarray(A).shape

    if len(vals) == 0:
        return [f"{varname} = sparse({shape[0]}, {shape[1]});"]

    lines = ["ijs = ["]
    for i, j, s in zip(rows, cols, vals):
        lines.append(f"\t{int(i)}\t{int(j)}\t{float(s):.9g};")
    lines.append("];")
    lines.append(f"{varname} = sparse(ijs(:, 1), ijs(:, 2), ijs(:, 3), {shape[0]}, {shape[1]});")
    return lines


def _matrix_lines(prefix: str, name: str, header: str, data: np.ndarray, fmts: list[str]) -> list[str]:
    lines = ["", header, f"{prefix}{name} = ["]
    for row in data:
        vals = []
        for value, fmt in zip(row, fmts):
            if fmt == "%d":
                vals.append(fmt % int(round(float(value))))
            else:
                vals.append(fmt % float(value))
        lines.append("\t" + "\t".join(vals) + ";")
    lines.append("];")
    return lines


def _normalize_inputs(
    args: tuple[Any, ...],
) -> tuple[str, list[str], dict[str, Any], str, MatpowerCase | Mapping[str, object]]:
    if len(args) < 2:
        raise TypeError("savecase: expected at least a filename and case data")

    fname = str(args[0])
    remaining = list(args[1:])
    if isinstance(remaining[0], (str, list, tuple, np.ndarray)) and not isinstance(remaining[0], Mapping):
        comment = _matlab_comment(remaining.pop(0))
    else:
        comment = [""]

    if not remaining or not isinstance(remaining[0], (Mapping, MatpowerCase)):
        raise NotImplementedError("savecase: Python port currently supports the MATPOWER struct form only")

    case_value = remaining[0]
    if isinstance(case_value, (MatpowerCase, StructuredMapping)):
        mpc = copy.deepcopy(case_value.to_dict())
    else:
        mpc = copy.deepcopy(dict(cast(Mapping[str, Any], case_value)))
    mpc_ver = "2"
    if len(remaining) > 1:
        mpc_ver = str(remaining[1])
        if mpc_ver != "2":
            raise NotImplementedError(f"savecase: unsupported MATPOWER version '{mpc_ver}'")
    return fname, comment, mpc, mpc_ver, case_value


def savecase(*args: Any, nargout: int | None = None):
    """Save a MATPOWER case to ``.m`` or ``.mat`` format.

    .. deprecated:: 0.1.1
       Use :func:`pwrs.save` for JSON, MAT, NPZ, and Excel case IO. This
       compatibility function remains available for MATLAB ``.m`` output.

    Mirrors MATPOWER's ``savecase`` entry point for the struct form of case
    data. It writes either a MATLAB case file or a MAT-file, preserving the
    requested MATPOWER version layout where supported.

    Parameters
    ----------
    *args
        MATPOWER-style ``savecase`` inputs beginning with the output file
        name, followed by an optional comment block, the case struct, and an
        optional version selector.
    nargout : int, optional
        MATLAB compatibility flag controlling whether the output file name is
        returned.

    Returns
    -------
    str or None
        Output file name when ``nargout == 1``, otherwise ``None``.
    """
    if nargout not in (None, 0, 1):
        raise ValueError("savecase: expected zero or one outputs")

    warnings.warn(
        "savecase() is deprecated for data IO; use pwrs.save() instead",
        DeprecationWarning,
        stacklevel=2,
    )

    fname, comment, mpc, mpc_ver, case_source = _normalize_inputs(args)

    baseMVA = float(mpc["baseMVA"])
    bus = _as_array(mpc["bus"]).astype(float)
    gen = _as_array(mpc["gen"]).astype(float)
    branch = _as_array(mpc["branch"]).astype(float)
    areas = _as_array(mpc["areas"]).astype(float) if "areas" in mpc and np.asarray(mpc["areas"]).size else None
    gencost = _as_array(mpc["gencost"]).astype(float) if "gencost" in mpc and np.asarray(mpc["gencost"]).size else None

    path = Path(fname)
    ext = path.suffix or ".m"
    raw_name = path.stem
    safe_name = re.sub(r"\W", "_", raw_name)
    if safe_name != raw_name:
        print(f"WARNING: '{raw_name}' is not a valid function name, changed to '{safe_name}'")
    fname_out = str(path.with_name(safe_name + ext))

    if ext.lower() in {".json", ".mat", ".npz", ".xlsx", ".xls"}:
        save_case_file(case_source, fname_out)
        return fname_out if nargout == 1 else None

    prefix = "" if mpc_ver == "1" else "mpc."
    lines: list[str] = []

    lines.append(f"function mpc = {safe_name}")

    if not comment or comment[0] == "":
        comment[0] = safe_name.upper()
    else:
        comment[0] = f"{safe_name.upper()}  {comment[0]}"
    for item in comment:
        lines.append(f"%{item}")
    lines.append("")
    lines.append(f"%% MATPOWER Case Format : Version {mpc_ver}")
    lines.append(f"mpc.version = '{mpc_ver}';")
    lines.append("")
    lines.append("%%-----  Power Flow Data  -----%%")
    lines.append("%% system MVA base")
    lines.append(f"{prefix}baseMVA = {baseMVA:.9g};")

    bus_header = "%% bus data\n%\tbus_i\ttype\tPd\tQd\tGs\tBs\tarea\tVm\tVa\tbaseKV\tzone\tVmax\tVmin"
    if bus.shape[1] >= MU_VMIN:
        bus_header += "\tlam_P\tlam_Q\tmu_Vmax\tmu_Vmin"
    bus_cols = bus[:, : MU_VMIN if bus.shape[1] >= MU_VMIN else VMIN]
    bus_fmts = ["%d", "%d", "%.9g", "%.9g", "%.9g", "%.9g", "%d", "%.9g", "%.9g", "%.9g", "%d", "%.9g", "%.9g"]
    if bus.shape[1] >= MU_VMIN:
        bus_fmts += ["%.4f", "%.4f", "%.4f", "%.4f"]
    lines.extend(_matrix_lines(prefix, "bus", bus_header, bus_cols, bus_fmts))

    gen_header = "%% generator data\n%\tbus\tPg\tQg\tQmax\tQmin\tVg\tmBase\tstatus\tPmax\tPmin"
    gen_header += "\tPc1\tPc2\tQc1min\tQc1max\tQc2min\tQc2max\tramp_agc\tramp_10\tramp_30\tramp_q\tapf"
    if gen.shape[1] >= MU_QMIN:
        gen_header += "\tmu_Pmax\tmu_Pmin\tmu_Qmax\tmu_Qmin"
    if gen.shape[1] < MU_QMIN:
        gen_cols = gen[:, :APF]
    else:
        gen_cols = gen[:, :MU_QMIN]
    gen_fmts = ["%d", "%.9g", "%.9g", "%.9g", "%.9g", "%.9g", "%.9g", "%d", "%.9g", "%.9g"]
    gen_fmts += ["%.9g"] * 11
    if gen.shape[1] >= MU_QMIN:
        gen_fmts += ["%.4f", "%.4f", "%.4f", "%.4f"]
    lines.extend(_matrix_lines(prefix, "gen", gen_header, gen_cols, gen_fmts))

    branch_header = "%% branch data\n%\tfbus\ttbus\tr\tx\tb\trateA\trateB\trateC\tratio\tangle\tstatus"
    branch_header += "\tangmin\tangmax"
    if branch.shape[1] >= QT:
        branch_header += "\tPf\tQf\tPt\tQt"
    if branch.shape[1] >= MU_ST:
        branch_header += "\tmu_Sf\tmu_St"
        branch_header += "\tmu_angmin\tmu_angmax"
    if branch.shape[1] < QT:
        branch_cols = branch[:, :ANGMAX]
    elif branch.shape[1] < MU_ST:
        branch_cols = branch[:, :QT]
    else:
        branch_cols = branch[:, :MU_ANGMAX]
    branch_fmts = ["%d", "%d", "%.9g", "%.9g", "%.9g", "%.9g", "%.9g", "%.9g", "%.9g", "%.9g", "%d"]
    branch_fmts += ["%.9g", "%.9g"]
    if branch.shape[1] >= QT:
        branch_fmts += ["%.4f", "%.4f", "%.4f", "%.4f"]
    if branch.shape[1] >= MU_ST:
        branch_fmts += ["%.4f", "%.4f"]
        branch_fmts += ["%.4f", "%.4f"]
    lines.extend(_matrix_lines(prefix, "branch", branch_header, branch_cols, branch_fmts))

    if (areas is not None and areas.size) or (gencost is not None and gencost.size):
        lines.append("")
        lines.append("%%-----  OPF Data  -----%%")
    if areas is not None and areas.size:
        lines.extend(_matrix_lines(prefix, "areas", "%% area data\n%\tarea\trefbus", areas[:, :2], ["%d", "%d"]))
    if gencost is not None and gencost.size:
        lines.append("%% generator cost data")
        lines.append("%\t1\tstartup\tshutdown\tn\tx1\ty1\t...\txn\tyn")
        lines.append("%\t2\tstartup\tshutdown\tn\tc(n-1)\t...\tc0")
        lines.append(f"{prefix}gencost = [")
        n1 = 0
        if np.any(gencost[:, MODEL - 1] == PW_LINEAR):
            n1 = int(2 * np.max(gencost[gencost[:, MODEL - 1] == PW_LINEAR, NCOST - 1]))
        n2 = 0
        if np.any(gencost[:, MODEL - 1] == POLYNOMIAL):
            n2 = int(np.max(gencost[gencost[:, MODEL - 1] == POLYNOMIAL, NCOST - 1]))
        n = max(n1, n2)
        if gencost.shape[1] < n + 4:
            raise ValueError("savecase: gencost data claims it has more columns than it does")
        for row in gencost[:, : n + 4]:
            pieces = [f"{int(round(row[0]))}", f"{row[1]:.9g}", f"{row[2]:.9g}", f"{int(round(row[3]))}"]
            pieces.extend(f"{value:.9g}" for value in row[4 : n + 4])
            lines.append("\t" + "\t".join(pieces) + ";")
        lines.append("];")

    if ("A" in mpc and np.asarray(mpc["A"]).size) or ("N" in mpc and np.asarray(mpc["N"]).size):
        lines.append("")
        lines.append("%%-----  Generalized OPF User Data  -----%%")
    if "A" in mpc and np.asarray(mpc["A"]).size:
        lines.append("")
        lines.append("%% user constraints")
        lines.extend(_print_sparse_lines(f"{prefix}A", mpc["A"]))
        if "l" in mpc and np.asarray(mpc["l"]).size and "u" in mpc and np.asarray(mpc["u"]).size:
            lines.append("lbub = [")
            for lo, hi in zip(np.asarray(mpc["l"]).reshape(-1), np.asarray(mpc["u"]).reshape(-1)):
                lines.append(f"\t{float(lo):.9g}\t{float(hi):.9g};")
            lines.append("];")
            lines.append(f"{prefix}l = lbub(:, 1);")
            lines.append(f"{prefix}u = lbub(:, 2);")
            lines.append("")
        elif "l" in mpc and np.asarray(mpc["l"]).size:
            lines.append(f"{prefix}l = [")
            for value in np.asarray(mpc["l"]).reshape(-1):
                lines.append(f"\t{float(value):.9g};")
            lines.append("];")
            lines.append("")
        elif "u" in mpc and np.asarray(mpc["u"]).size:
            lines.append(f"{prefix}u = [")
            for value in np.asarray(mpc["u"]).reshape(-1):
                lines.append(f"\t{float(value):.9g};")
            lines.append("];")
    if "N" in mpc and np.asarray(mpc["N"]).size:
        lines.append("")
        lines.append("%% user costs")
        lines.extend(_print_sparse_lines(f"{prefix}N", mpc["N"]))
        if "H" in mpc and np.asarray(mpc["H"]).size:
            lines.extend(_print_sparse_lines(f"{prefix}H", mpc["H"]))
        if "fparm" in mpc and np.asarray(mpc["fparm"]).size:
            lines.append("Cw_fparm = [")
            for cw, fp in zip(np.asarray(mpc["Cw"]).reshape(-1), _as_array(mpc["fparm"])):
                lines.append(
                    "\t"
                    + "\t".join(
                        [f"{float(cw):.9g}"]
                        + [f"{float(v):.9g}" if i != 1 else f"{int(round(float(v)))}" for i, v in enumerate(fp)]
                    )
                    + ";"
                )
            lines.append("];")
            lines.append(f"{prefix}Cw    = Cw_fparm(:, 1);")
            lines.append(f"{prefix}fparm = Cw_fparm(:, 2:5);")
        elif "Cw" in mpc and np.asarray(mpc["Cw"]).size:
            lines.append(f"{prefix}Cw = [")
            for value in np.asarray(mpc["Cw"]).reshape(-1):
                lines.append(f"\t{float(value):.9g};")
            lines.append("];")
    if any(key in mpc and np.asarray(mpc[key]).size for key in ("z0", "zl", "zu")):
        lines.append("")
        lines.append("%% user vars")
        for key in ("z0", "zl", "zu"):
            if key in mpc and np.asarray(mpc[key]).size:
                lines.append(f"{prefix}{key} = [")
                for value in np.asarray(mpc[key]).reshape(-1):
                    lines.append(f"\t{float(value):.9g};")
                lines.append("];")
    for field, title in (
        ("gentype", "%% generator unit type (see GENTYPES)"),
        ("genfuel", "%% generator fuel type (see GENFUELS)"),
        ("bus_name", "%% bus names"),
    ):
        values = _cellstr(mpc[field]) if field in mpc else []
        if values:
            lines.append("")
            lines.append(title)
            lines.append(f"{prefix}{field} = {{")
            for item in values:
                escaped = item.replace("'", "''")
                lines.append(f"\t'{escaped}';")
            lines.append("};")

    if "userfcn" in mpc:
        run_userfcn(mpc["userfcn"], "savecase", mpc, lines, prefix)

    Path(fname_out).write_text("\n".join(lines) + "\n", encoding="ascii")
    return fname_out if nargout == 1 else None
