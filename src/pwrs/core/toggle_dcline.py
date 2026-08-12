# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy
from typing import Any

import numpy as np
from scipy import sparse

from ..corex import MatpowerConfig
from ..utils import map_e2i
from .add_userfcn import add_userfcn
from .idx_bus import BUS_TYPE, PV, REF
from .idx_cost import COST, MODEL, NCOST, POLYNOMIAL
from .idx_dcline import (
    BR_STATUS,
    F_BUS,
    LOSS0,
    LOSS1,
    MU_PMAX,
    MU_PMIN,
    MU_QMAXF,
    MU_QMAXT,
    MU_QMINF,
    MU_QMINT,
    PF,
    PMAX,
    PMIN,
    PT,
    QF,
    QMAXF,
    QMAXT,
    QMINF,
    QMINT,
    QT,
    T_BUS,
    VF,
    VT,
    idx_dcline,
)
from .idx_gen import GEN_BUS, GEN_STATUS, MBASE, MU_QMAX, MU_QMIN, PG, QG, QMAX, QMIN, VG
from .idx_gen import MU_PMAX as G_MU_PMAX
from .idx_gen import MU_PMIN as G_MU_PMIN
from .idx_gen import PMAX as G_PMAX
from .idx_gen import PMIN as G_PMIN
from .isload import isload
from .mpoption import mpoption
from .pqcost import pqcost
from .remove_userfcn import remove_userfcn


def _as_array(value: Any, *, dtype=float) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return np.array(arr, copy=True)


def _column(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).reshape(-1, 1)


def _pad_cols(value: Any, ncols: int) -> np.ndarray:
    arr = _as_array(value, dtype=float)
    if arr.shape[1] >= ncols:
        return arr
    return np.hstack([arr, np.zeros((arr.shape[0], ncols - arr.shape[1]), dtype=arr.dtype)])


def _isempty(value: Any) -> bool:
    if sparse.issparse(value):
        return value.shape[0] == 0 or value.shape[1] == 0
    return np.asarray(value).size == 0


def _zero_block(rows: int, cols: int, template: Any):
    if sparse.issparse(template):
        return sparse.csc_matrix((rows, cols), dtype=template.dtype)
    return np.zeros((rows, cols), dtype=np.asarray(template).dtype)


def _hstack(parts: list[Any], template: Any):
    if sparse.issparse(template):
        return sparse.hstack(parts, format="csc")
    return np.hstack(parts)


def _ensure_status(mpc: dict[str, Any]) -> None:
    if "userfcn" not in mpc:
        mpc["userfcn"] = {}
    if "status" not in mpc["userfcn"] or not isinstance(mpc["userfcn"]["status"], dict):
        mpc["userfcn"]["status"] = {}


def toggle_dcline(mpc, on_off):
    """Enable, disable or query DC line modeling callbacks.

    Parameters
    ----------
    mpc : dict
        MATPOWER case dict. When enabling the extension, it must contain a
        ``"dcline"`` field whose columns follow :mod:`idx_dcline`.
    on_off : {"on", "off", "status"}
        Requested mode. ``"on"`` registers the userfcn callbacks,
        ``"off"`` removes them, and ``"status"`` returns the current state.

    Returns
    -------
    dict or int
        Returns the updated case dict for ``"on"`` and ``"off"``. Returns a
        status flag for ``"status"``.

    Notes
    -----
    The DC line extension models each in-service DC line as a linked pair of
    generators. Although implemented via the OPF userfcn extension
    mechanism, it is used by both simple power flow and OPF workflows.

    See Also
    --------
    idx_dcline, add_userfcn, remove_userfcn, run_userfcn
    """
    mpc = copy.deepcopy(mpc)
    mode = str(on_off).upper()

    if mode == "ON":
        c = idx_dcline()
        if "dcline" not in mpc:
            raise ValueError(
                f"toggle_dcline: case must contain a 'dcline' field, an ndc x {c['LOSS1'] + 1} matrix."
            )
        mpc["dcline"] = _as_array(mpc["dcline"], dtype=float)
        if mpc["dcline"].shape[1] <= c["LOSS1"]:
            raise ValueError(
                f"toggle_dcline: case must contain a 'dcline' field, an ndc x {c['LOSS1'] + 1} matrix."
            )
        if "dclinecost" in mpc:
            mpc["dclinecost"] = _as_array(mpc["dclinecost"], dtype=float)
            if mpc["dcline"].shape[0] != mpc["dclinecost"].shape[0]:
                raise ValueError(
                    "toggle_dcline: number of rows in 'dcline' field (%d) and 'dclinecost' field (%d) do not match."
                    % (mpc["dcline"].shape[0], mpc["dclinecost"].shape[0])
                )

        l0 = mpc["dcline"][:, c["LOSS0"]]
        l1 = mpc["dcline"][:, c["LOSS1"]]
        k = np.flatnonzero(
            (l0 + l1 * mpc["dcline"][:, c["PMIN"]] < 0) | (l0 + l1 * mpc["dcline"][:, c["PMAX"]] < 0)
        )
        if k.size:
            buses = mpc["dcline"][k[0], c["F_BUS"] : c["T_BUS"] + 1]
            print(
                f"warning: toggle_dcline: loss can be negative for DC line from bus {int(buses[0])} to {int(buses[1])}"
            )

        mpc = add_userfcn(mpc, "ext2int", userfcn_dcline_ext2int)
        mpc = add_userfcn(mpc, "formulation", userfcn_dcline_formulation)
        mpc = add_userfcn(mpc, "int2ext", userfcn_dcline_int2ext)
        mpc = add_userfcn(mpc, "printpf", userfcn_dcline_printpf)
        mpc = add_userfcn(mpc, "savecase", userfcn_dcline_savecase)
        _ensure_status(mpc)
        mpc["userfcn"]["status"]["dcline"] = 1
        return mpc

    if mode == "OFF":
        mpc = remove_userfcn(mpc, "savecase", userfcn_dcline_savecase)
        mpc = remove_userfcn(mpc, "printpf", userfcn_dcline_printpf)
        mpc = remove_userfcn(mpc, "int2ext", userfcn_dcline_int2ext)
        mpc = remove_userfcn(mpc, "formulation", userfcn_dcline_formulation)
        mpc = remove_userfcn(mpc, "ext2int", userfcn_dcline_ext2int)
        _ensure_status(mpc)
        mpc["userfcn"]["status"]["dcline"] = 0
        return mpc

    if mode == "STATUS":
        return int(bool(mpc.get("userfcn", {}).get("status", {}).get("dcline", 0)))

    raise ValueError("toggle_dcline: 2nd argument must be 'on', 'off' or 'status'")


def userfcn_dcline_ext2int(mpc, mpopt, args):
    havecost = "dclinecost" in mpc
    mpc["dcline"] = _as_array(mpc["dcline"], dtype=float)
    if havecost:
        mpc["dclinecost"] = _as_array(mpc["dclinecost"], dtype=float)

    mpc["order"]["ext"]["dcline"] = np.array(mpc["dcline"], copy=True)
    if havecost:
        mpc["order"]["ext"]["dclinecost"] = np.array(mpc["dclinecost"], copy=True)

    on = np.flatnonzero(mpc["dcline"][:, BR_STATUS] > 0) + 1
    off = np.flatnonzero(mpc["dcline"][:, BR_STATUS] <= 0) + 1
    mpc["order"]["dcline"] = {"status": {"on": _column(on), "off": _column(off)}}

    dc = np.array(mpc["dcline"][on - 1, :], copy=True)
    if havecost:
        dcc = np.array(mpc["dclinecost"][on - 1, :], copy=True)
        mpc["dclinecost"] = dcc
    else:
        dcc = np.array([])
    ndc = dc.shape[0]
    e2i = mpc["order"]["bus"]["e2i"]
    dc[:, F_BUS] = map_e2i(e2i, dc[:, F_BUS])
    dc[:, T_BUS] = map_e2i(e2i, dc[:, T_BUS])
    mpc["dcline"] = dc

    fg = np.zeros((ndc, mpc["gen"].shape[1]), dtype=float)
    fg[:, MBASE] = 100.0
    fg[:, GEN_STATUS] = dc[:, BR_STATUS]
    fg[:, G_PMIN] = -np.inf
    fg[:, G_PMAX] = np.inf
    tg = np.array(fg, copy=True)
    fg[:, GEN_BUS] = dc[:, F_BUS]
    tg[:, GEN_BUS] = dc[:, T_BUS]
    fg[:, PG] = -dc[:, PF]
    tg[:, PG] = dc[:, PF] - (dc[:, LOSS0] + dc[:, LOSS1] * dc[:, PF])
    fg[:, QG] = dc[:, QF]
    tg[:, QG] = dc[:, QT]
    fg[:, VG] = dc[:, VF]
    tg[:, VG] = dc[:, VT]

    k = np.flatnonzero(dc[:, PMIN] >= 0)
    if k.size:
        fg[k, G_PMAX] = -dc[k, PMIN]
    k = np.flatnonzero(dc[:, PMAX] >= 0)
    if k.size:
        fg[k, G_PMIN] = -dc[k, PMAX]
    k = np.flatnonzero(dc[:, PMIN] < 0)
    if k.size:
        tg[k, G_PMIN] = dc[k, PMIN]
    k = np.flatnonzero(dc[:, PMAX] < 0)
    if k.size:
        tg[k, G_PMAX] = dc[k, PMAX]

    fg[:, QMIN] = dc[:, QMINF]
    fg[:, QMAX] = dc[:, QMAXF]
    tg[:, QMIN] = dc[:, QMINT]
    tg[:, QMAX] = dc[:, QMAXT]

    fg[isload(fg), G_PMAX] = -1e-6
    tg[isload(tg), G_PMAX] = -1e-6

    refbus = np.flatnonzero(mpc["bus"][:, BUS_TYPE] == REF)
    mpc["bus"][np.asarray(dc[:, F_BUS], dtype=int) - 1, BUS_TYPE] = PV
    mpc["bus"][np.asarray(dc[:, T_BUS], dtype=int) - 1, BUS_TYPE] = PV
    if refbus.size:
        mpc["bus"][refbus, BUS_TYPE] = REF

    nb = mpc["bus"].shape[0]
    ng = mpc["gen"].shape[0]
    if "A" in mpc and not _isempty(mpc["A"]):
        mA, nA = mpc["A"].shape
        z = _zero_block(mA, 2 * ndc, mpc["A"])
        if nA >= 2 * nb + 2 * ng:
            mpc["A"] = _hstack(
                [
                    mpc["A"][:, : 2 * nb + ng],
                    z,
                    mpc["A"][:, 2 * nb + ng : 2 * nb + 2 * ng],
                    z,
                    mpc["A"][:, 2 * nb + 2 * ng :],
                ],
                mpc["A"],
            )
        else:
            mpc["A"] = _hstack([mpc["A"][:, : nb + ng], z, mpc["A"][:, nb + ng :]], mpc["A"])
    if "N" in mpc and not _isempty(mpc["N"]):
        mN, nN = mpc["N"].shape
        z = _zero_block(mN, 2 * ndc, mpc["N"])
        if nN >= 2 * nb + 2 * ng:
            mpc["N"] = _hstack(
                [
                    mpc["N"][:, : 2 * nb + ng],
                    z,
                    mpc["N"][:, 2 * nb + ng : 2 * nb + 2 * ng],
                    z,
                    mpc["N"][:, 2 * nb + 2 * ng :],
                ],
                mpc["N"],
            )
        else:
            mpc["N"] = _hstack([mpc["N"][:, : nb + ng], z, mpc["N"][:, nb + ng :]], mpc["N"])

    mpc["gen"] = np.vstack([mpc["gen"], fg, tg])

    if "gencost" in mpc and not _isempty(mpc["gencost"]):
        ngcc = mpc["gencost"].shape[1]
        pc, qc = pqcost(mpc["gencost"], ng)
        if havecost:
            ccc = max(ngcc, dcc.shape[1])
            if ccc > ngcc:
                pc = np.hstack([pc, np.zeros((pc.shape[0], ccc - ngcc))])
                if np.asarray(qc).size:
                    qc = np.hstack([qc, np.zeros((qc.shape[0], ccc - ngcc))])
                ngcc = ccc
            for k in range(ndc):
                if int(dcc[k, MODEL]) == POLYNOMIAL:
                    nc = int(dcc[k, NCOST])
                    temp = np.array(dcc[k, COST : COST + nc], copy=True)
                    for j in range(nc - 2, -1, -2):
                        temp[j] = -temp[j]
                else:
                    nc = int(dcc[k, NCOST])
                    temp = np.array(dcc[k, COST : COST + 2 * nc], copy=True)
                    xx = -temp[0 : 2 * nc : 2]
                    yy = temp[1 : 2 * nc : 2]
                    temp[0 : 2 * nc : 2] = xx[::-1]
                    temp[1 : 2 * nc : 2] = yy[::-1]
                gck = np.r_[dcc[k, :COST], temp, np.zeros(max(0, ngcc - COST - temp.size))]
                pc = np.vstack([pc, gck])
            dcgc = np.tile(np.r_[2, 0, 0, 2, np.zeros(max(0, ngcc - 4))], (ndc, 1))
        else:
            dcgc = np.tile(np.r_[2, 0, 0, 2, np.zeros(max(0, ngcc - 4))], (ndc, 1))
            pc = np.vstack([pc, dcgc])
        pc = np.vstack([pc, dcgc])
        if np.asarray(qc).size:
            qc = np.vstack([qc, dcgc, dcgc])
            mpc["gencost"] = np.vstack([pc, qc])
        else:
            mpc["gencost"] = pc

    return mpc


def userfcn_dcline_formulation(om, mpopt, args):
    mpc = om.get_mpc()
    dc = _as_array(mpc["dcline"], dtype=float)
    ndc = dc.shape[0]
    ng = mpc["gen"].shape[0] - 2 * ndc
    nL0 = -dc[:, LOSS0] / float(np.asarray(mpc["baseMVA"]).reshape(-1)[0])
    L1 = dc[:, LOSS1]
    Adc = sparse.hstack(
        [
            sparse.csc_matrix((ndc, ng)),
            sparse.diags(1.0 - L1, 0, shape=(ndc, ndc), format="csc"),
            sparse.eye(ndc, format="csc"),
        ],
        format="csc",
    )
    om.add_lin_constraint("dcline", Adc, nL0.reshape(-1, 1), nL0.reshape(-1, 1), ["Pg"])
    return om


def userfcn_dcline_int2ext(results, mpopt, args):
    o = results["order"]
    k = np.flatnonzero(np.asarray(o["ext"]["dcline"][:, BR_STATUS]).reshape(-1) != 0)
    ndc = int(k.size)
    ng = results["gen"].shape[0] - 2 * ndc
    nb = results["bus"].shape[0]

    fg = np.array(results["gen"][ng : ng + ndc, :], copy=True)
    tg = np.array(results["gen"][ng + ndc : ng + 2 * ndc, :], copy=True)
    results["gen"] = np.array(results["gen"][:ng, :], copy=True)
    if "gencost" in results and not _isempty(results["gencost"]):
        results["gencost"] = np.array(results["gencost"][:ng, :], copy=True)

    if "A" in results and not _isempty(results["A"]):
        _, nA = results["A"].shape
        if nA >= 2 * nb + 2 * ng + 4 * ndc:
            keep = np.r_[
                0 : 2 * nb + ng, 2 * nb + ng + 2 * ndc : 2 * nb + 2 * ng + 2 * ndc, 2 * nb + 2 * ng + 4 * ndc : nA
            ]
        else:
            keep = np.r_[0 : nb + ng, nb + ng + 2 * ndc : nA]
        results["A"] = results["A"][:, keep]
    if "N" in results and not _isempty(results["N"]):
        _, nN = results["N"].shape
        if nN >= 2 * nb + 2 * ng + 4 * ndc:
            keep = np.r_[
                0 : 2 * nb + ng, 2 * nb + ng + 2 * ndc : 2 * nb + 2 * ng + 2 * ndc, 2 * nb + 2 * ng + 4 * ndc : nN
            ]
        else:
            keep = np.r_[0 : nb + ng, nb + ng + 2 * ndc : nN]
        results["N"] = results["N"][:, keep]

    need_mu = fg.shape[1] > MU_QMIN
    if need_mu:
        results["dcline"] = _pad_cols(results["dcline"], MU_QMAXT + 1)
    else:
        results["dcline"] = _as_array(results["dcline"], dtype=float)
    results["dcline"][:, PF] = -fg[:, PG]
    results["dcline"][:, PT] = tg[:, PG]
    results["dcline"][:, QF] = fg[:, QG]
    results["dcline"][:, QT] = tg[:, QG]
    results["dcline"][:, VF] = fg[:, VG]
    results["dcline"][:, VT] = tg[:, VG]
    if need_mu:
        results["dcline"][:, MU_PMIN] = fg[:, G_MU_PMAX] + tg[:, G_MU_PMIN]
        results["dcline"][:, MU_PMAX] = fg[:, G_MU_PMIN] + tg[:, G_MU_PMAX]
        results["dcline"][:, MU_QMINF] = fg[:, MU_QMIN]
        results["dcline"][:, MU_QMAXF] = fg[:, MU_QMAX]
        results["dcline"][:, MU_QMINT] = tg[:, MU_QMIN]
        results["dcline"][:, MU_QMAXT] = tg[:, MU_QMAX]

    if "int" not in results["order"] or not isinstance(results["order"]["int"], dict):
        results["order"]["int"] = {}
    results["order"]["int"]["dcline"] = np.array(results["dcline"], copy=True)
    ext_dcline = (
        _pad_cols(o["ext"]["dcline"], results["dcline"].shape[1])
        if need_mu
        else _as_array(o["ext"]["dcline"], dtype=float)
    )
    ext_dcline[k, PF : VT + 1] = results["dcline"][:, PF : VT + 1]
    if need_mu:
        ext_dcline[k, MU_PMIN : MU_QMAXT + 1] = results["dcline"][:, MU_PMIN : MU_QMAXT + 1]
    results["dcline"] = ext_dcline
    return results


def _write(fd, text: str) -> None:
    if fd == 1:
        print(text, end="")
    elif isinstance(fd, list):
        for line in text.splitlines():
            fd.append(line)
    else:
        fd.write(text)


def userfcn_dcline_printpf(results, fd, mpopt, args):
    if not isinstance(mpopt, MatpowerConfig):
        mpopt = mpoption(mpopt)
    c = idx_dcline()
    out = mpopt.out
    suppress = int(np.asarray(out.suppress_detail).reshape(-1)[0])
    if suppress == -1:
        suppress = 1 if results["bus"].shape[0] > 500 else 0
    out_all = int(np.asarray(out.all).reshape(-1)[0])
    out_branch = out_all == 1 or (
        out_all == -1 and not suppress and int(np.asarray(out.branch).reshape(-1)[0])
    )
    lim = out.lim
    if out_all == -1:
        out_all_lim = (0 if suppress else 1) * int(np.asarray(lim.all).reshape(-1)[0])
    elif out_all == 1:
        out_all_lim = 2
    else:
        out_all_lim = 0
    out_line_lim = (
        (0 if suppress else 1) * int(np.asarray(lim.line).reshape(-1)[0])
        if out_all_lim == -1
        else out_all_lim
    )
    ctol = float(np.asarray(mpopt.opf.violation).reshape(-1)[0])
    ptol = 1e-4

    dc = _as_array(results["dcline"], dtype=float)
    kk = np.flatnonzero(dc[:, c["BR_STATUS"]] != 0)
    if out_branch:
        _write(fd, "\n================================================================================")
        _write(fd, "\n|     DC Line Data                                                             |")
        _write(fd, "\n================================================================================")
        _write(fd, "\n Line    From     To        Power Flow           Loss     Reactive Inj (MVAr)")
        _write(fd, "\n   #      Bus     Bus   From (MW)   To (MW)      (MW)       From        To   ")
        _write(fd, "\n------  ------  ------  ---------  ---------  ---------  ---------  ---------")
        loss = 0.0
        for i, row in enumerate(dc, start=1):
            if row[c["BR_STATUS"]]:
                _write(
                    fd,
                    "\n%5d%8d%8d%11.2f%11.2f%11.2f%11.2f%11.2f"
                    % (
                        i,
                        int(row[c["F_BUS"]]),
                        int(row[c["T_BUS"]]),
                        row[c["PF"]],
                        row[c["PT"]],
                        row[c["PF"]] - row[c["PT"]],
                        row[c["QF"]],
                        row[c["QT"]],
                    ),
                )
                loss += row[c["PF"]] - row[c["PT"]]
            else:
                _write(
                    fd,
                    "\n%5d%8d%8d%11s%11s%11s%11s%11s"
                    % (i, int(row[0]), int(row[1]), "-  ", "-  ", "-  ", "-  ", "-  "),
                )
        _write(fd, "\n                                              ---------")
        _write(fd, "\n                                     Total:%11.2f\n" % loss)

    show_lim = out_line_lim == 2 or (
        out_line_lim == 1
        and (
            np.any(dc[kk, c["PF"]] > dc[kk, c["PMAX"]] - ctol)
            or np.any(dc[kk, c["MU_PMIN"]] > ptol)
            or np.any(dc[kk, c["MU_PMAX"]] > ptol)
        )
    )
    if show_lim:
        _write(fd, "\n================================================================================")
        _write(fd, "\n|     DC Line Constraints                                                      |")
        _write(fd, "\n================================================================================")
        _write(fd, "\n Line    From     To          Minimum        Actual Flow       Maximum")
        _write(fd, "\n   #      Bus     Bus    Pmin mu     Pmin       (MW)       Pmax      Pmax mu ")
        _write(fd, "\n------  ------  ------  ---------  ---------  ---------  ---------  ---------")
        for i, row in enumerate(dc, start=1):
            cond = out_line_lim == 2 or (
                row[c["PF"]] > row[c["PMAX"]] - ctol
                or row[c["MU_PMIN"]] > ptol
                or row[c["MU_PMAX"]] > ptol
            )
            if not cond:
                continue
            if row[c["BR_STATUS"]]:
                _write(fd, "\n%5d%8d%8d" % (i, int(row[c["F_BUS"]]), int(row[c["T_BUS"]])))
                _write(fd, "%11.3f" % row[c["MU_PMIN"]] if row[c["MU_PMIN"]] > ptol else "%11s" % "-  ")
                _write(fd, "%11.2f%11.2f%11.2f" % (row[c["PMIN"]], row[c["PF"]], row[c["PMAX"]]))
                _write(fd, "%11.3f" % row[c["MU_PMAX"]] if row[c["MU_PMAX"]] > ptol else "%11s" % "-  ")
            else:
                _write(
                    fd,
                    "\n%5d%8d%8d%11s%11s%11s%11s%11s"
                    % (i, int(row[0]), int(row[1]), "-  ", "-  ", "-  ", "-  ", "-  "),
                )
        _write(fd, "\n")
    return results


def userfcn_dcline_savecase(mpc, fd, prefix, args):
    c = idx_dcline()
    dcline = _as_array(mpc["dcline"], dtype=float)
    _write(fd, "\n%%%%-----  DC Line Data  -----%%%%")
    if dcline.shape[1] <= c["MU_QMAXT"]:
        _write(
            fd, "\n%%\tfbus\ttbus\tstatus\tPf\tPt\tQf\tQt\tVf\tVt\tPmin\tPmax\tQminF\tQmaxF\tQminT\tQmaxT\tloss0\tloss1"
        )
    else:
        _write(
            fd,
            "\n%%\tfbus\ttbus\tstatus\tPf\tPt\tQf\tQt\tVf\tVt\tPmin\tPmax\tQminF\tQmaxF\tQminT\tQmaxT\tloss0\tloss1\tmuPmin\tmuPmax\tmuQminF\tmuQmaxF\tmuQminT\tmuQmaxT",
        )
    _write(fd, f"\n{prefix}dcline = [")
    for row in dcline:
        text = (
            f"\n\t{int(round(row[0]))}\t{int(round(row[1]))}\t{int(round(row[2]))}\t"
            f"{row[3]:.9g}\t{row[4]:.9g}\t{row[5]:.9g}\t{row[6]:.9g}\t{row[7]:.9g}\t{row[8]:.9g}\t"
            f"{row[9]:.9g}\t{row[10]:.9g}\t{row[11]:.9g}\t{row[12]:.9g}\t{row[13]:.9g}\t{row[14]:.9g}\t{row[15]:.9g}\t{row[16]:.9g}"
        )
        if dcline.shape[1] > c["MU_QMAXT"]:
            text += f"\t{row[17]:.4f}\t{row[18]:.4f}\t{row[19]:.4f}\t{row[20]:.4f}\t{row[21]:.4f}\t{row[22]:.4f}"
        _write(fd, text + ";")
    _write(fd, "\n];")
    return mpc
