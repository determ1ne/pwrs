# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy

import numpy as np

from .idx_brch import ANGMAX, ANGMIN, BR_B, BR_R, BR_STATUS, BR_X, F_BUS, RATE_A, RATE_B, RATE_C, SHIFT, T_BUS, TAP
from .idx_bus import BS, BUS_AREA, BUS_I, GS, PD, QD, VMAX, VMIN
from .idx_ct import (
    CT_ADD,
    CT_CHGTYPE,
    CT_COL,
    CT_LABEL,
    CT_LOAD_ALL_P,
    CT_LOAD_ALL_PQ,
    CT_LOAD_DIS_P,
    CT_LOAD_DIS_PQ,
    CT_LOAD_FIX_P,
    CT_LOAD_FIX_PQ,
    CT_MODCOST_F,
    CT_MODCOST_X,
    CT_NEWVAL,
    CT_REL,
    CT_REP,
    CT_ROW,
    CT_TABLE,
    CT_TAREABRCH,
    CT_TAREABUS,
    CT_TAREAGEN,
    CT_TAREAGENCOST,
    CT_TAREALOAD,
    CT_TBRCH,
    CT_TBUS,
    CT_TGEN,
    CT_TGENCOST,
    CT_TLOAD,
)
from .idx_gen import (
    APF,
    GEN_BUS,
    GEN_STATUS,
    PC1,
    PC2,
    PMAX,
    PMIN,
    QC1MAX,
    QC1MIN,
    QC2MAX,
    QC2MIN,
    QMAX,
    QMIN,
    RAMP_10,
    RAMP_30,
    RAMP_AGC,
    RAMP_Q,
)
from .modcost import modcost
from .scale_load import scale_load
from .total_load import total_load


def _modify_vector(matrix: np.ndarray, row: int, col: int, typ: int, val: float, message: str) -> np.ndarray:
    col_idx = col - 1
    if row == 0:
        if typ == CT_REP:
            matrix[:, col_idx] = val * np.ones(matrix.shape[0])
        elif typ == CT_REL:
            matrix[:, col_idx] = val * matrix[:, col_idx]
        elif typ == CT_ADD:
            matrix[:, col_idx] = val + matrix[:, col_idx]
        else:
            raise ValueError(message.format(typ))
    else:
        row_idx = row - 1
        if typ == CT_REP:
            matrix[row_idx, col_idx] = val
        elif typ == CT_REL:
            matrix[row_idx, col_idx] = val * matrix[row_idx, col_idx]
        elif typ == CT_ADD:
            matrix[row_idx, col_idx] = val + matrix[row_idx, col_idx]
        else:
            raise ValueError(message.format(typ))
    return matrix


def apply_changes(label, mpc, chgtab, *, nargout=None):
    """Apply a labeled change set from a MATPOWER change table.

    Mirrors MATPOWER's ``apply_changes`` helper by selecting all rows in
    ``chgtab`` matching ``label`` and applying the corresponding table,
    area-wide, load, and cost modifications to the case struct ``mpc``.

    Parameters
    ----------
    label : float or array_like
        Change-table label identifying the rows to apply.
    mpc : dict
        MATPOWER case struct to modify.
    chgtab : array_like
        MATPOWER change table.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    dict
        Updated MATPOWER case struct with the selected changes applied.
    """
    label = float(np.asarray(label).reshape(-1)[0])
    mpc = copy.deepcopy(mpc)
    chgtab = np.atleast_2d(np.asarray(chgtab, dtype=float))

    kk = np.flatnonzero(label == chgtab[:, CT_LABEL - 1])
    if kk.size == 0:
        raise ValueError(f"apply_changes: LABEL {label:g} not found in CHGTAB")

    i2e = mpc["bus"][:, BUS_I - 1].astype(int)
    e2i = np.zeros(int(np.max(i2e)) + 1, dtype=int)
    e2i[i2e] = np.arange(mpc["bus"].shape[0])

    for k in kk:
        tbl = int(chgtab[k, CT_TABLE - 1])
        row = int(chgtab[k, CT_ROW - 1])
        col = int(chgtab[k, CT_COL - 1])
        typ = int(chgtab[k, CT_CHGTYPE - 1])
        val = float(chgtab[k, CT_NEWVAL - 1])

        if tbl == CT_TBUS:
            if col not in [PD, QD, GS, BS, VMAX, VMIN]:
                raise ValueError(f"apply_changes: modification to column {col} of bus table not supported")
            mpc["bus"] = _modify_vector(
                mpc["bus"], row, col, typ, val, "apply_changes: unsupported modification type {} for bus table"
            )
        elif tbl == CT_TBRCH:
            if col not in [BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX]:
                raise ValueError(f"apply_changes: modification to column {col} of branch table not supported")
            mpc["branch"] = _modify_vector(
                mpc["branch"], row, col, typ, val, "apply_changes: unsupported modification type {} for branch table"
            )
        elif tbl == CT_TGEN:
            if col not in [
                QMAX,
                QMIN,
                GEN_STATUS,
                PMAX,
                PMIN,
                PC1,
                PC2,
                QC1MIN,
                QC1MAX,
                QC2MIN,
                QC2MAX,
                RAMP_AGC,
                RAMP_10,
                RAMP_30,
                RAMP_Q,
                APF,
            ]:
                raise ValueError(f"apply_changes: modification to column {col} of gen table not supported")
            mpc["gen"] = _modify_vector(
                mpc["gen"], row, col, typ, val, "apply_changes: unsupported modification type {} for gen table"
            )
        elif tbl == CT_TGENCOST:
            if col in [CT_MODCOST_F, CT_MODCOST_X]:
                if typ == CT_REL:
                    modcost_type = "SCALE_F" if col == CT_MODCOST_F else "SCALE_X"
                elif typ == CT_ADD:
                    modcost_type = "SHIFT_F" if col == CT_MODCOST_F else "SHIFT_X"
                else:
                    raise ValueError(
                        f"apply_changes: unsupported modification type {typ} for gencost table CT_MODCOST_F/X modification"
                    )
                if row == 0:
                    mpc["gencost"] = modcost(mpc["gencost"], val, modcost_type, nargout=1)
                else:
                    row_idx = row - 1
                    mpc["gencost"][row_idx : row_idx + 1, :] = modcost(
                        mpc["gencost"][row_idx : row_idx + 1, :], val, modcost_type, nargout=1
                    )
            else:
                if col < 1 or int(col) != col:
                    raise ValueError(f"apply_changes: modification to column {col} of gencost table not supported")
                mpc["gencost"] = _modify_vector(
                    mpc["gencost"],
                    row,
                    col,
                    typ,
                    val,
                    "apply_changes: unsupported modification type {} for gencost table",
                )
        elif tbl == CT_TAREABUS:
            if col not in [PD, QD, GS, BS, VMAX, VMIN]:
                raise ValueError(f"apply_changes: area-wide modification to column {col} of bus table not supported")
            jj = np.flatnonzero(mpc["bus"][:, BUS_AREA - 1] == row)
            if typ == CT_REP:
                mpc["bus"][jj, col - 1] = val * np.ones(jj.size)
            elif typ == CT_REL:
                mpc["bus"][jj, col - 1] = val * mpc["bus"][jj, col - 1]
            elif typ == CT_ADD:
                mpc["bus"][jj, col - 1] = val + mpc["bus"][jj, col - 1]
            else:
                raise ValueError(f"apply_changes: unsupported area-wide modification type {typ} for bus table")
        elif tbl == CT_TAREABRCH:
            if col not in [BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX]:
                raise ValueError(f"apply_changes: area-wide modification to column {col} of branch table not supported")
            f_area = mpc["bus"][e2i[mpc["branch"][:, F_BUS - 1].astype(int)], BUS_AREA - 1]
            t_area = mpc["bus"][e2i[mpc["branch"][:, T_BUS - 1].astype(int)], BUS_AREA - 1]
            jj = np.flatnonzero((row == f_area) | (row == t_area))
            if typ == CT_REP:
                mpc["branch"][jj, col - 1] = val * np.ones(jj.size)
            elif typ == CT_REL:
                mpc["branch"][jj, col - 1] = val * mpc["branch"][jj, col - 1]
            elif typ == CT_ADD:
                mpc["branch"][jj, col - 1] = val + mpc["branch"][jj, col - 1]
            else:
                raise ValueError(f"apply_changes: unsupported area-wide modification type {typ} for branch table")
        elif tbl == CT_TAREAGEN:
            if col not in [
                QMAX,
                QMIN,
                GEN_STATUS,
                PMAX,
                PMIN,
                PC1,
                PC2,
                QC1MIN,
                QC1MAX,
                QC2MIN,
                QC2MAX,
                RAMP_AGC,
                RAMP_10,
                RAMP_30,
                RAMP_Q,
                APF,
            ]:
                raise ValueError(f"apply_changes: area-wide modification to column {col} of gen table not supported")
            jj = np.flatnonzero(row == mpc["bus"][e2i[mpc["gen"][:, GEN_BUS - 1].astype(int)], BUS_AREA - 1])
            if typ == CT_REP:
                mpc["gen"][jj, col - 1] = val * np.ones(jj.size)
            elif typ == CT_REL:
                mpc["gen"][jj, col - 1] = val * mpc["gen"][jj, col - 1]
            elif typ == CT_ADD:
                mpc["gen"][jj, col - 1] = val + mpc["gen"][jj, col - 1]
            else:
                raise ValueError(f"apply_changes: unsupported area-wide modification type {typ} for gen table")
        elif tbl == CT_TAREAGENCOST:
            jj = np.flatnonzero(row == mpc["bus"][e2i[mpc["gen"][:, GEN_BUS - 1].astype(int)], BUS_AREA - 1])
            if col in [CT_MODCOST_F, CT_MODCOST_X]:
                if typ == CT_REL:
                    modcost_type = "SCALE_F" if col == CT_MODCOST_F else "SCALE_X"
                elif typ == CT_ADD:
                    modcost_type = "SHIFT_F" if col == CT_MODCOST_F else "SHIFT_X"
                else:
                    raise ValueError(
                        f"apply_changes: unsupported area-wide modification type {typ} for gencost table CT_MODCOST_F/X modification"
                    )
                if jj.size:
                    mpc["gencost"][jj, :] = modcost(mpc["gencost"][jj, :], val, modcost_type, nargout=1)
            else:
                if col < 1 or int(col) != col:
                    raise ValueError(
                        f"apply_changes: area-wide modification to column {col} of gencost table not supported"
                    )
                if typ == CT_REP:
                    mpc["gencost"][jj, col - 1] = val * np.ones(jj.size)
                elif typ == CT_REL:
                    mpc["gencost"][jj, col - 1] = val * mpc["gencost"][jj, col - 1]
                elif typ == CT_ADD:
                    mpc["gencost"][jj, col - 1] = val + mpc["gencost"][jj, col - 1]
                else:
                    raise ValueError(f"apply_changes: unsupported area-wide modification type {typ} for gencost table")
        elif tbl in [CT_TLOAD, CT_TAREALOAD]:
            if abs(col) not in list(range(CT_LOAD_ALL_PQ, CT_LOAD_DIS_P + 1)):
                raise ValueError(f"apply_changes: column={col} for load modifications is not supported")
            opt = {}
            if abs(col) in [CT_LOAD_ALL_PQ, CT_LOAD_FIX_PQ, CT_LOAD_DIS_PQ]:
                opt["pq"] = "PQ"
            else:
                opt["pq"] = "P"
            if abs(col) in [CT_LOAD_ALL_PQ, CT_LOAD_ALL_P]:
                opt["which"] = "BOTH"
            elif abs(col) in [CT_LOAD_FIX_PQ, CT_LOAD_FIX_P]:
                opt["which"] = "FIXED"
            else:
                opt["which"] = "DISPATCHABLE"

            if tbl == CT_TLOAD:
                nb = mpc["bus"].shape[0]
                if row == 0:
                    load_zone = np.ones((nb, 1))
                else:
                    load_zone = np.zeros((nb, 1))
                    load_zone[row - 1] = 1
            else:
                load_zone = (mpc["bus"][:, BUS_AREA - 1] == row).astype(float).reshape(-1, 1)

            if typ == CT_REP:
                opt["scale"] = "QUANTITY"
                dmd = val
            elif typ == CT_REL:
                opt["scale"] = "FACTOR"
                dmd = val
            elif typ == CT_ADD:
                opt["scale"] = "QUANTITY"
                old_val = float(np.asarray(total_load(mpc, load_zone, nargout=1)).reshape(-1)[0])
                dmd = old_val + val
            else:
                raise ValueError(f"apply_changes: unsupported modification type {typ} for loads")

            if col < 0:
                mpc["bus"], mpc["gen"], mpc["gencost"] = scale_load(
                    dmd, mpc["bus"], mpc["gen"], load_zone, opt, mpc["gencost"], nargout=3
                )
            else:
                mpc["bus"], mpc["gen"] = scale_load(dmd, mpc["bus"], mpc["gen"], load_zone, opt, nargout=2)
        else:
            raise ValueError("apply_changes: CHGTAB attempts to modify unsupported table type")

    return mpc
