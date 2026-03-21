# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy

import numpy as np

from .idx_brch import F_BUS, T_BUS
from .idx_bus import BUS_I, BUS_TYPE, REF
from .idx_gen import GEN_BUS


def _first_occurrence_indices(values, mask):
    selected = values[mask]
    if selected.size == 0:
        return np.array([], dtype=int)
    unique_vals = np.unique(selected)
    out = []
    for val in unique_vals:
        out.append(np.flatnonzero(mask & (values == val))[0])
    return np.asarray(out, dtype=int)


def order_radial(mpc, *, nargout=None):
    """Order a radial network for backward/forward sweep methods.

    Mirrors MATPOWER's ``order_radial`` helper. It derives bus and branch
    orderings for radial power flow algorithms, detects loops and
    disconnected branches, and stores the ordering metadata back into the
    case struct.

    Parameters
    ----------
    mpc : dict
        MATPOWER case struct.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    dict
        Updated MATPOWER case struct with radial ordering metadata.
    """
    mpc = copy.deepcopy(mpc)
    bus = np.atleast_2d(np.array(mpc["bus"], dtype=float, copy=True))
    branch = np.atleast_2d(np.array(mpc["branch"], dtype=float, copy=True))
    gen = np.atleast_2d(np.array(mpc["gen"], dtype=float, copy=True))

    slack = np.asarray(bus[bus[:, BUS_TYPE - 1] == REF, BUS_I - 1], dtype=int).reshape(-1)
    f = branch[:, F_BUS - 1].astype(int).copy()
    t = branch[:, T_BUS - 1].astype(int).copy()
    nl = branch.shape[0]
    branch_number = np.arange(1, nl + 1, dtype=int)

    branch_order = []
    loops = []
    bus_order = slack.tolist()
    iter_count = 1

    while f.size and iter_count <= nl:
        bus_order_arr = np.asarray(bus_order, dtype=int)
        mf = np.isin(f, bus_order_arr)
        mt = np.isin(t, bus_order_arr)
        is_loop = mf & mt
        if np.any(is_loop):
            loops.extend(branch_number[is_loop].tolist())
            mf = mf[~is_loop]
            f = f[~is_loop]
            t = t[~is_loop]
            branch_number = branch_number[~is_loop]
        if np.any(mf):
            idx = _first_occurrence_indices(t, mf)
            bus_order.extend(t[idx].tolist())
            branch_order.extend(branch_number[idx].tolist())
            keep = np.ones(f.shape[0], dtype=bool)
            keep[idx] = False
            mf = mf[keep]
            f = f[keep]
            t = t[keep]
            branch_number = branch_number[keep]
        if np.any(mf):
            loops.extend(branch_number[mf].tolist())
            f = f[~mf]
            t = t[~mf]
            branch_number = branch_number[~mf]

        bus_order_arr = np.asarray(bus_order, dtype=int)
        mf = np.isin(f, bus_order_arr)
        mt = np.isin(t, bus_order_arr)
        is_loop = mf & mt
        if np.any(is_loop):
            loops.extend(branch_number[is_loop].tolist())
            mt = mt[~is_loop]
            f = f[~is_loop]
            t = t[~is_loop]
            branch_number = branch_number[~is_loop]
        if np.any(mt):
            idx = _first_occurrence_indices(f, mt)
            bus_order.extend(f[idx].tolist())
            branch_order.extend(branch_number[idx].tolist())
            keep = np.ones(f.shape[0], dtype=bool)
            keep[idx] = False
            mt = mt[keep]
            f = f[keep]
            t = t[keep]
            branch_number = branch_number[keep]
        if np.any(mt):
            loops.extend(branch_number[mt].tolist())
            f = f[~mt]
            t = t[~mt]
            branch_number = branch_number[~mt]

        iter_count += 1

    if f.size:
        not_connected = branch_number.copy()
    else:
        not_connected = np.array([], dtype=int)

    mpc["branch_order"] = np.asarray(branch_order, dtype=float).reshape(-1, 1)
    mpc["loop"] = np.asarray(loops, dtype=float).reshape(-1, 1)
    mpc["bus_order"] = np.asarray(bus_order, dtype=float).reshape(-1, 1)
    mpc["not_connected"] = np.asarray(not_connected, dtype=float).reshape(-1, 1)

    if len(loops) == 0:
        mpc["branch"] = branch[np.asarray(branch_order, dtype=int) - 1, :]
        bus_order_arr = np.asarray(bus_order, dtype=int)
        bus_order_inv = np.zeros(int(np.max(bus_order_arr)) + 1, dtype=int)
        bus_order_inv[bus_order_arr] = np.arange(1, nl + 2, dtype=int)

        f = mpc["branch"][:, F_BUS - 1].astype(int)
        t = mpc["branch"][:, T_BUS - 1].astype(int)
        f = bus_order_inv[f]
        t = bus_order_inv[t]
        br_reverse = f > t
        tmp = f[br_reverse].copy()
        f[br_reverse] = t[br_reverse]
        t[br_reverse] = tmp
        mpc["branch"][:, [F_BUS - 1, T_BUS - 1]] = np.column_stack([f, t])

        branch_order_arr = np.asarray(branch_order, dtype=int)
        branch_order_inv = np.zeros(branch.shape[0] + 1, dtype=int)
        branch_order_inv[branch_order_arr] = np.arange(1, branch.shape[0] + 1, dtype=int)

        mpc["bus"] = bus[bus_order_arr - 1, :]
        mpc["bus"][:, BUS_I - 1] = bus_order_inv[mpc["bus"][:, BUS_I - 1].astype(int)]
        mpc["gen"] = gen
        mpc["gen"][:, GEN_BUS - 1] = bus_order_inv[mpc["gen"][:, GEN_BUS - 1].astype(int)]
        mpc["bus_order_inv"] = bus_order_inv.reshape(-1, 1)
        mpc["branch_order_inv"] = branch_order_inv.reshape(-1, 1)
        mpc["br_reverse"] = br_reverse.reshape(-1, 1).astype(float)

    return mpc
