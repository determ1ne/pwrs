# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy

import numpy as np
from scipy import sparse

from .connected_components import connected_components_full
from .get_reorder import get_reorder
from .idx_brch import BR_STATUS, F_BUS, T_BUS
from .idx_bus import BUS_I
from .idx_dcline import idx_dcline
from .idx_gen import GEN_BUS


def _field_path(field):
    if isinstance(field, str):
        return [field]
    return [str(part) for part in field]


def _set_nested(data, path, value):
    cur = data
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value


def _get_nested(data, path):
    cur = data
    for key in path:
        cur = cur[key]
    return cur


def _has_value(value):
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    if isinstance(value, np.ndarray):
        return value.size > 0
    return True


def extract_islands(mpc, *args):
    """Extract one or more electrical islands from a MATPOWER case.

    Mirrors MATPOWER's ``extract_islands`` helper. It identifies islands from
    branch connectivity, then returns case structs restricted to selected
    islands, including optional reordering of custom fields.

    Parameters
    ----------
    mpc : dict
        MATPOWER case struct.
    *args
        Optional MATPOWER-style arguments specifying precomputed island
        groups, island indices, and custom field reorderings.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    dict or list of dict
        Extracted island case struct, or a list of island case structs when
        multiple islands are requested.
    """
    c = idx_dcline()
    nb = mpc["bus"].shape[0]
    nl = mpc["branch"].shape[0]
    ndc = mpc["dcline"].shape[0] if "dcline" in mpc else 0
    ng = mpc["gen"].shape[0]

    bus_i = np.asarray(mpc["bus"][:, BUS_I - 1], dtype=int).reshape(-1)
    e2i = np.zeros(int(np.max(bus_i)) + 1, dtype=int)
    e2i[bus_i] = np.arange(1, nb + 1, dtype=int)
    f = e2i[np.asarray(mpc["branch"][:, F_BUS - 1], dtype=int)]
    t = e2i[np.asarray(mpc["branch"][:, T_BUS - 1], dtype=int)]
    status = np.asarray(mpc["branch"][:, BR_STATUS - 1], dtype=float).reshape(-1)
    C_on = sparse.csc_matrix((-status, (np.arange(nl), f - 1)), shape=(nl, nb)) + sparse.csc_matrix(
        (status, (np.arange(nl), t - 1)), shape=(nl, nb)
    )
    C = sparse.csc_matrix((-np.ones(nl), (np.arange(nl), f - 1)), shape=(nl, nb)) + sparse.csc_matrix(
        (np.ones(nl), (np.arange(nl), t - 1)), shape=(nl, nb)
    )
    if ndc:
        fdc = e2i[np.asarray(mpc["dcline"][:, c["F_BUS"] - 1], dtype=int)]
        tdc = e2i[np.asarray(mpc["dcline"][:, c["T_BUS"] - 1], dtype=int)]
        Cdc = sparse.csc_matrix((-np.ones(ndc), (np.arange(ndc), fdc - 1)), shape=(ndc, nb)) + sparse.csc_matrix(
            (np.ones(ndc), (np.arange(ndc), tdc - 1)), shape=(ndc, nb)
        )
    else:
        Cdc = None
    gb = e2i[np.asarray(mpc["gen"][:, GEN_BUS - 1], dtype=int)]
    Cg = sparse.csc_matrix((np.ones(ng), (np.arange(ng), gb - 1)), shape=(ng, nb))

    if not C.nnz:
        return []

    n = len(args)
    if n >= 1 and isinstance(args[0], list) and len(args[0]) > 0:
        groups = args[0]
        z = 1
    else:
        groups = []
        z = 0
    k = args[z] if z < n else []
    custom = args[z + 1] if z + 1 < n else {}

    if len(groups) == 0:
        groups = connected_components_full(C_on)[0]

    k_given = not (k is None or k == [] or (isinstance(k, (list, tuple, np.ndarray)) and np.asarray(k).size == 0))
    if not k_given:
        g1 = 1
        gn = len(groups)
    else:
        if isinstance(k, str):
            if k.upper() == "ALL":
                k = np.arange(1, len(groups) + 1, dtype=int).reshape(-1, 1)
            else:
                raise ValueError(f"extract_islands: K = '{k}' is not a valid input")
        k = np.asarray(k, dtype=int).reshape(-1)
        if np.max(k) > len(groups):
            raise ValueError(
                f"extract_islands: cannot extract island {int(np.max(k))}, network has only {len(groups)} islands"
            )
        if len(k) > 1:
            tmpgroup = np.asarray(groups[k[0] - 1]).reshape(-1)
            for j in range(1, len(k)):
                tmpgroup = np.union1d(tmpgroup, np.asarray(groups[k[j] - 1]).reshape(-1))
            groups = [tmpgroup.reshape(-1, 1)]
            g1 = 1
            gn = 1
        else:
            g1 = int(k[0])
            gn = int(k[0])

    mpck = []
    orderings = ["bus", "gen", "branch", "dcline"]
    for gi in range(g1, gn + 1):
        b = np.asarray(groups[gi - 1], dtype=int).reshape(-1)
        ibr = (
            np.flatnonzero(
                (np.asarray(np.abs(C[:, b - 1]).sum(axis=1)).reshape(-1) != 0)
                & (np.asarray(C[:, b - 1].sum(axis=1)).reshape(-1) == 0)
            )
            + 1
        )
        ig = np.flatnonzero(np.asarray(Cg[:, b - 1].sum(axis=1)).reshape(-1) != 0) + 1
        if Cdc is not None:
            idc = (
                np.flatnonzero(
                    (np.asarray(np.abs(Cdc[:, b - 1]).sum(axis=1)).reshape(-1) != 0)
                    & (np.asarray(Cdc[:, b - 1].sum(axis=1)).reshape(-1) == 0)
                )
                + 1
            )
        else:
            idc = np.array([], dtype=int)

        out = copy.deepcopy(mpc)
        out["bus"] = mpc["bus"][b - 1, :]
        out["branch"] = mpc["branch"][ibr - 1, :]
        out["gen"] = mpc["gen"][ig - 1, :]
        gencost = out.get("gencost")
        source_gencost = mpc.get("gencost")
        if isinstance(gencost, np.ndarray) and isinstance(source_gencost, np.ndarray):
            if gencost.shape[0] == 2 * ng:
                out["gencost"] = source_gencost[np.r_[ig - 1, ng + ig - 1], :]
            else:
                out["gencost"] = source_gencost[ig - 1, :]
        if "gentype" in out:
            out["gentype"] = [mpc["gentype"][i - 1] for i in ig]
        if "genfuel" in out:
            out["genfuel"] = [mpc["genfuel"][i - 1] for i in ig]
        if "bus_name" in out:
            out["bus_name"] = [mpc["bus_name"][i - 1] for i in b]
        if ndc:
            out["dcline"] = mpc["dcline"][idc - 1, :]
            if "dclinecost" in out:
                out["dclinecost"] = mpc["dclinecost"][idc - 1, :]

        indices = [b.reshape(-1, 1), ig.reshape(-1, 1), ibr.reshape(-1, 1), idc.reshape(-1, 1)]
        for nidx, ord_name in enumerate(orderings):
            if ord_name in custom:
                for dim in range(len(custom[ord_name])):
                    for field in custom[ord_name][dim]:
                        path = _field_path(field)
                        value = _get_nested(out, path)
                        if _has_value(value):
                            _set_nested(out, path, get_reorder(value, indices[nidx], dim + 1))
        mpck.append(out)

    if k_given:
        return mpck[0]
    return mpck
