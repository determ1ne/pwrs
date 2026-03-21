# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from typing import Any

import numpy as np
import numpy.typing as npt

from ..corex import MatpowerCase, MatpowerConfig
from ..utils import build_e2i_map, map_e2i
from .e2i_field import e2i_field as _port_e2i_field
from .idx_brch import BR_STATUS, F_BUS, T_BUS
from .idx_bus import BUS_I, BUS_TYPE, NONE, PQ, PV, REF
from .idx_gen import GEN_BUS, GEN_STATUS
from .run_userfcn import run_userfcn


def ext2int_mpc(
    mpc: MatpowerCase,
    config: MatpowerConfig | None = None,
    reorder_gens: int = 0,
) -> MatpowerCase:
    """Convert a MATPOWER case struct from external to internal indexing.

    Parameters
    ----------
    mpc : MatpowerCase
        MATPOWER case struct in external order.
    config : MatpowerConfig or None, optional
        Configuration options for user-defined functions, by default None.
    reorder_gens : int, optional
        Whether to reorder generators by their connected bus number, by default 0 (False).

    Returns
    -------
    MatpowerCase
        MATPOWER case struct in internal order, with ``order`` metadata.
    """
    mpc = deepcopy(mpc)
    first = mpc.order is None if isinstance(mpc, MatpowerCase) else mpc.get("order") is None
    if "baseMVA" in mpc:
        mpc["baseMVA"] = mpc["baseMVA"]
    if "gencost" in mpc:
        mpc["gencost"] = np.copy(mpc["gencost"])

    if first or mpc["order"]["state"] == "e":
        if first:
            status = {
                "on": np.array([], dtype=np.int64).reshape(-1, 1),
                "off": np.array([], dtype=np.int64).reshape(-1, 1),
            }
            tmp = {
                "e2i": np.array([], dtype=np.int64).reshape(-1, 1),
                "i2e": np.array([], dtype=np.int64).reshape(-1, 1),
                "status": status,
            }
            o = {
                "ext": {"bus": [], "branch": [], "gen": []},
                "bus": deepcopy(tmp),
                "gen": deepcopy(tmp),
                "branch": {"status": deepcopy(status)},
            }
        else:
            o = deepcopy(mpc["order"])

        nb = np.shape(mpc["bus"])[0]
        ng = np.shape(mpc["gen"])[0]
        ng0 = ng
        if "A" in mpc and np.shape(mpc["A"])[1] < 2 * nb + 2 * ng:
            dc = 1
        elif "N" in mpc and np.shape(mpc["N"])[1] < 2 * nb + 2 * ng:
            dc = 1
        else:
            dc = 0

        mpc["bus"] = np.copy(mpc["bus"])
        mpc["gen"] = np.copy(mpc["gen"])
        mpc["branch"] = np.copy(mpc["branch"])

        o["ext"]["bus"] = deepcopy(mpc["bus"])
        o["ext"]["branch"] = deepcopy(mpc["branch"])
        o["ext"]["gen"] = deepcopy(mpc["gen"])

        bt = mpc["bus"][:, BUS_TYPE - 1].reshape(-1)
        err = np.flatnonzero(~((bt == PQ) | (bt == PV) | (bt == REF) | (bt == NONE)))
        if err.size:
            raise ValueError(f"ext2int: bus {int(err[0]) + 1} has an invalid BUS_TYPE")

        bus_numbers = np.asarray(mpc["bus"][:, BUS_I - 1], dtype=np.int64)
        n2i = build_e2i_map(bus_numbers, start=1)
        bs = bt != NONE
        o["bus"]["status"]["on"] = np.flatnonzero(bs) + 1
        o["bus"]["status"]["off"] = np.flatnonzero(~bs) + 1
        gen_bus_rows = map_e2i(n2i, mpc["gen"][:, GEN_BUS - 1]) - 1
        gs = (mpc["gen"][:, GEN_STATUS - 1] > 0) & bs[gen_bus_rows]
        o["gen"]["status"]["on"] = np.flatnonzero(gs) + 1
        o["gen"]["status"]["off"] = np.flatnonzero(~gs) + 1
        f_rows = map_e2i(n2i, mpc["branch"][:, F_BUS - 1]) - 1
        t_rows = map_e2i(n2i, mpc["branch"][:, T_BUS - 1]) - 1
        brs = (mpc["branch"][:, BR_STATUS - 1] != 0) & bs[f_rows] & bs[t_rows]
        o["branch"]["status"]["on"] = np.flatnonzero(brs) + 1
        o["branch"]["status"]["off"] = np.flatnonzero(~brs) + 1

        if o["bus"]["status"]["off"].size:
            mpc["bus"] = mpc["bus"][o["bus"]["status"]["on"] - 1, :]
        if o["branch"]["status"]["off"].size:
            mpc["branch"] = mpc["branch"][o["branch"]["status"]["on"] - 1, :]
        if o["gen"]["status"]["off"].size:
            mpc["gen"] = mpc["gen"][o["gen"]["status"]["on"] - 1, :]

        nb = np.shape(mpc["bus"])[0]
        ng = np.shape(mpc["gen"])[0]

        o["bus"]["i2e"] = np.array(mpc["bus"][:, BUS_I - 1], copy=True).reshape(-1, 1)
        bus_e2i = build_e2i_map(o["bus"]["i2e"], start=1)
        if nb:
            mpc["bus"][:, BUS_I - 1] = map_e2i(bus_e2i, mpc["bus"][:, BUS_I - 1])
            mpc["gen"][:, GEN_BUS - 1] = map_e2i(bus_e2i, mpc["gen"][:, GEN_BUS - 1])
            mpc["branch"][:, F_BUS - 1] = map_e2i(bus_e2i, mpc["branch"][:, F_BUS - 1])
            mpc["branch"][:, T_BUS - 1] = map_e2i(bus_e2i, mpc["branch"][:, T_BUS - 1])
        o["bus"]["e2i"] = bus_e2i

        if reorder_gens:
            order = np.argsort(mpc["gen"][:, GEN_BUS - 1], kind="stable")
            o["gen"]["i2e"] = order + 1
            e2i = np.empty(ng, dtype=np.int64)
            e2i[order] = np.arange(1, ng + 1, dtype=np.int64)
            o["gen"]["e2i"] = e2i
            mpc["gen"] = mpc["gen"][order, :]
        else:
            seq = np.arange(1, ng + 1, dtype=np.int64)
            o["gen"]["i2e"] = seq
            o["gen"]["e2i"] = seq

        if "int" in o:
            del o["int"]
        o["state"] = "i"
        mpc["order"] = o

        if "gencost" in mpc:
            ordering: Any = ["gen"]
            if np.shape(mpc["gencost"])[0] == 2 * ng0:
                ordering = ["gen", "gen"]
            mpc = _port_e2i_field(mpc, "gencost", ordering)
        if "bus_name" in mpc:
            mpc = _port_e2i_field(mpc, "bus_name", ["bus"])
        if "gentype" in mpc:
            mpc = _port_e2i_field(mpc, "gentype", ["gen"])
        if "genfuel" in mpc:
            mpc = _port_e2i_field(mpc, "genfuel", ["gen"])
        if "A" in mpc or "N" in mpc:
            ordering = ["bus", "gen"] if dc else ["bus", "bus", "gen", "gen"]
            if "A" in mpc:
                mpc = _port_e2i_field(mpc, "A", ordering, 2)
            if "N" in mpc:
                mpc = _port_e2i_field(mpc, "N", ordering, 2)
        if "userfcn" in mpc:
            mpopt = {} if config is None else config
            mpc = run_userfcn(mpc["userfcn"], "ext2int", mpc, mpopt)

    return mpc


def ext2int_old(
    bus: npt.NDArray[np.float64],
    gen: npt.NDArray[np.float64],
    branch: npt.NDArray[np.float64],
    areas: npt.NDArray[np.float64] | None = None,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64] | None,
]:
    """Convert bus, gen, branch (and optionally areas) data from external to internal indexing.

    This legacy helper function supports the old MATLAB-style call forms for
    ``ext2int`` that operate directly on the bus, gen, branch, and areas matrices.
    It is not recommended for use in new code; use `ext2int_mpc()` instead.

    The bus, gen, and branch matrices are modified in-place to convert from
    external to internal numbering.
    The returned ``i2e`` vector maps internal bus indices to external bus numbers.

    Parameters
    ----------
    bus : npt.NDArray[np.float64]
        Bus matrix.
    gen : npt.NDArray[np.float64]
        Generator matrix.
    branch : npt.NDArray[np.float64]
        Branch matrix.
    areas : npt.NDArray[np.float64] or None, optional
        Areas matrix, or None if not provided.

    Returns
    -------
    tuple
        ``(i2e, bus, gen, branch, areas)`` where ``i2e`` is a column vector mapping internal bus indices to external bus numbers, and the bus, gen, branch, and areas matrices have been modified in-place to use internal numbering.
    """
    bus = np.array(bus, copy=True)
    gen = np.array(gen, copy=True)
    branch = np.array(branch, copy=True)

    i2e = np.array(bus[:, BUS_I - 1], copy=True).reshape(-1, 1)
    e2i = build_e2i_map(i2e, start=1)

    bus[:, BUS_I - 1] = map_e2i(e2i, bus[:, BUS_I - 1])
    gen[:, GEN_BUS - 1] = map_e2i(e2i, gen[:, GEN_BUS - 1])
    branch[:, F_BUS - 1] = map_e2i(e2i, branch[:, F_BUS - 1])
    branch[:, T_BUS - 1] = map_e2i(e2i, branch[:, T_BUS - 1])
    return i2e, bus, gen, branch, areas


def ext2int(*args: Any):
    """Convert MATPOWER data from external to internal indexing.

    This Python port supports only the two primary MATPOWER forms:

    1. ``mpc = ext2int(mpc)``
    2. ``mpc = ext2int(mpc, mpopt)``
    3. ``mpc = ext2int(mpc, mpopt, reorder_gens)``
    4. ``i2e, bus, gen, branch = ext2int(bus, gen, branch)``
    5. ``i2e, bus, gen, branch, areas = ext2int(bus, gen, branch, areas)``

    The deprecated helper-call forms from MATLAB,
    ``ext2int(mpc, val, ordering, dim)`` and
    ``ext2int(mpc, 'field', ordering, dim)``, are intentionally not
    supported here. Use `e2i_data()` or `e2i_field()` directly instead.

    Parameters
    ----------
    *args : tuple
        Positional arguments for either the MATPOWER case-struct form or
        the legacy matrix form.

    Returns
    -------
    dict or tuple
        Internal-order MATPOWER case struct, or the legacy matrix-form
        outputs with ordering metadata.
    """
    nargin = len(args)
    if nargin == 0:
        raise TypeError("ext2int: missing required input arguments")

    first = args[0]
    if isinstance(first, dict) or isinstance(first, MatpowerCase):
        return ext2int_mpc(*args)
    else:
        return ext2int_old(*args)
