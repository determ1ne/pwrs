# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy
from typing import Any, cast

import numpy as np
from scipy import sparse

from ..corex import MatpowerCase
from .idx_bus import BUS_AREA, BUS_I, PD, QD
from .idx_gen import GEN_BUS, GEN_STATUS, PG, PMIN, QG, QMAX, QMIN
from .isload import isload
from .loadcase import loadcase_struct
from .modcost import modcost
from .pqcost import pqcost


def _isempty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return False
    if isinstance(value, str):
        return len(value) == 0
    return np.size(value) == 0


def _struct(value):
    if isinstance(value, dict):
        return dict(value)
    return {}


def scale_load(dmd, bus, gen=None, load_zone=None, opt=None, gencost=None, *, nargout=None):
    """Scale fixed and/or dispatchable loads in a MATPOWER case.

    Mirrors MATPOWER's ``scale_load`` helper. It scales fixed bus loads,
    dispatchable loads represented as negative generators, and optionally
    their costs, either by multiplicative factors or by target zone load
    quantities.

    Parameters
    ----------
    dmd : array_like
        Desired scale factors or target load levels by zone.
    bus : dict, str, or array_like
        MATPOWER case struct, case name/path, or bus matrix.
    gen : array_like, optional
        Generator matrix when operating on explicit matrices.
    load_zone : array_like, optional
        Zone mapping for each bus.
    opt : dict, optional
        MATPOWER ``scale_load`` options struct.
    gencost : array_like, optional
        Generator cost matrix to scale along with dispatchable loads when
        requested.
    nargout : int, optional
        MATLAB compatibility flag controlling the returned output form.

    Returns
    -------
    dict or tuple
        Updated case struct, or updated ``bus``, ``gen``, and optional
        ``gencost`` matrices.
    """
    dmd = np.asarray(dmd, dtype=float).reshape(-1)

    mpc: dict[str, Any] = {}
    if isinstance(bus, str):
        bus = loadcase_struct(bus)

    if isinstance(bus, (dict, MatpowerCase)):
        use_mpc = 1
        if load_zone is None:
            load_zone = {}
            if gen is None:
                gen = []
        opt = load_zone
        load_zone = gen
        mpc = copy.deepcopy(bus.to_dict() if isinstance(bus, MatpowerCase) else cast(dict[str, Any], bus))
        gen = mpc["gen"]
        bus = mpc["bus"]
        if "gencost" in mpc:
            gencost = mpc["gencost"]
        else:
            gencost = []
        if nargout is not None and nargout > 1:
            raise ValueError("scale_load: too many output arguments")
    else:
        use_mpc = 0
        if gen is None:
            gen = []
        if load_zone is None:
            load_zone = []
        if opt is None:
            opt = {}
        if gencost is None:
            gencost = []

    bus = np.atleast_2d(np.array(bus, dtype=float, copy=True))
    if _isempty(gen):
        gen = np.array([])
    else:
        gen = np.atleast_2d(np.array(gen, dtype=float, copy=True))
    if not _isempty(gencost):
        gencost = np.atleast_2d(np.array(gencost, dtype=float, copy=True))
    else:
        gencost = np.array([])

    opt = _struct(opt)
    if _isempty(gen):
        opt["which"] = "FIXED"
    if "pq" not in opt:
        opt["pq"] = "PQ"
    if "which" not in opt:
        opt["which"] = "BOTH"
    if "scale" not in opt:
        opt["scale"] = "FACTOR"
    if "cost" not in opt:
        opt["cost"] = -1

    if opt["pq"] != "P" and opt["pq"] != "PQ":
        raise ValueError("scale_load: opt.pq must equal 'PQ' or 'P'")
    if opt["which"][0] not in {"F", "D", "B"}:
        raise ValueError("scale_load: opt.which should be 'FIXED', 'DISPATCHABLE' or 'BOTH'")
    if opt["scale"][0] not in {"F", "Q"}:
        raise ValueError("scale_load: opt.scale should be 'FACTOR' or 'QUANTITY'")
    if _isempty(gen) and opt["which"][0] != "F":
        raise ValueError("scale_load: need gen matrix to scale dispatchable loads")
    if opt["cost"] == -1:
        opt["cost"] = 0 if _isempty(gencost) else 1
    if not use_mpc and (nargout is None or nargout < 3) and opt["cost"]:
        raise ValueError("scale_load: missing gencost as output argument")
    if (nargout or 0) > 2 and _isempty(gencost):
        raise ValueError("scale_load: missing gencost as input argument")

    nb = bus.shape[0]
    if not _isempty(gen):
        ng = gen.shape[0]
        is_ld = isload(gen) & (gen[:, GEN_STATUS - 1] > 0)
        ld = np.flatnonzero(is_ld)

        i2e = bus[:, BUS_I - 1].astype(int)
        e2i = np.zeros(int(np.max(i2e)) + 1, dtype=int)
        e2i[i2e] = np.arange(1, nb + 1)
        Cld = sparse.csc_matrix(
            (is_ld.astype(float), (e2i[gen[:, GEN_BUS - 1].astype(int)] - 1, np.arange(ng))),
            shape=(nb, ng),
        )
    else:
        ng = 0
        ld = np.array([], dtype=int)
        e2i = np.array([], dtype=int)
        Cld = sparse.csc_matrix((nb, 0))

    if _isempty(load_zone):
        if dmd.size == 1:
            load_zone = np.zeros(nb, dtype=float)
            load_zone[(bus[:, PD - 1] != 0) | (bus[:, QD - 1] != 0)] = 1
            if not _isempty(gen):
                load_zone[e2i[gen[ld, GEN_BUS - 1].astype(int)] - 1] = 1
        else:
            load_zone = bus[:, BUS_AREA - 1].copy()
    load_zone = np.asarray(load_zone, dtype=float).reshape(-1)

    if np.max(load_zone, initial=0) > dmd.size:
        raise ValueError("scale_load: load vector must have a value for each load zone specified")

    scale = dmd.copy()
    Pdd = np.zeros(nb, dtype=float)
    if opt["scale"][0] == "Q":
        if not _isempty(gen):
            Pdd = -np.asarray(Cld @ gen[:, PMIN - 1]).reshape(-1)

        for k in range(dmd.size):
            idx = np.flatnonzero(load_zone == k + 1)
            fixed = np.sum(bus[idx, PD - 1])
            dispatchable = np.sum(Pdd[idx])
            total = fixed + dispatchable
            if opt["which"][0] == "B":
                if total != 0:
                    scale[k] = dmd[k] / total
                elif dmd[k] == total:
                    scale[k] = 1
                else:
                    raise ValueError(
                        f"scale_load: impossible to make zone {k + 1} load equal {dmd[k]:g} by scaling non-existent loads"
                    )
            elif opt["which"][0] == "F":
                if fixed != 0:
                    scale[k] = (dmd[k] - dispatchable) / fixed
                elif dmd[k] == dispatchable:
                    scale[k] = 1
                else:
                    raise ValueError(
                        f"scale_load: impossible to make zone {k + 1} load equal {dmd[k]:g} by scaling non-existent fixed load"
                    )
            elif opt["which"][0] == "D":
                if dispatchable != 0:
                    scale[k] = (dmd[k] - fixed) / dispatchable
                elif dmd[k] == fixed:
                    scale[k] = 1
                else:
                    raise ValueError(
                        f"scale_load: impossible to make zone {k + 1} load equal {dmd[k]:g} by scaling non-existent dispatchable load"
                    )

    if opt["which"][0] != "D":
        for k in range(scale.size):
            idx = np.flatnonzero(load_zone == k + 1)
            bus[idx, PD - 1] = bus[idx, PD - 1] * scale[k]
            if opt["pq"] == "PQ":
                bus[idx, QD - 1] = bus[idx, QD - 1] * scale[k]

    if opt["which"][0] != "F":
        for k in range(scale.size):
            idx = np.flatnonzero(load_zone == k + 1)
            i = np.flatnonzero(np.isin(e2i[gen[ld, GEN_BUS - 1].astype(int)] - 1, idx))
            ig = np.asarray(ld[i], dtype=np.int64)

            gen[np.ix_(ig, np.array([PG - 1, PMIN - 1], dtype=int))] = gen[
                np.ix_(ig, np.array([PG - 1, PMIN - 1], dtype=int))
            ] * scale[k]
            if opt["cost"]:
                gencost[ig, :] = modcost(gencost[ig, :], scale[k], "SCALE_F")
                gencost[ig, :] = modcost(gencost[ig, :], scale[k], "SCALE_X")
            if opt["pq"] == "PQ":
                gen[np.ix_(ig, np.array([QG - 1, QMIN - 1, QMAX - 1], dtype=int))] = gen[
                    np.ix_(ig, np.array([QG - 1, QMIN - 1, QMAX - 1], dtype=int))
                ] * scale[k]
                if opt["cost"]:
                    pcost, qcost = pqcost(gencost, ng)
                    if not _isempty(qcost):
                        qcost[ig, :] = modcost(qcost[ig, :], scale[k], "SCALE_F")
                        qcost[ig, :] = modcost(qcost[ig, :], scale[k], "SCALE_X")
                        gencost = np.vstack([pcost, qcost])

    if use_mpc:
        mpc["bus"] = bus
        mpc["gen"] = gen
        if opt["cost"]:
            mpc["gencost"] = gencost
        return mpc

    if nargout == 1 or nargout is None:
        return bus
    if nargout == 2:
        return bus, gen
    return bus, gen, gencost


def scale_load_bus(dmd, bus, gen=None, load_zone=None, opt=None, gencost=None):
    """Scale loads and return the updated bus matrix."""
    return scale_load(dmd, bus, gen, load_zone, opt, gencost, nargout=1)


def scale_load_bus_gen(dmd, bus, gen=None, load_zone=None, opt=None, gencost=None):
    """Scale loads and return updated bus and generator matrices."""
    return scale_load(dmd, bus, gen, load_zone, opt, gencost, nargout=2)


def scale_load_bus_gen_cost(dmd, bus, gen=None, load_zone=None, opt=None, gencost=None):
    """Scale loads and return updated bus, generator and cost matrices."""
    return scale_load(dmd, bus, gen, load_zone, opt, gencost, nargout=3)
