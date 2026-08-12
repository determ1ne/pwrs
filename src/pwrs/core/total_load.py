# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
from scipy import sparse

from .idx_bus import BUS_AREA, BUS_I, BUS_TYPE, NONE, VM
from .idx_gen import GEN_BUS, GEN_STATUS, PG, PMIN, QG, QMAX, QMIN
from .isload import isload
from .makeSdzip import makeSdzip


def _as_struct(opt: Any) -> dict[str, Any]:
    if opt is None:
        return {}
    if isinstance(opt, list) and len(opt) == 0:
        return {}
    if isinstance(opt, dict):
        return opt
    if isinstance(opt, str):
        return {"type": opt, "nominal": 1}
    return {}


def _total_load(bus, gen=None, load_zone=None, opt=None, mpopt=None, *, want_q=False):
    """Compute total fixed and/or dispatchable load by zone.

    It can operate on either a case struct or explicit bus/gen matrices
    and returns the real load totals, and optionally reactive load totals,
    aggregated by area, bus, or a custom load-zone mapping.

    Parameters
    ----------
    bus : dict or array_like
        MATPOWER case struct, or the bus matrix.
    gen : array_like, optional
        Generator matrix when ``bus`` is not a case struct.
    load_zone : array_like or str, optional
        Zone definition, or one of MATPOWER's string shorthands such as
        ``"all"``, ``"area"``, or ``"bus"``.
    opt : dict or str, optional
        Options controlling fixed vs dispatchable load handling and nominal
        vs actual load evaluation.
    mpopt : dict, optional
        MATPOWER options struct used for ZIP load handling.
    nargout : int, optional
        MATLAB compatibility flag controlling whether reactive totals are
        returned.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        Column vector of real load totals, and optionally reactive load
        totals.
    """
    if gen is None:
        gen = np.array([])
    if load_zone is None:
        load_zone = []

    if isinstance(bus, dict):
        mpc = bus
        mpopt = opt
        opt = load_zone
        load_zone = gen
        bus = mpc["bus"]
        gen = mpc["gen"]

    bus = np.atleast_2d(np.asarray(bus, dtype=float))
    if gen is None or np.size(gen) == 0:
        gen = np.zeros((0, PMIN + 1), dtype=float)
    else:
        gen = np.atleast_2d(np.asarray(gen, dtype=float))

    nb = bus.shape[0]
    opt = _as_struct(opt)
    if "type" not in opt:
        opt["type"] = "FIXED" if gen.size == 0 else "BOTH"
    if "nominal" not in opt:
        opt["nominal"] = 0

    kind = opt["type"].upper()[:1]
    if kind not in {"F", "D", "B"}:
        raise ValueError("total_load: OPT.type should be 'FIXED', 'DISPATCHABLE' or 'BOTH'")
    want_Q = want_q
    want_fixed = kind in {"B", "F"}
    want_disp = kind in {"B", "D"}

    if isinstance(load_zone, np.ndarray) and load_zone.size == 0:
        load_zone = []
    if isinstance(load_zone, str):
        lower = load_zone.lower()
        if lower == "bus":
            load_zone = np.arange(1, nb + 1)
        elif lower == "all":
            load_zone = np.ones(nb)
        elif lower == "area":
            load_zone = bus[:, BUS_AREA]
    elif load_zone is None or (isinstance(load_zone, (list, tuple)) and len(load_zone) == 0):
        load_zone = bus[:, BUS_AREA]
    else:
        load_zone = np.asarray(load_zone).reshape(-1)

    load_zone = np.asarray(load_zone, dtype=int).reshape(-1)
    nz = int(np.max(load_zone)) if load_zone.size else 0

    if want_fixed:
        Sd = makeSdzip(1, bus, mpopt)
        Vm = bus[:, VM]
        Sbusd = Sd["p"].reshape(-1) + Sd["i"].reshape(-1) * Vm + Sd["z"].reshape(-1) * Vm**2
        Pdf = np.real(Sbusd)
        Qdf = np.imag(Sbusd)
    else:
        Pdf = np.zeros(nb)
        Qdf = np.zeros(nb)

    if want_disp:
        ng = gen.shape[0]
        is_ld = isload(gen) & (gen[:, GEN_STATUS] > 0)
        ld = np.flatnonzero(is_ld)
        i2e = bus[:, BUS_I].astype(int)
        max_i2e = int(np.max(i2e)) if i2e.size else 0
        e2i = np.zeros(max_i2e + 1, dtype=int)
        if i2e.size:
            e2i[i2e] = np.arange(1, nb + 1)
        rows = e2i[gen[:, GEN_BUS].astype(int)] - 1
        Cld = sparse.csc_matrix((is_ld.astype(float), (rows, np.arange(ng))), shape=(nb, ng))
        if int(opt["nominal"]):
            Pdd = -np.asarray(Cld @ gen[:, PMIN])
            Qdd = np.zeros(nb)
            if want_Q:
                Q = np.zeros(ng)
                Q[ld] = (gen[ld, QMIN] == 0) * gen[ld, QMAX] + (gen[ld, QMAX] == 0) * gen[ld, QMIN]
                Qdd = -np.asarray(Cld @ Q)
        else:
            Pdd = -np.asarray(Cld @ gen[:, PG])
            Qdd = np.zeros(nb)
            if want_Q:
                Qdd = -np.asarray(Cld @ gen[:, QG])
    else:
        Pdd = np.zeros(nb)
        Qdd = np.zeros(nb)

    Qd_out = np.zeros(nz)
    if nz == nb and np.array_equal(load_zone, np.arange(1, nb + 1)):
        mask = bus[:, BUS_TYPE] != NONE
        Pd_out = (Pdf + Pdd) * mask
        if want_Q:
            Qd_out = (Qdf + Qdd) * mask
    else:
        Pd_out = np.zeros(nz)
        mask = bus[:, BUS_TYPE] != NONE
        for k in range(1, nz + 1):
            idx = np.flatnonzero((load_zone == k) & mask)
            Pd_out[k - 1] = np.sum(Pdf[idx]) + np.sum(Pdd[idx])
            if want_Q:
                Qd_out[k - 1] = np.sum(Qdf[idx]) + np.sum(Qdd[idx])

    if want_Q:
        return Pd_out, Qd_out
    return Pd_out


def total_load_p(bus, gen=None, load_zone=None, opt=None, mpopt=None):
    """Return active-power load totals only."""
    return _total_load(bus, gen, load_zone, opt, mpopt, want_q=False)


def total_load_pq(bus, gen=None, load_zone=None, opt=None, mpopt=None):
    """Return active- and reactive-power load totals."""
    return _total_load(bus, gen, load_zone, opt, mpopt, want_q=True)


def total_load(bus, gen=None, load_zone=None, opt=None, mpopt=None, *, nargout=None):
    """MATPOWER-compatible output shim for :func:`total_load_p`/`total_load_pq`."""
    if nargout is not None and nargout > 1:
        return total_load_pq(bus, gen, load_zone, opt, mpopt)
    return total_load_p(bus, gen, load_zone, opt, mpopt)
