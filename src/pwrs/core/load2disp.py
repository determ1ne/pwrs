# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy
from typing import Any, cast

import numpy as np

from .idx_bus import BUS_I, PD, QD, VM
from .idx_cost import POLYNOMIAL
from .idx_gen import APF
from .loadcase import loadcase_struct
from .savecase import savecase


def _isempty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return len(value) == 0
    return np.size(value) == 0


def load2disp(mpc0, fname=None, idx=None, voll=None):
    """Convert fixed loads into dispatchable loads.

    Mirrors MATPOWER's ``load2disp`` helper by replacing selected fixed bus
    loads with negative generators and corresponding piecewise-linear cost
    data representing value of lost load.

    Parameters
    ----------
    mpc0 : dict or str
        MATPOWER case struct or case name/path.
    fname : str, optional
        Output file name for saving the converted case.
    idx : array_like, optional
        One-based indices of buses whose loads should be converted.
    voll : float or array_like, optional
        Value of lost load coefficients for the converted dispatchable loads.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    dict
        Updated MATPOWER case struct with dispatchable loads appended to the
        generator and cost matrices.
    """
    mpc = cast(dict[str, Any], copy.deepcopy(loadcase_struct(mpc0)))

    if idx is None or _isempty(idx):
        idx = np.flatnonzero(mpc["bus"][:, PD - 1] > 0) + 1
    idx = np.asarray(idx).reshape(-1).astype(int) - 1

    voll0 = 5000
    mBase = 100
    nld = len(idx)
    v1 = np.ones((nld, 1))

    gen = np.hstack(
        [
            mpc["bus"][np.ix_(idx, np.array([BUS_I - 1], dtype=int))],
            -mpc["bus"][np.ix_(idx, np.array([PD - 1], dtype=int))],
            -mpc["bus"][np.ix_(idx, np.array([QD - 1], dtype=int))],
            np.maximum(0, -mpc["bus"][np.ix_(idx, np.array([QD - 1], dtype=int))]),
            np.minimum(0, -mpc["bus"][np.ix_(idx, np.array([QD - 1], dtype=int))]),
            mpc["bus"][np.ix_(idx, np.array([VM - 1], dtype=int))],
            mBase * v1,
            v1,
            np.maximum(0, -mpc["bus"][np.ix_(idx, np.array([PD - 1], dtype=int))]),
            np.minimum(0, -mpc["bus"][np.ix_(idx, np.array([PD - 1], dtype=int))]),
            np.zeros((nld, 6)),
            np.full((nld, 4), np.inf),
            np.zeros((nld, 1)),
        ]
    )
    ng, nc = mpc["gen"].shape
    mpc["gen"] = np.vstack([mpc["gen"], np.zeros((nld, nc))])
    mpc["gen"][ng : ng + nld, :APF] = gen

    mpc["bus"][np.ix_(idx, np.array([PD - 1, QD - 1], dtype=int))] = 0

    nc = mpc["gencost"].shape[1]
    if voll is None:
        voll = voll0 * v1
    else:
        voll = np.asarray(voll, dtype=float)
        if voll.size == 1:
            voll = float(voll.reshape(-1)[0]) * v1
        else:
            voll = voll.reshape(-1, 1)

    gencost = np.hstack(
        [
            POLYNOMIAL * v1,
            np.zeros((nld, 2)),
            2 * v1,
            voll,
            np.zeros((nld, nc - 5)),
        ]
    )
    mpc["gencost"] = np.vstack([mpc["gencost"], gencost])

    if "genfuel" in mpc and isinstance(mpc["genfuel"], list):
        mpc["genfuel"] = list(mpc["genfuel"]) + ["dl" for _ in range(nld)]
    if "gentype" in mpc and isinstance(mpc["gentype"], list):
        mpc["gentype"] = list(mpc["gentype"]) + ["DL" for _ in range(nld)]

    if fname is not None and not _isempty(fname):
        savecase(fname, mpc, "2")

    return mpc
