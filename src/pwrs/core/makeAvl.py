# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .idx_gen import PG, PMIN, QG, QMAX, QMIN
from .isload import isload


def makeAvl(baseMVA, gen=None, nargout=1):
    """Construct constant-power-factor constraints for dispatchable loads.

    Builds the linear constraint ``lvl <= Avl * [Pg; Qg] <= uvl`` for
    generators representing dispatchable loads with non-unity power factor.

    Parameters
    ----------
    baseMVA : float or dict
        System power base, or a MATPOWER case dict when using the
        one-argument form.
    gen : ndarray, optional
        Generator matrix. Omit to pass a MATPOWER case dict as the first
        argument.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    tuple or sparse matrix
        Returns ``(Avl, lvl, uvl, ivl)`` or the leading subset requested by
        ``nargout``.
    """
    if gen is None:
        mpc = baseMVA
        baseMVA = mpc["baseMVA"]
        gen = mpc["gen"]
    else:
        mpc = None

    ng = gen.shape[0]
    Pg = gen[:, PG - 1] / baseMVA
    Qg = gen[:, QG - 1] / baseMVA
    Pmin = gen[:, PMIN - 1] / baseMVA
    Qmin = gen[:, QMIN - 1] / baseMVA
    Qmax = gen[:, QMAX - 1] / baseMVA

    ivl = np.flatnonzero(isload(gen) & ((Qmin != 0) | (Qmax != 0))) + 1
    nvl = ivl.shape[0]

    rows = ivl - 1
    if np.any((Qmin[rows] != 0) & (Qmax[rows] != 0)):
        if mpc is None:
            s = ""
        else:
            k = np.flatnonzero((Qmin[rows] != 0) & (Qmax[rows] != 0))
            if "order" in mpc and mpc["order"]["state"] == "i":
                gidx = mpc["order"]["gen"]["i2e"][ivl[k] - 1]
            else:
                gidx = ivl[k]
            s = "".join(
                f"Invalid Q limits for dispatchable load in row {int(g)} of gen matrix\n"
                for g in np.asarray(gidx).reshape(-1)
            )
        raise ValueError(f"makeAvl: Either Qmin or Qmax must be equal to zero for each dispatchable load.\n{s}")

    Qlim = (Qmin[rows] == 0) * Qmax[rows] + (Qmax[rows] == 0) * Qmin[rows]
    bad = np.abs(Qg[rows] - Pg[rows] * Qlim / Pmin[rows]) > 1e-6
    if np.any(bad):
        if mpc is None:
            s = ""
        else:
            k = np.flatnonzero(bad)
            if "order" in mpc and mpc["order"]["state"] == "i":
                gidx = mpc["order"]["gen"]["i2e"][ivl[k] - 1]
            else:
                gidx = ivl[k]
            ratios = Qlim[k] / Pmin[rows][k]
            s = "".join(
                f"QG for dispatchable load in row {int(g)} of gen matrix must be PG * {r:g}\n"
                for g, r in zip(np.asarray(gidx).reshape(-1), np.asarray(ratios).reshape(-1))
            )
        raise ValueError(
            "makeAvl: For a dispatchable load, PG and QG must be consistent\n"
            "         with the power factor defined by PMIN and the relevant\n"
            "         (non-zero) QMIN or QMAX limit.\n"
            "         Note: Setting PG = QG = 0 satisfies this condition.\n"
            f"{s}"
        )

    if nvl > 0:
        xx = Pmin[rows]
        yy = Qlim
        pftheta = np.arctan2(yy, xx)
        pc = np.sin(pftheta)
        qc = -np.cos(pftheta)
        ii = np.r_[np.arange(1, nvl + 1), np.arange(1, nvl + 1)] - 1
        jj = np.r_[ivl, ivl + ng] - 1
        Avl = sparse.csc_matrix((np.r_[pc, qc], (ii, jj)), shape=(nvl, 2 * ng))
        lvl = np.zeros(nvl)
        uvl = np.zeros(nvl)
    else:
        Avl = sparse.csc_matrix((0, 2 * ng))
        lvl = np.array([])
        uvl = np.array([])
    outputs = (Avl, lvl, uvl, ivl)
    return outputs[:nargout] if nargout > 1 else Avl
