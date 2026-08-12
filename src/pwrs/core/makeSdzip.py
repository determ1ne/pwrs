# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt

from ..corex import MatpowerConfig
from .idx_bus import PD, QD


def makeSdzip(baseMVA: float, bus: npt.NDArray[np.float64], mpopt: MatpowerConfig | None = None):
    """Build vectors of nominal complex bus power demands for ZIP loads.

    Parameters
    ----------
    baseMVA : float
        System base MVA.
    bus : npt.NDArray[np.float64]
        MATPOWER bus matrix.
    mpopt : MatpowerConfig, optional
        MATPOWER options. The system-wide ZIP fractions are read from
        ``exp.sys_wide_zip_loads.pw`` and ``exp.sys_wide_zip_loads.qw``.

    Returns
    -------
    dict
        Dictionary with fields ``"z"``, ``"i"`` and ``"p"``, each containing
        an ``nb x 1`` complex vector of nominal constant-impedance, constant-
        current and constant-power demand in per unit.

    Notes
    -----
    If ZIP fractions are omitted, the load defaults to constant-power only.
    """
    if mpopt is not None and mpopt.exp.sys_wide_zip_loads.pw is not None:
        pw = np.asarray(mpopt.exp.sys_wide_zip_loads.pw).reshape(-1)
        if pw.size != 3:
            raise ValueError("makeSdzip: 'exp.sys_wide_zip_loads.pw' must be a 1 x 3 vector")
        if abs(np.sum(pw) - 1) > np.finfo(float).eps:
            raise ValueError("makeSdzip: elements of 'exp.sys_wide_zip_loads.pw' must sum to 1")
    else:
        pw = np.array([1.0, 0.0, 0.0])

    if mpopt is not None and mpopt.exp.sys_wide_zip_loads.qw is not None:
        qw = np.asarray(mpopt.exp.sys_wide_zip_loads.qw).reshape(-1)
        if qw.size != 3:
            raise ValueError("makeSdzip: 'exp.sys_wide_zip_loads.qw' must be a 1 x 3 vector")
        if abs(np.sum(qw) - 1) > np.finfo(float).eps:
            raise ValueError("makeSdzip: elements of 'exp.sys_wide_zip_loads.qw' must sum to 1")
    else:
        qw = pw

    Sd = {
        "z": (bus[:, PD] * pw[2] + 1j * bus[:, QD] * qw[2]) / baseMVA,
        "i": (bus[:, PD] * pw[1] + 1j * bus[:, QD] * qw[1]) / baseMVA,
        "p": (bus[:, PD] * pw[0] + 1j * bus[:, QD] * qw[0]) / baseMVA,
    }
    return Sd
