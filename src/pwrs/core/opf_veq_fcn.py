# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from ..corex import MatpowerConfig
from .idx_bus import VMAX


def opf_veq_fcn(x, mpc, idx, mpopt: MatpowerConfig, nargout=1):
    """Evaluate cartesian fixed-voltage equality constraints and Jacobian.

    Computes the equality constraints that enforce specified voltage
    magnitudes at the selected buses in the cartesian OPF formulation.

    Parameters
    ----------
    x : sequence of array_like
        Cartesian voltage state blocks ``(Vr, Vi)``.
    mpc : dict
        Internal MATPOWER case struct.
    idx : array_like
        One-based indices of buses with fixed voltage magnitude constraints.
    mpopt : dict
        MATPOWER options struct.
    nargout : int, optional
        MATLAB compatibility flag controlling whether the Jacobian is
        returned.

    Returns
    -------
    numpy.ndarray or tuple
        Equality constraint vector for the selected voltage magnitudes, and
        optionally its Jacobian.
    """
    Vr, Vi = [np.asarray(v).reshape(-1) for v in x]
    idx = np.asarray(idx).reshape(-1).astype(int)
    nb = len(Vi)
    n = len(idx)
    ii = idx - 1

    Vm2 = Vr[ii] ** 2 + Vi[ii] ** 2
    Veq = Vm2 - mpc["bus"][ii, VMAX - 1] ** 2

    if nargout > 1:
        dVm_dVr = sparse.csc_matrix((2 * Vr[ii], (np.arange(n), ii)), shape=(n, nb))
        dVm_dVi = sparse.csc_matrix((2 * Vi[ii], (np.arange(n), ii)), shape=(n, nb))
        dVeq = sparse.hstack([dVm_dVr, dVm_dVi], format="csc")
        outputs = (Veq, dVeq)
    else:
        outputs = (Veq,)
    return outputs[:nargout] if nargout > 1 else Veq


def opf_veq_fcn_with_jacobian(x, mpc, idx, mpopt: MatpowerConfig):
    """Evaluate fixed-voltage equality constraints and return their Jacobian."""
    return opf_veq_fcn(x, mpc, idx, mpopt, nargout=2)
