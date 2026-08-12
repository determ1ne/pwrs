# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from ..corex import MatpowerConfig
from .idx_bus import VA


def opf_vref_fcn(x, mpc, refs, mpopt: MatpowerConfig, nargout=1):
    """Evaluate reference angle constraints and Jacobian.

    Computes the equality constraints that pin the voltage angles at the
    reference buses to their specified values in the cartesian-voltage OPF
    formulation.

    Parameters
    ----------
    x : sequence of array_like
        Cartesian voltage state blocks ``(Vr, Vi)``.
    mpc : dict
        Internal MATPOWER case struct.
    refs : array_like
        One-based indices of reference buses.
    mpopt : MatpowerConfig
        Typed MATPOWER options configuration.
    nargout : int, optional
        MATLAB compatibility flag controlling whether the Jacobian is
        returned.

    Returns
    -------
    numpy.ndarray or tuple
        Reference-angle constraint vector, and optionally its Jacobian with
        respect to ``Vr`` and ``Vi``.
    """
    Vr, Vi = [np.asarray(v).reshape(-1) for v in x]
    refs = np.asarray(refs).reshape(-1).astype(int)
    n = len(refs)
    nb = len(Vr)

    ref_idx = refs - 1
    Vref = np.angle(Vr[ref_idx] + 1j * Vi[ref_idx]) - mpc["bus"][ref_idx, VA - 1] * np.pi / 180

    if nargout > 1:
        Vm2 = Vr[ref_idx] ** 2 + Vi[ref_idx] ** 2
        dVa_dVr = sparse.csc_matrix((-Vi[ref_idx] / Vm2, (np.arange(n), ref_idx)), shape=(n, nb))
        dVa_dVi = sparse.csc_matrix((Vr[ref_idx] / Vm2, (np.arange(n), ref_idx)), shape=(n, nb))
        dVref = sparse.hstack([dVa_dVr, dVa_dVi], format="csc")
        outputs = (Vref, dVref)
    else:
        outputs = (Vref,)
    return outputs[:nargout] if nargout > 1 else Vref


def opf_vref_fcn_with_jacobian(x, mpc, refs, mpopt: MatpowerConfig):
    """Evaluate voltage-reference constraints and return their Jacobian."""
    return opf_vref_fcn(x, mpc, refs, mpopt, nargout=2)
