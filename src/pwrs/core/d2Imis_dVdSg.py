# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def d2Imis_dVdSg(Cg, V, lam, vcart=0, nargout=1):
    """Return mixed 2nd derivatives of current mismatch w.r.t. ``V`` and ``Sg``.

    Computes the cross-derivative blocks for the current balance equations
    with respect to the voltage variables and the real/reactive generator
    injections mapped to buses by ``Cg``.

    Parameters
    ----------
    Cg : sparse matrix
        Generator connection matrix mapping generators to buses.
    V : array_like
        Complex bus voltage vector.
    lam : array_like
        Multiplier vector for the current balance equations.
    vcart : int or bool, optional
        Voltage coordinate flag. Use polar coordinates when false and
        cartesian coordinates when true.
    nargout : int, optional
        MATLAB compatibility flag controlling how many outputs are returned.

    Returns
    -------
    scipy.sparse.csc_matrix
        Mixed derivative matrix with the MATPOWER block layout for ``Pg`` and
        ``Qg`` against the selected voltage coordinates.
    """
    if vcart is None:
        vcart = 0

    V = np.asarray(V).reshape(-1)
    lam = np.asarray(lam).reshape(-1)
    nb = len(V)

    if vcart:
        D = Cg.T @ sparse.diags(lam / np.conjugate(V**2), offsets=0, shape=(nb, nb), format="csc")

        G_Pg_Vr = D
        G_Pg_Vi = -1j * D
        G_Qg_Vr = G_Pg_Vi
        G_Qg_Vi = -D

        Gsv = sparse.vstack(
            [
                sparse.hstack([G_Pg_Vr, G_Pg_Vi], format="csc"),
                sparse.hstack([G_Qg_Vr, G_Qg_Vi], format="csc"),
            ],
            format="csc",
        )
    else:
        D = sparse.diags(1 / np.abs(V), offsets=0, shape=(nb, nb), format="csc")
        E = sparse.diags(lam / np.conjugate(V), offsets=0, shape=(nb, nb), format="csc")
        K = Cg.T @ E
        L = K @ D

        G_Pg_Va = -1j * K
        G_Pg_Vm = L
        G_Qg_Va = -K
        G_Qg_Vm = -1j * L

        Gsv = sparse.vstack(
            [
                sparse.hstack([G_Pg_Va, G_Pg_Vm], format="csc"),
                sparse.hstack([G_Qg_Va, G_Qg_Vm], format="csc"),
            ],
            format="csc",
        )

    outputs = (Gsv,)
    return outputs[:nargout] if nargout > 1 else Gsv
