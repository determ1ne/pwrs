# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from scipy import sparse

from ..corex import ArrayLike, ComplexArray, FloatArray, MatpowerConfig, SparseMatrix, as_csc_matrix
from .idx_gen import GEN_BUS, GEN_STATUS, PG, QG
from .makeSdzip import makeSdzip


def makeSbus_dV(
    baseMVA: float,
    bus: FloatArray,
    gen: FloatArray,
    mpopt: MatpowerConfig | None = None,
    Vm: ArrayLike | None = None,
) -> tuple[FloatArray, SparseMatrix]:
    """Return ``(empty, dSbus_dVm)`` with explicit Python semantics."""
    nb = bus.shape[0]
    Sd = makeSdzip(baseMVA, bus, mpopt)
    if Vm is None or np.size(Vm) == 0:
        dSbus_dVm = sparse.csc_matrix((nb, nb), dtype=complex)
    else:
        Vm = np.asarray(Vm)
        diag = -(Sd["i"] + 2 * Vm * Sd["z"])
        dSbus_dVm = sparse.diags(diag, shape=(nb, nb), format="csc").tocsc()
    return np.zeros((0, 0)), as_csc_matrix(dSbus_dVm)


def makeSbus_value_dV(
    baseMVA: float,
    bus: FloatArray,
    gen: FloatArray,
    mpopt: MatpowerConfig | None = None,
    Vm: ArrayLike | None = None,
    Sg: ArrayLike | None = None,
) -> tuple[ComplexArray, SparseMatrix]:
    """Return power injections together with the ZIP-load voltage derivative."""
    return makeSbus_value(baseMVA, bus, gen, mpopt, Vm, Sg), makeSbus_dV(baseMVA, bus, gen, mpopt, Vm)[1]


def makeSbus_value(
    baseMVA: float,
    bus: FloatArray,
    gen: FloatArray,
    mpopt: MatpowerConfig | None = None,
    Vm: ArrayLike | None = None,
    Sg: ArrayLike | None = None,
) -> ComplexArray:
    """Return the complex bus injection vector with explicit Python semantics."""
    nb = bus.shape[0]
    Sd = makeSdzip(baseMVA, bus, mpopt)

    on = np.where(gen[:, GEN_STATUS - 1] > 0)[0]
    gbus = gen[on, GEN_BUS - 1].astype(int) - 1
    ngon = on.size
    Cg = sparse.csc_matrix((np.ones(ngon), (gbus, np.arange(ngon))), shape=(nb, ngon))
    if Sg is not None and np.size(Sg):
        Sg = np.asarray(Sg)
        Sbusg = Cg @ Sg[on]
    else:
        Sbusg = Cg @ ((gen[on, PG - 1] + 1j * gen[on, QG - 1]) / baseMVA)

    if Vm is None or np.size(Vm) == 0:
        Vm = np.ones(nb)
    else:
        Vm = np.asarray(Vm)
    Sbusd = Sd["p"] + Sd["i"] * Vm + Sd["z"] * Vm**2
    return Sbusg - Sbusd


def makeSbus(
    baseMVA: float,
    bus: FloatArray,
    gen: FloatArray,
    mpopt: MatpowerConfig | None = None,
    Vm: ArrayLike | None = None,
    Sg: ArrayLike | None = None,
    nargout: int | None = None,
) -> ComplexArray | tuple[FloatArray, SparseMatrix]:
    """Build the vector of complex bus power injections.

    Parameters
    ----------
    baseMVA : float
        System base MVA.
    bus : array_like
        MATPOWER bus matrix.
    gen : array_like
        MATPOWER generator matrix.
    mpopt : dict, optional
        MATPOWER options dict. When provided together with ``Vm``, ZIP loads
        are evaluated using the configured system-wide ZIP coefficients.
    Vm : array_like, optional
        Bus voltage magnitudes used to evaluate voltage-dependent ZIP loads.
        If omitted or empty, nominal voltage is assumed.
    Sg : array_like, optional
        Complex generator injections in per unit. When provided, these values
        override the ``PG``/``QG`` columns in ``gen`` and ``gen`` is used only
        for connectivity.
    nargout : int, optional
        MATLAB-compatibility output selector. If ``nargout == 2``, the first
        return value is an empty array and the second is ``dSbus_dVm``.

    Returns
    -------
    numpy.ndarray or tuple
        With the default single-output form, returns the complex bus
        injection vector in per unit, i.e. generation minus load. With
        ``nargout == 2``, returns ``(empty, dSbus_dVm)`` where ``dSbus_dVm``
        is the partial derivative of bus injections with respect to voltage
        magnitude.

    See Also
    --------
    makeYbus, makeSdzip
    """
    if nargout == 2:
        return makeSbus_dV(baseMVA, bus, gen, mpopt, Vm)
    return makeSbus_value(baseMVA, bus, gen, mpopt, Vm, Sg)
