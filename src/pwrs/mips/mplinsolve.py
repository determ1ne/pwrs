# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve


def mplinsolve(A, b, solver="", opt=None):
    """Solve a linear system for MATPOWER/MIPS.

    Parameters
    ----------
    A : array_like or sparse matrix
        Coefficient matrix.
    b : array_like
        Right-hand side vector or matrix.
    solver : str, optional
        Linear solver selector. The current Python port supports the
        built-in dense solve path and sparse LU-based variants used by MIPS.
    opt : dict, optional
        Solver-specific options. Present for MATLAB interface compatibility.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    ndarray or tuple
        Solution array ``x`` or ``(x, info)`` when ``nargout > 1``.
    """
    if solver is None:
        solver = ""

    if solver in {"", "\\"}:
        x = spsolve(A, b) if sparse.issparse(A) else np.linalg.solve(A, b)
    elif solver in {"LU", "LU3", "LU3a", "LU4", "LU5", "LU3m", "LU3am", "LU4m", "LU5m"}:
        if sparse.issparse(A):
            lu = splu(A)
            x = lu.solve(b)
        else:
            x = np.linalg.solve(A, b)
    elif solver == "KLU":
        try:
            import nbklu
        except ImportError:
            warnings.warn("mplinsolve: KLU solver requested but nbklu is not installed, falling back to default solver")
            return mplinsolve(A, b, solver="", opt=opt)
        if not sparse.issparse(A):
            warnings.warn("mplinsolve: KLU solver requires sparse matrix input, falling back to default solver")
            return mplinsolve(A, b, solver="", opt=opt)
        lu = nbklu.KLUSolver()
        # disable BTF, as it usually does not help for our problems
        lu.common.btf = 0
        lu.analyze(A.shape[0], A.indptr, A.indices)
        lu.factor(A.data)
        x = lu.solve(b)
    else:
        warnings.warn(f"mplinsolve: unrecognized solver '{solver}', falling back to default solver")
        return mplinsolve(A, b, solver="", opt=opt)

    return x
