# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from ..corex import QpMultipliers, QpResult, normalize_qp_problem
from .mips import mips_full


def qps_mips_full(H, c=None, A=None, l=None, u=None, xmin=None, xmax=None, x0=None, opt=None) -> QpResult:
    """Quadratic-program solver wrapper based on MIPS.

    Solves ``min 0.5 * x' * H * x + c' * x`` subject to linear constraints
    and variable bounds.

    Parameters
    ----------
    H : array_like or dict
        Quadratic cost matrix or a problem dict containing the full solver
        inputs.
    c : array_like, optional
        Linear cost vector.
    A, l, u : array_like, optional
        Linear constraints ``l <= A*x <= u``.
    xmin, xmax : array_like, optional
        Variable bounds.
    x0 : array_like, optional
        Initial point.
    opt : dict, optional
        MIPS options dict.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    tuple or ndarray
        Returns ``(x, f, eflag, output, lambda_)`` or the leading subset
        requested by ``nargout``.
    """
    problem = normalize_qp_problem(H, c, A, l, u, xmin, xmax, x0, opt, solver_name="qps_mips")

    def qp_f(x):
        f = 0.5 * x @ (problem.H @ x) + problem.c @ x
        df = problem.H @ x + problem.c
        return f, df, problem.H

    solution, objective, eflag, output, fields = mips_full(
        qp_f,
        problem.x0,
        problem.A,
        problem.l,
        problem.u,
        problem.xmin,
        problem.xmax,
        opt=problem.options,
    )
    multipliers: QpMultipliers = {
        "mu_l": fields["mu_l"],
        "mu_u": fields["mu_u"],
        "lower": fields["lower"],
        "upper": fields["upper"],
    }
    return solution, objective, eflag, output, multipliers


def qps_mips(H, c=None, A=None, l=None, u=None, xmin=None, xmax=None, x0=None, opt=None, nargout: int = 1):
    """MATPOWER-compatible MIPS QP entry point; use ``qps_mips_full`` in typed code."""
    result = qps_mips_full(H, c, A, l, u, xmin, xmax, x0, opt)
    return result[:nargout] if nargout > 1 else result[0]
