# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping

from ..corex import MatpowerConfig, SolverOptions
from ..corex.solver_options import merge_nested_options


def glpk_options(
    overrides: Mapping[str, object] | None = None,
    mpopt: MatpowerConfig | Mapping[str, object] | str | None = None,
) -> SolverOptions:
    """Build a GLPK options dict.

    Parameters
    ----------
    overrides : dict, optional
        Values applied after the defaults and any ``mpopt["glpk"]["opts"]``
        overrides.
    mpopt : dict or str, optional
        MATPOWER options dict or a user option function name. User option
        functions are not implemented in the current Python port.
    Returns
    -------
    dict
        Options dict for the GLPK backend.
    """
    verbose = 2
    fname = ""
    have_mpopt = False

    glpk_config: Mapping[str, object] = {}
    if isinstance(mpopt, str):
        fname = mpopt
    elif isinstance(mpopt, MatpowerConfig):
        have_mpopt = True
        verbose = mpopt.verbose
    elif isinstance(mpopt, Mapping):
        have_mpopt = True
        configured_verbose = mpopt.get("verbose")
        if isinstance(configured_verbose, int):
            verbose = configured_verbose
        candidate = mpopt.get("glpk")
        if isinstance(candidate, Mapping):
            glpk_config = candidate
            configured_fname = candidate.get("opt_fname")
            if isinstance(configured_fname, str):
                fname = configured_fname

    opt: SolverOptions = {"msglev": verbose}

    if fname:
        raise NotImplementedError("glpk_options user option functions are not yet implemented")

    if have_mpopt:
        mp_glpk_opt = glpk_config.get("opts")
        if isinstance(mp_glpk_opt, Mapping):
            merge_nested_options(opt, mp_glpk_opt)
    if overrides is not None:
        merge_nested_options(opt, overrides)

    return opt
