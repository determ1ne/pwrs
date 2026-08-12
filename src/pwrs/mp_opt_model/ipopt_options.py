# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping

from ..corex import MatpowerConfig, SolverOptions
from ..corex.solver_options import merge_nested_options


def ipopt_options(
    overrides: Mapping[str, object] | None = None,
    mpopt: MatpowerConfig | Mapping[str, object] | str | None = None,
) -> SolverOptions:
    """Build an IPOPT options dict.

    Parameters
    ----------
    overrides : dict, optional
        Values applied after the defaults and any ``mpopt["ipopt"]["opts"]``
        overrides.
    mpopt : dict or str, optional
        MATPOWER options dict or a user option function name. User option
        functions are not implemented in the current Python port.
    Returns
    -------
    dict
        Options dict for the IPOPT backend.
    """
    verbose = 2
    fname = ""
    have_mpopt = False

    ipopt_config: Mapping[str, object] = {}
    violation = 5e-6
    if isinstance(mpopt, str):
        fname = mpopt
    elif isinstance(mpopt, MatpowerConfig):
        have_mpopt = True
        verbose = mpopt.verbose
        violation = mpopt.opf.violation
        if mpopt.ipopt is not None:
            ipopt_config = mpopt.ipopt.to_dict()
            fname = mpopt.ipopt.opt_fname
            if not fname and mpopt.ipopt.opt:
                fname = f"ipopt_user_options_{mpopt.ipopt.opt}"
    elif isinstance(mpopt, Mapping):
        have_mpopt = True
        configured_verbose = mpopt.get("verbose")
        if isinstance(configured_verbose, int):
            verbose = configured_verbose
        opf = mpopt.get("opf")
        if isinstance(opf, Mapping):
            configured_violation = opf.get("violation")
            if isinstance(configured_violation, (int, float)):
                violation = float(configured_violation)
        candidate = mpopt.get("ipopt")
        if isinstance(candidate, Mapping):
            ipopt_config = candidate
            configured_fname = candidate.get("opt_fname")
            if isinstance(configured_fname, str):
                fname = configured_fname
            configured_opt = candidate.get("opt")
            if not fname and isinstance(configured_opt, int) and configured_opt:
                fname = f"ipopt_user_options_{configured_opt}"

    if verbose:
        print_level = min(12, int(verbose) * 2 + 1)
    else:
        print_level = 0

    opt: SolverOptions = {
        "print_level": print_level,
        "tol": 1e-8,
        "max_iter": 250,
        "dual_inf_tol": 0.1,
        "compl_inf_tol": 1e-5,
        "acceptable_tol": 1e-8,
        "acceptable_compl_inf_tol": 1e-3,
        "mu_strategy": "adaptive",
    }
    if have_mpopt:
        opt["constr_viol_tol"] = violation
        opt["acceptable_constr_viol_tol"] = violation * 100

    if fname:
        raise NotImplementedError("ipopt_options user option functions are not yet implemented")

    if have_mpopt:
        mp_ipopt_opt = ipopt_config.get("opts")
        if isinstance(mp_ipopt_opt, Mapping):
            merge_nested_options(opt, mp_ipopt_opt)
    if overrides is not None:
        merge_nested_options(opt, overrides)

    return opt
