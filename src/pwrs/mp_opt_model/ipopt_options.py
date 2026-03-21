# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any


def _nested_struct_copy(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _nested_struct_copy(dst[key], value)
        else:
            dst[key] = value
    return dst


def ipopt_options(overrides=None, mpopt=None, nargout=1):
    """Build an IPOPT options dict.

    Parameters
    ----------
    overrides : dict, optional
        Values applied after the defaults and any ``mpopt["ipopt"]["opts"]``
        overrides.
    mpopt : dict or str, optional
        MATPOWER options dict or a user option function name. User option
        functions are not implemented in the current Python port.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    dict
        Options dict for the IPOPT backend.
    """
    verbose = 2
    fname = ""
    have_mpopt = False

    if mpopt not in (None, []):
        if isinstance(mpopt, str):
            fname = mpopt
        else:
            have_mpopt = True
            verbose = mpopt.get("verbose", verbose)
            ipopt = mpopt.get("ipopt", {})
            fname = ipopt.get("opt_fname", "") or ""
            if not fname and ipopt.get("opt", 0):
                fname = f"ipopt_user_options_{int(ipopt['opt'])}"

    if verbose:
        print_level = min(12, int(verbose) * 2 + 1)
    else:
        print_level = 0

    opt: dict[str, Any] = {
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
        viol = mpopt.get("opf", {}).get("violation", 5e-6)
        opt["constr_viol_tol"] = viol
        opt["acceptable_constr_viol_tol"] = viol * 100

    if fname:
        raise NotImplementedError("ipopt_options user option functions are not yet implemented")

    if have_mpopt:
        mp_ipopt_opt = mpopt.get("ipopt", {}).get("opts")
        if isinstance(mp_ipopt_opt, dict):
            _nested_struct_copy(opt, mp_ipopt_opt)
    if isinstance(overrides, dict):
        _nested_struct_copy(opt, overrides)

    outputs = (opt,)
    return outputs[:nargout] if nargout > 1 else opt
