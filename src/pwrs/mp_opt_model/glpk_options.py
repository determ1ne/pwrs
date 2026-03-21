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


def glpk_options(overrides=None, mpopt=None, nargout=1):
    """Build a GLPK options dict.

    Parameters
    ----------
    overrides : dict, optional
        Values applied after the defaults and any ``mpopt["glpk"]["opts"]``
        overrides.
    mpopt : dict or str, optional
        MATPOWER options dict or a user option function name. User option
        functions are not implemented in the current Python port.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    dict
        Options dict for the GLPK backend.
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
            fname = mpopt.get("glpk", {}).get("opt_fname", "") or ""

    opt = {"msglev": verbose}

    if fname:
        raise NotImplementedError("glpk_options user option functions are not yet implemented")

    if have_mpopt:
        mp_glpk_opt = mpopt.get("glpk", {}).get("opts")
        if isinstance(mp_glpk_opt, dict):
            _nested_struct_copy(opt, mp_glpk_opt)
    if isinstance(overrides, dict):
        _nested_struct_copy(opt, overrides)

    outputs = (opt,)
    return outputs[:nargout] if nargout > 1 else opt
