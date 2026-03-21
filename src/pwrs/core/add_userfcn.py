# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy

_VALID_STAGES = {"ext2int", "formulation", "int2ext", "printpf", "savecase"}


def _fcn_name(fcn):
    if isinstance(fcn, str):
        return fcn
    return getattr(fcn, "__name__", repr(fcn))


def add_userfcn(mpc, stage, fcn, args=None, allow_multiple=0):
    """Add a user callback to a MATPOWER case struct.

    Registers a callback function for a named MATPOWER processing stage in
    ``mpc['userfcn']``. By default it rejects duplicate registrations of the
    same callback at the same stage.

    Parameters
    ----------
    mpc : dict
        MATPOWER case struct.
    stage : str
        Callback stage name, such as ``'ext2int'``, ``'formulation'``,
        ``'int2ext'``, ``'printpf'``, or ``'savecase'``.
    fcn : callable or str
        Callback function or function name.
    args : list, optional
        Extra callback arguments stored with the registration record.
    allow_multiple : int or bool, optional
        Allow duplicate registrations of the same callback when true.

    Returns
    -------
    dict
        Updated MATPOWER case struct with the callback registration added.
    """
    if stage not in _VALID_STAGES:
        raise ValueError(f"add_userfcn : '{stage}' is not the name of a valid callback stage\n")

    mpc = copy.deepcopy(mpc)
    if "userfcn" not in mpc:
        mpc["userfcn"] = {}
    if stage not in mpc["userfcn"]:
        mpc["userfcn"][stage] = []

    if not allow_multiple:
        fcn_name = _fcn_name(fcn)
        for cb in mpc["userfcn"][stage]:
            if _fcn_name(cb["fcn"]) == fcn_name:
                raise ValueError(f"add_userfcn: the function '{fcn_name}' has already been added")

    cb = {"fcn": fcn}
    if args is not None and args != []:
        cb["args"] = args
    mpc["userfcn"][stage].append(cb)
    return mpc
