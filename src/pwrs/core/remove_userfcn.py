# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy


def _fcn_name(fcn):
    if isinstance(fcn, str):
        return fcn
    return getattr(fcn, "__name__", repr(fcn))


def remove_userfcn(mpc, stage, fcn, *, nargout=None):
    """Remove a user callback from a MATPOWER case struct.

    Deletes the first matching callback registration for ``fcn`` at the given
    callback stage.

    Parameters
    ----------
    mpc : dict
        MATPOWER case struct.
    stage : str
        Callback stage name.
    fcn : callable or str
        Callback function or function name to remove.

    Returns
    -------
    dict
        Updated MATPOWER case struct with the callback removed.
    """
    mpc = copy.deepcopy(mpc)
    callbacks = mpc["userfcn"][stage]
    fcn_name = _fcn_name(fcn)
    for k in range(len(callbacks) - 1, -1, -1):
        if _fcn_name(callbacks[k]["fcn"]) == fcn_name:
            del callbacks[k]
            break
    return mpc
