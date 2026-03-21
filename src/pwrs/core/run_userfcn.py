# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from .feval_w_path import _resolve_callable


def run_userfcn(userfcn, stage, *args, nargout=None):
    """Run all user callbacks registered for a stage.

    Executes each callback registered under ``stage`` in sequence, feeding
    the return value from one callback into the next to mirror MATPOWER's
    userfcn execution semantics.

    Parameters
    ----------
    userfcn : dict
        User callback registry from a MATPOWER case struct.
    stage : str
        Callback stage name.
    *args
        Arguments passed to each callback. The first argument is threaded
        through as the return value accumulator.

    Returns
    -------
    Any
        Final callback return value after all stage callbacks have run.
    """
    rv = args[0]
    if userfcn and stage in userfcn:
        for cb in userfcn[stage]:
            cb_args = cb.get("args", [])
            fcn = _resolve_callable(cb["fcn"])
            rv = fcn(rv, *args[1:], cb_args)
    return rv
