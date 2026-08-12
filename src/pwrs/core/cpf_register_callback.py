# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy

from .feval_w_path import _resolve_callable


def _lazy_callable(name):
    def _wrapped(*args, **kwargs):
        return _resolve_callable(name)(*args, **kwargs)

    _wrapped.__name__ = str(name)
    return _wrapped


def cpf_register_callback(cpf_callbacks, fcn, priority=None, args=None):
    """Register a CPF callback.

    Adds a CPF callback to the existing callback list, resolving string
    function names lazily and sorting callbacks by descending priority to
    match MATPOWER's execution order.

    Parameters
    ----------
    cpf_callbacks : dict or list of dict
        Existing CPF callback registry, or an empty value to create a new
        one.
    fcn : callable or str
        Callback function or resolvable function name.
    priority : int, optional
        Callback priority. Higher-priority callbacks run first.
    args : list, optional
        Extra callback arguments stored with the registration record.

    Returns
    -------
    dict or list of dict
        Updated CPF callback registry.
    """
    if args is None:
        args = []
    if priority is None or priority == []:
        priority = 20

    cb = {
        "fcn": fcn,
        "priority": priority,
        "args": args,
    }
    if not callable(cb["fcn"]):
        cb["fcn"] = _lazy_callable(cb["fcn"])

    if not cpf_callbacks:
        return cb

    cpf_callbacks = copy.deepcopy(cpf_callbacks)
    if isinstance(cpf_callbacks, dict):
        cpf_callbacks = [cpf_callbacks]
    cpf_callbacks.append(cb)
    cpf_callbacks = sorted(cpf_callbacks, key=lambda item: item["priority"], reverse=True)
    return cpf_callbacks
