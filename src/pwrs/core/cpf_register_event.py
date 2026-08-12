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


def cpf_register_event(cpf_events, name, fcn, tol, locate):
    """Register a CPF event detector.

    Adds a CPF event definition to the existing event list, resolving string
    function names lazily and enforcing unique event names to match
    MATPOWER's callback registry behavior.

    Parameters
    ----------
    cpf_events : dict or list of dict
        Existing CPF event registry, or an empty value to create a new one.
    name : str
        Unique event name.
    fcn : callable or str
        Event function or resolvable function name.
    tol : float
        Event detection tolerance.
    locate : int or bool
        Flag indicating whether the event should be located precisely.

    Returns
    -------
    dict or list of dict
        Updated CPF event registry.
    """
    e = {
        "name": name,
        "fcn": fcn,
        "tol": tol,
        "locate": locate,
    }

    if not callable(e["fcn"]):
        e["fcn"] = _lazy_callable(e["fcn"])

    if not cpf_events:
        return e

    cpf_events = copy.deepcopy(cpf_events)
    if isinstance(cpf_events, dict):
        cpf_events = [cpf_events]
    for cb in cpf_events:
        if cb["name"] == name:
            raise ValueError(f"cpf_register_event: duplicate event name: '{name}'")
    cpf_events.append(e)
    return cpf_events
