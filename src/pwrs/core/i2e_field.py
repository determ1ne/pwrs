# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from .i2e_data import i2e_data


def _field_path(field: Any) -> list[str]:
    if isinstance(field, str):
        return [field]
    if isinstance(field, (list, tuple)):
        return [str(part) for part in field]
    raise TypeError("field must be a string or list of strings")


def _get_nested(data: dict[str, Any], path: list[str]) -> Any:
    cur = data
    for key in path:
        cur = cur[key]
    return cur


def _set_nested(data: dict[str, Any], path: list[str], value: Any) -> None:
    cur = data
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def i2e_field(mpc, field, ordering, dim=1):
    """Convert a case struct field from internal to external ordering.

    Saves the current internal-order value for ``field`` under
    ``mpc['order']['int']`` and replaces the live field value with the
    external-order version produced by ``i2e_data``.

    Parameters
    ----------
    mpc : dict
        MATPOWER case struct in internal order, with ``order`` metadata.
    field : str or sequence of str
        Field name or nested field path to convert.
    ordering : str or sequence
        MATPOWER ordering descriptor used for the conversion.
    dim : int, optional
        One-based dimension along which to reorder.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    dict
        Updated MATPOWER case struct with the converted field.
    """
    path = _field_path(field)
    if "int" not in mpc["order"]:
        mpc["order"]["int"] = {}
    _set_nested(mpc["order"]["int"], path, _get_nested(mpc, path))
    oldval = _get_nested(mpc["order"]["ext"], path)
    _set_nested(mpc, path, i2e_data(mpc, _get_nested(mpc, path), oldval, ordering, dim))
    return mpc
