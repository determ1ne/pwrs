# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy
from typing import Any, cast

import numpy as np
from scipy import sparse

from ..corex import MatpowerCase
from .i2e_data import i2e_data as _port_i2e_data
from .i2e_field import i2e_field as _port_i2e_field
from .idx_brch import F_BUS, T_BUS
from .idx_bus import BUS_I
from .idx_gen import GEN_BUS
from .run_userfcn import run_userfcn


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _state(value: Any) -> str:
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    return str(arr.reshape(-1)[0])


def _scalar_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return int(arr.reshape(-1)[0])


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


def _get_dim_size(value: Any, dim: int) -> int:
    axis = dim - 1
    if sparse.issparse(value):
        if axis > 1:
            raise ValueError("sparse reordering only supports dimensions 1 and 2")
        return value.shape[axis]
    arr = np.asarray(value, dtype=object if isinstance(value, list) else None)
    if arr.ndim == 0:
        return 1
    if axis >= arr.ndim:
        return 1
    return arr.shape[axis]


def _get_reorder(value: Any, idx: np.ndarray, dim: int) -> Any:
    axis = dim - 1
    sel = np.asarray(idx, dtype=np.int64).reshape(-1) - 1
    if sparse.issparse(value):
        if axis == 0:
            return value[sel, :]
        if axis == 1:
            return value[:, sel]
        raise ValueError("sparse reordering only supports dimensions 1 and 2")
    if isinstance(value, list):
        if axis != 0:
            raise ValueError("list reordering only supports dimension 1")
        return [value[i] for i in sel]
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr
    return np.take(arr, sel, axis=axis)


def _set_reorder(oldval: Any, value: Any, idx: np.ndarray, dim: int) -> Any:
    axis = dim - 1
    sel = np.asarray(idx, dtype=np.int64).reshape(-1) - 1
    if isinstance(oldval, list):
        if axis != 0:
            raise ValueError("list reordering only supports dimension 1")
        out = list(oldval)
        if sel.size == 0:
            return out
        src = list(value)
        for dest, item in zip(sel, src):
            out[dest] = item
        return out
    out = np.array(oldval, copy=True)
    if out.ndim == 0:
        return out
    if sel.size == 0:
        return out
    target: list[Any] = [slice(None)] * out.ndim
    target[axis] = sel
    out[tuple(target)] = value
    return out


def _concat(parts: list[Any], dim: int) -> Any:
    if not parts:
        return np.array([])
    first = parts[0]
    if sparse.issparse(first):
        return sparse.vstack(parts) if dim == 1 else sparse.hstack(parts)
    if isinstance(first, list):
        out: list[Any] = []
        for part in parts:
            out.extend(part)
        return out
    return np.concatenate(parts, axis=dim - 1)


def _i2e_data(mpc: dict[str, Any], val: Any, oldval: Any, ordering: Any, dim: int = 1) -> Any:
    if "order" not in mpc:
        raise ValueError(
            "i2e_data: mpc does not have the 'order' field required for conversion back to external numbering."
        )
    o = mpc["order"]
    if _state(o["state"]) != "i":
        raise ValueError("i2e_data: mpc does not appear to be in internal order")
    if isinstance(ordering, str):
        if ordering == "gen":
            v = _get_reorder(val, np.asarray(o[ordering]["e2i"]).reshape(-1), dim)
        else:
            v = val
        return _set_reorder(oldval, v, np.asarray(o[ordering]["status"]["on"]).reshape(-1), dim)

    be = 0
    bi = 0
    pieces: list[Any] = []
    for item in ordering:
        ne = _get_dim_size(o["ext"][item], 1)
        ni = _get_dim_size(mpc[item], 1)
        v = _get_reorder(val, np.arange(bi + 1, bi + ni + 1), dim)
        oldv = _get_reorder(oldval, np.arange(be + 1, be + ne + 1), dim)
        pieces.append(_i2e_data(mpc, v, oldv, item, dim))
        be += ne
        bi += ni
    ni = _get_dim_size(val, dim)
    if ni > bi:
        pieces.append(_get_reorder(val, np.arange(bi + 1, ni + 1), dim))
    return _concat(pieces, dim)


def _i2e_field(mpc: dict[str, Any], field: Any, ordering: Any, dim: int = 1) -> dict[str, Any]:
    path = _field_path(field)
    if "int" not in mpc["order"]:
        mpc["order"]["int"] = {}
    _set_nested(mpc["order"]["int"], path, _copy(_get_nested(mpc, path)))
    oldval = _get_nested(mpc["order"]["ext"], path)
    if path == ["gencost"]:
        oldval = np.atleast_2d(np.array(oldval, copy=True))
        _set_nested(mpc["order"]["ext"], path, oldval)
        cur = np.atleast_2d(np.array(_get_nested(mpc, path), copy=True))
        _set_nested(mpc, path, cur)
    converted = _i2e_data(mpc, _get_nested(mpc, path), oldval, ordering, dim)
    if path == ["bus_name"] or path == ["gentype"] or path == ["genfuel"]:
        converted = np.asarray(converted, dtype=object).reshape(-1, 1)
    _set_nested(mpc, path, converted)
    return mpc


def _pad_columns(original: Any, target_cols: int) -> Any:
    arr = np.array(original, copy=True)
    if arr.ndim != 2 or arr.shape[1] >= target_cols:
        return arr
    extra = np.zeros((arr.shape[0], target_cols - arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, extra], axis=1)


def _ensure_matrix(value: Any, cols: int) -> np.ndarray:
    arr = np.array(value, copy=True)
    if arr.ndim == 2:
        return arr
    if arr.size == 0:
        return arr.reshape(0, cols)
    return arr.reshape(-1, cols)


def _old_form(i2e: Any, bus: Any, gen: Any, branch: Any, areas: Any) -> tuple[Any, Any, Any, Any]:
    i2e_arr = np.asarray(i2e).reshape(-1)
    bus = np.array(bus, copy=True)
    gen = np.array(gen, copy=True)
    branch = np.array(branch, copy=True)

    bus[:, BUS_I] = i2e_arr[np.asarray(bus[:, BUS_I], dtype=np.int64) - 1]
    gen[:, GEN_BUS] = i2e_arr[np.asarray(gen[:, GEN_BUS], dtype=np.int64) - 1]
    branch[:, F_BUS] = i2e_arr[np.asarray(branch[:, F_BUS], dtype=np.int64) - 1]
    branch[:, T_BUS] = i2e_arr[np.asarray(branch[:, T_BUS], dtype=np.int64) - 1]
    return bus, gen, branch, areas


def int2ext(i2e: Any, bus: Any = None, gen: Any = None, branch: Any = None, areas: Any = None):
    """Convert MATPOWER data from internal to external indexing.

    Mirrors MATPOWER's ``int2ext`` for both full case structs and the older
    matrix-form API. It restores original bus numbering, reinstates
    out-of-service elements, and maps reordered data back to the stored
    external ordering.

    Parameters
    ----------
    i2e : dict or array_like
        MATPOWER case struct in internal order, or the ``i2e`` index vector
        for the legacy matrix-form API.
    bus : array_like or dict, optional
        Bus matrix for the legacy matrix-form API, or the ``mpopt``/extra
        argument used by the struct form.
    gen : array_like, optional
        Generator matrix for the legacy matrix-form API.
    branch : array_like, optional
        Branch matrix for the legacy matrix-form API.
    areas : array_like, optional
        Areas matrix for the legacy matrix-form API.
    nargout : int, optional
        MATLAB compatibility flag controlling the returned output form.

    Returns
    -------
    dict or tuple
        External-order MATPOWER case struct, or the legacy matrix-form
        outputs with restored external numbering.
    """
    if isinstance(i2e, dict) or isinstance(i2e, MatpowerCase):
        mpc = cast(dict[str, Any], _copy(i2e))
        if "baseMVA" in mpc:
            mpc["baseMVA"] = float(np.asarray(mpc["baseMVA"]).reshape(-1)[0])
        if "bus" in mpc:
            mpc["bus"] = np.atleast_2d(np.array(mpc["bus"], copy=True))
        if "branch" in mpc:
            mpc["branch"] = np.atleast_2d(np.array(mpc["branch"], copy=True))
        if "gen" in mpc:
            mpc["gen"] = np.atleast_2d(np.array(mpc["gen"], copy=True))
        if "gencost" in mpc:
            mpc["gencost"] = np.atleast_2d(np.array(mpc["gencost"], copy=True))
        if (
            "order" in mpc
            and isinstance(mpc["order"], dict)
            and "ext" in mpc["order"]
            and isinstance(mpc["order"]["ext"], dict)
        ):
            for field in ("bus", "branch", "gen", "gencost"):
                if field in mpc["order"]["ext"]:
                    mpc["order"]["ext"][field] = np.atleast_2d(np.array(mpc["order"]["ext"][field], copy=True))
        if (
            "order" in mpc
            and isinstance(mpc["order"], dict)
            and "int" in mpc["order"]
            and isinstance(mpc["order"]["int"], dict)
        ):
            for field in ("bus", "branch", "gen", "gencost"):
                if field in mpc["order"]["int"]:
                    mpc["order"]["int"][field] = np.atleast_2d(np.array(mpc["order"]["int"][field], copy=True))
        if gen is None:
            if "order" not in mpc:
                raise ValueError(
                    "int2ext: mpc does not have the 'order' field required for conversion back to external numbering."
                )
            o = cast(dict[str, Any], _copy(mpc["order"]))

            if _state(o["state"]) == "i":
                if "userfcn" in mpc:
                    mpopt = bus if bus is not None else {}
                    mpc = cast(dict[str, Any], run_userfcn(mpc["userfcn"], "int2ext", mpc, mpopt))
                    o = cast(dict[str, Any], _copy(mpc["order"]))
                if "gencost" in mpc:
                    ordering: Any = ["gen"]
                    if np.shape(mpc["gencost"])[0] == 2 * np.shape(mpc["gen"])[0] and np.shape(mpc["gencost"])[0] != 0:
                        ordering = ["gen", "gen"]
                    mpc = cast(dict[str, Any], _port_i2e_field(mpc, "gencost", ordering))
                    o = cast(dict[str, Any], _copy(mpc["order"]))
                if "bus_name" in mpc:
                    mpc = cast(dict[str, Any], _port_i2e_field(mpc, "bus_name", ["bus"]))
                    o = cast(dict[str, Any], _copy(mpc["order"]))
                if "gentype" in mpc:
                    mpc = cast(dict[str, Any], _port_i2e_field(mpc, "gentype", ["gen"]))
                    o = cast(dict[str, Any], _copy(mpc["order"]))
                if "genfuel" in mpc:
                    mpc = cast(dict[str, Any], _port_i2e_field(mpc, "genfuel", ["gen"]))
                    o = cast(dict[str, Any], _copy(mpc["order"]))
                if "A" in mpc:
                    if "int" not in o:
                        o["int"] = {}
                    o["int"]["A"] = _copy(mpc["A"])
                    mpc["A"] = _copy(o["ext"]["A"])
                if "N" in mpc:
                    if "int" not in o:
                        o["int"] = {}
                    o["int"]["N"] = _copy(mpc["N"])
                    mpc["N"] = _copy(o["ext"]["N"])

                if "int" not in o:
                    o["int"] = {}
                o["int"]["bus"] = _copy(mpc["bus"])
                o["int"]["branch"] = _copy(mpc["branch"])
                o["int"]["gen"] = _copy(mpc["gen"])
                mpc["bus"] = _copy(o["ext"]["bus"])
                mpc["branch"] = _copy(o["ext"]["branch"])
                mpc["gen"] = _copy(o["ext"]["gen"])

                bus_cols = np.shape(o["ext"]["bus"])[1]
                branch_cols = np.shape(o["ext"]["branch"])[1]
                gen_cols = np.shape(o["ext"]["gen"])[1]
                o["int"]["bus"] = _ensure_matrix(o["int"]["bus"], bus_cols)
                o["int"]["branch"] = _ensure_matrix(o["int"]["branch"], branch_cols)
                o["int"]["gen"] = _ensure_matrix(o["int"]["gen"], gen_cols)
                mpc["bus"] = _pad_columns(mpc["bus"], o["int"]["bus"].shape[1])
                mpc["branch"] = _pad_columns(mpc["branch"], o["int"]["branch"].shape[1])
                mpc["gen"] = _pad_columns(mpc["gen"], o["int"]["gen"].shape[1])

                bus_on = np.asarray(o["bus"]["status"]["on"], dtype=np.int64).reshape(-1) - 1
                br_on = np.asarray(o["branch"]["status"]["on"], dtype=np.int64).reshape(-1) - 1
                gen_on = np.asarray(o["gen"]["status"]["on"], dtype=np.int64).reshape(-1) - 1
                gen_e2i = np.asarray(o["gen"]["e2i"], dtype=np.int64).reshape(-1) - 1
                bus_i2e = np.asarray(o["bus"]["i2e"]).reshape(-1)

                mpc["bus"][bus_on, :] = o["int"]["bus"]
                mpc["branch"][br_on, :] = o["int"]["branch"]
                mpc["gen"][gen_on, :] = o["int"]["gen"][gen_e2i, :]

                bus_internal = np.asarray(mpc["bus"][bus_on, BUS_I], dtype=np.int64) - 1
                f_internal = np.asarray(mpc["branch"][br_on, F_BUS], dtype=np.int64) - 1
                t_internal = np.asarray(mpc["branch"][br_on, T_BUS], dtype=np.int64) - 1
                gen_internal = np.asarray(mpc["gen"][gen_on, GEN_BUS], dtype=np.int64) - 1
                mpc["bus"][bus_on, BUS_I] = bus_i2e[bus_internal]
                mpc["branch"][br_on, F_BUS] = bus_i2e[f_internal]
                mpc["branch"][br_on, T_BUS] = bus_i2e[t_internal]
                mpc["gen"][gen_on, GEN_BUS] = bus_i2e[gen_internal]

                if "ext" in o:
                    del o["ext"]
                o["state"] = "e"
                mpc["order"] = o
                return mpc

            raise ValueError("int2ext: mpc claims it is already using external numbering.")

        if isinstance(bus, (str, list, tuple)):
            dim = 1 if branch is None else _scalar_int(branch, 1)
            return _port_i2e_field(mpc, bus, gen, dim)
        dim = 1 if areas is None else _scalar_int(areas, 1)
        return _port_i2e_data(mpc, bus, gen, branch, dim)

    return _old_form(i2e, bus, gen, branch, areas)
