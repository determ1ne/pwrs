# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class _NamedIndex:
    i1: dict[str, Any] = field(default_factory=dict)
    iN: dict[str, Any] = field(default_factory=dict)
    N: dict[str, Any] = field(default_factory=dict)


@dataclass
class _NamedSet:
    idx: _NamedIndex = field(default_factory=_NamedIndex)
    N: int = 0
    NS: int = 0
    order: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, dict[str, Any]] = field(default_factory=dict)
    params: Any = None


class MPIdxManager:
    """MATPOWER index-manager base class.

    The index manager tracks named and indexed blocks for set types such as
    variables, constraints, and costs. It keeps the ordering and index
    ranges for each block as sets are added to the model object.

    Parameters
    ----------
    s : MPIdxManager or dict, optional
        Existing manager or compatible struct-like dict used to initialize
        the object.
    """

    def __init__(self, s: Any | None = None):
        self.userdata: dict[str, Any] = {}
        self.set_types: dict[str, str] = {}
        if s is not None:
            if isinstance(s, MPIdxManager):
                for key, value in s.__dict__.items():
                    setattr(self, key, copy(value))
            elif isinstance(s, dict):
                for key in list(self.__dict__.keys()):
                    if key in s:
                        setattr(self, key, s[key])
            else:
                raise TypeError("@mp_idx_manager/mp_idx_manager: input must be an 'mp_idx_manager' object or a struct")
        self.def_set_types()

    def def_set_types(self) -> None:
        raise NotImplementedError

    def init_set_types(self) -> None:
        for name in self.set_types:
            setattr(self, name, _NamedSet())

    def copy(self):
        new_obj = self.__class__()
        for key, value in self.__dict__.items():
            setattr(new_obj, key, copy(value))
        return new_obj

    def valid_named_set_type(self, set_type: str) -> str:
        return self.set_types.get(set_type, "")

    def _zero_container(self, dim_list: list[int]) -> Any:
        if len(dim_list) == 1:
            dim_list = [dim_list[0], 1]
        return np.zeros(tuple(dim_list), dtype=int)

    def init_indexed_name(self, set_type: str, name: str, dim_list: list[int]) -> None:
        st_label = self.valid_named_set_type(set_type)
        if not st_label:
            stypes = "".join(f"\n  '{f}'" for f in self.set_types)
            raise ValueError(
                f"@mp_idx_manager/init_indexed_name: '{set_type}' is not a valid SET_TYPE, must be one of the following:{stypes}"
            )
        obj_ff: _NamedSet = getattr(self, set_type)
        if name in obj_ff.idx.N:
            raise ValueError(f"@mp_idx_manager/init_indexed_name: {st_label} set named '{name}' already exists")
        zero_vector = self._zero_container(list(dim_list))
        obj_ff.idx.i1[name] = zero_vector.copy()
        obj_ff.idx.iN[name] = zero_vector.copy()
        obj_ff.idx.N[name] = zero_vector.copy()

    def add_named_set(self, set_type: str, name: str, idx: list[int] | tuple[int, ...], N: int, *args: Any) -> None:
        st_label = self.valid_named_set_type(set_type)
        if not st_label:
            stypes = "".join(f"\n  '{f}'" for f in self.set_types)
            raise ValueError(
                f"@mp_idx_manager/add_named_set: '{set_type}' is not a valid SET_TYPE, must be one of the following:{stypes}"
            )
        obj_ff: _NamedSet = getattr(self, set_type)
        idx = list(idx)
        if not idx:
            if name in obj_ff.idx.N:
                raise ValueError(f"@mp_idx_manager/add_named_set: {st_label} set named '{name}' already exists")
            obj_ff.idx.i1[name] = obj_ff.N + 1
            obj_ff.idx.iN[name] = obj_ff.N + N
            obj_ff.idx.N[name] = N
            obj_ff.N = obj_ff.idx.iN[name]
        else:
            arr_i1 = obj_ff.idx.i1[name]
            ref = tuple(i - 1 for i in idx)
            if int(arr_i1[ref]) != 0:
                nname = f"{name}({','.join(str(i) for i in idx)})"
                raise ValueError(f"@mp_idx_manager/add_named_set: {st_label} set named '{nname}' already exists")
            obj_ff.idx.i1[name][ref] = obj_ff.N + 1
            obj_ff.idx.iN[name][ref] = obj_ff.N + N
            obj_ff.idx.N[name][ref] = N
            obj_ff.N = int(obj_ff.idx.iN[name][ref])
        obj_ff.NS += 1
        obj_ff.order.append({"name": name, "idx": idx[:]})
        setattr(self, set_type, obj_ff)

    def get(self, *fields: Any) -> Any:
        val: Any = self
        for member in fields:
            if isinstance(member, str):
                val = getattr(val, member) if not isinstance(val, dict) else val[member]
            else:
                val = val[member]
        return val

    def getN(self, set_type: str, name: str | None = None, idx: list[int] | tuple[int, ...] | None = None) -> int:
        obj_ff: _NamedSet = getattr(self, set_type)
        if name is None:
            return obj_ff.N
        if name not in obj_ff.idx.N:
            return 0
        if not idx:
            raw = obj_ff.idx.N[name]
            scalar = raw if np.isscalar(raw) else np.asarray(raw).reshape(-1)[0]
            return int(np.asarray(scalar).item())
        scalar = obj_ff.idx.N[name][tuple(i - 1 for i in idx)]
        return int(np.asarray(scalar).item())

    def get_idx(self, *set_types: str) -> Any:
        if not set_types:
            return ()
        return tuple(getattr(self, st).idx for st in set_types)

    def get_userdata(self, name: str) -> Any:
        return self.userdata.get(name, [])

    def describe_idx(self, set_type: str, idxs: np.ndarray | list[int] | int) -> Any:
        values = np.atleast_1d(np.asarray(idxs, dtype=int))
        out: list[str] = []
        obj_ff: _NamedSet = getattr(self, set_type)
        for ii in values.reshape(-1):
            if ii > obj_ff.N:
                raise ValueError(f"@mp_idx_manager/describe_idx: index exceeds maximum {set_type} index ({obj_ff.N})")
            if ii < 1:
                raise ValueError("@mp_idx_manager/describe_idx: index must be positive")
            label = ""
            for entry in reversed(obj_ff.order):
                name = entry["name"]
                idx = entry["idx"]
                if not idx:
                    if ii >= int(obj_ff.idx.i1[name]):
                        label = f"{name}({ii - int(obj_ff.idx.i1[name]) + 1})"
                        break
                else:
                    start = int(obj_ff.idx.i1[name][tuple(i - 1 for i in idx)])
                    if ii >= start:
                        idxstr = ",".join(str(i) for i in idx)
                        label = f"{name}({idxstr})({ii - start + 1})"
                        break
            out.append(label)
        return out[0] if np.isscalar(idxs) else np.array(out, dtype=object).reshape(values.shape)
