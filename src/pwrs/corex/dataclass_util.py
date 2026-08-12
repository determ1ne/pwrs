# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping, Sequence
from typing import Protocol, Self

import numpy as np


class MatlabStructLike(Protocol):
    """Minimal interface exposed by SciPy for loaded MATLAB structs."""

    _fieldnames: Sequence[str]

    def __getattr__(self, name: str) -> object: ...


class DataclassDictMixin:
    @classmethod
    def from_dict(cls, data: Mapping[str, object] | MatlabStructLike) -> Self:
        if not isinstance(data, Mapping):
            data = {name: getattr(data, name) for name in data._fieldnames}
        field_map = getattr(cls, "__dataclass_fields__", None)
        if not isinstance(field_map, dict):
            raise TypeError(f"{cls.__name__}.from_dict: target must be a dataclass")
        field_names = set(field_map)
        kwargs = {k: v for k, v in data.items() if k in field_names}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {}
        field_map = getattr(type(self), "__dataclass_fields__", None)
        if not isinstance(field_map, dict):
            raise TypeError(f"{type(self).__name__}.to_dict: target must be a dataclass")
        for name in field_map:
            value = getattr(self, name)
            if isinstance(value, DataclassDictMixin):
                result[name] = value.to_dict()
            else:
                result[name] = value
        return result

    def __contains__(self, key):
        try:
            item = getattr(self, key)
        except AttributeError:
            return False

        if item is None:
            return False
        if isinstance(item, np.ndarray):
            return item.size > 0
        if isinstance(item, (list, tuple, dict, set, str, bytes)):
            return len(item) > 0
        return True

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        if key in self:
            return getattr(self, key)
        return default

    def pop(self, key, default=None):
        if key in self:
            value = getattr(self, key)
            setattr(self, key, None)
            return value
        return default
