# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import fields
from typing import Any

import numpy as np


class DataclassDictMixin:
    @classmethod
    def from_dict[T](cls: type[T], data: dict[str, Any]) -> T:
        field_names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in field_names}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, "to_dict"):
                result[f.name] = value.to_dict()
            else:
                result[f.name] = value
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
