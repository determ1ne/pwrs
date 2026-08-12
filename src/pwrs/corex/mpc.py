# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .dataclass_util import DataclassDictMixin
from .types import FloatArray, IntArray, Matrix


@dataclass
class OrderStatus(DataclassDictMixin):
    on: IntArray = field(default_factory=lambda: np.empty((0, 1), dtype=np.int64))
    off: IntArray = field(default_factory=lambda: np.empty((0, 1), dtype=np.int64))


@dataclass
class OrderBusGenInfo(DataclassDictMixin):
    e2i: IntArray = field(default_factory=lambda: np.empty((0, 1), dtype=np.int64))
    i2e: IntArray = field(default_factory=lambda: np.empty((0, 1), dtype=np.int64))
    status: OrderStatus = field(default_factory=OrderStatus)


@dataclass
class OrderBranchInfo(DataclassDictMixin):
    status: OrderStatus = field(default_factory=OrderStatus)


@dataclass
class OrderCaseData(DataclassDictMixin):
    bus: FloatArray = field(default_factory=lambda: np.empty((0, 0)))
    branch: FloatArray = field(default_factory=lambda: np.empty((0, 0)))
    gen: FloatArray = field(default_factory=lambda: np.empty((0, 0)))
    gencost: FloatArray | None = None
    A: Matrix | None = None
    N: Matrix | None = None


@dataclass
class OrderInfo(DataclassDictMixin):
    state: Literal["i", "e"] | None = None
    ext: OrderCaseData | None = None
    int: OrderCaseData | None = None
    bus: OrderBusGenInfo = field(default_factory=OrderBusGenInfo)
    gen: OrderBusGenInfo = field(default_factory=OrderBusGenInfo)
    branch: OrderBranchInfo = field(default_factory=OrderBranchInfo)


@dataclass
class MatpowerCase(DataclassDictMixin):
    version: str = "2"
    baseMVA: float = 100.0
    bus: FloatArray = field(default_factory=lambda: np.empty((0, 0)))
    gen: FloatArray = field(default_factory=lambda: np.empty((0, 0)))
    branch: FloatArray = field(default_factory=lambda: np.empty((0, 0)))
    gencost: FloatArray | None = None
    dcline: FloatArray | None = None
    dclinecost: FloatArray | None = None

    areas: FloatArray | None = None
    A: Matrix | None = None
    l: FloatArray | None = None
    u: FloatArray | None = None
    N: Matrix | None = None
    fparm: FloatArray | None = None
    H: Matrix | None = None
    Cw: FloatArray | None = None
    z0: FloatArray | None = None
    zl: FloatArray | None = None
    zu: FloatArray | None = None
    gentype: list[str] | None = None
    genfuel: list[str] | None = None
    bus_name: list[str] | None = None
    branch_name: list[str] | None = None
    comments: list[str] | None = None
    # Deprecated constructor aliases. Serialized data always uses *_name.
    busname: list[str] | None = None
    branchname: list[str] | None = None

    order: OrderInfo | None = None
    et: float = 0.0
    success: bool = True
    iterations: int = 4
    f: float | None = None

    def __post_init__(self) -> None:
        for canonical, legacy in (("bus_name", "busname"), ("branch_name", "branchname")):
            canonical_value = getattr(self, canonical)
            legacy_value = getattr(self, legacy)
            if canonical_value is not None and legacy_value is not None and canonical_value != legacy_value:
                raise ValueError(f"MatpowerCase received conflicting {canonical!r} and {legacy!r} values")
            if canonical_value is None and legacy_value is not None:
                setattr(self, canonical, legacy_value)

    @classmethod
    def from_dict(cls, data: Any) -> MatpowerCase:
        if hasattr(data, "_fieldnames"):
            data = {name: getattr(data, name) for name in data._fieldnames}
        else:
            data = dict(data)
        for canonical, legacy in (("bus_name", "busname"), ("branch_name", "branchname")):
            if canonical in data and legacy in data and data[canonical] != data[legacy]:
                raise ValueError(f"MatpowerCase received conflicting {canonical!r} and {legacy!r} values")
            if canonical not in data and legacy in data:
                data[canonical] = data[legacy]
        result = super().from_dict(data)
        field_names = set(cls.__dataclass_fields__)
        for key, value in data.items():
            if key not in field_names:
                setattr(result, key, value)
        return result

    def to_dict(self) -> dict[str, object]:
        """Return only fields present in the MATPOWER case struct."""
        result = {
            key: value
            for key, value in super().to_dict().items()
            if value is not None and key not in {"busname", "branchname"}
        }
        field_names = set(type(self).__dataclass_fields__)
        result.update({key: value for key, value in vars(self).items() if key not in field_names})
        return result
