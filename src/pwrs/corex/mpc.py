# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .dataclass_util import DataclassDictMixin


@dataclass
class OrderStatus(DataclassDictMixin):
    on: npt.NDArray[np.int64] = field(default_factory=lambda: np.empty((0, 1), dtype=np.int64))
    off: npt.NDArray[np.int64] = field(default_factory=lambda: np.empty((0, 1), dtype=np.int64))


@dataclass
class OrderBusGenInfo(DataclassDictMixin):
    e2i: Any = field(default_factory=dict)
    i2e: npt.NDArray[np.int64] = field(default_factory=lambda: np.empty((0, 1), dtype=np.int64))
    status: OrderStatus = field(default_factory=OrderStatus)


@dataclass
class OrderBranchInfo(DataclassDictMixin):
    status: OrderStatus = field(default_factory=OrderStatus)


@dataclass
class OrderCaseData(DataclassDictMixin):
    bus: Any = field(default_factory=list)
    branch: Any = field(default_factory=list)
    gen: Any = field(default_factory=list)
    gencost: Any = None
    A: Any = None
    N: Any = None


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
    bus: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    gen: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    branch: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    gencost: npt.NDArray[np.float64] | None = None

    areas: npt.NDArray[np.float64] | None = None
    gentype: list[str] | None = None
    genfuel: list[str] | None = None
    busname: list[str] | None = None
    branchname: list[str] | None = None

    order: OrderInfo | None = None
    et: float = 0.0
    success: int = 1
    iterations: int = 4
    f: None = None
