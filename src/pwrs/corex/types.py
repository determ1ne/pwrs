# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

"""Shared type vocabulary for the MATPOWER-compatible core.

The MATLAB interface accepts many array-like values and dynamic structs.  The
aliases in this module describe the normalized values used inside the Python
implementation, while the public compatibility functions may still accept
``npt.ArrayLike`` or mapping values and normalize them at their boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from typing import Any, Protocol, TypedDict, overload

import numpy as np
import numpy.typing as npt

type ArrayLike = npt.ArrayLike
type FloatArray = npt.NDArray[np.float64]
type ComplexArray = npt.NDArray[np.complex128]
type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]

type DenseMatrix = npt.NDArray[np.number]


class SparseMatrix(Protocol):
    """Typed subset of the SciPy sparse matrix API used by ``core``.

    SciPy 1.15 does not ship complete type information for sparse matrices.
    Keeping this protocol local prevents that missing information from
    turning every downstream matrix expression into ``Any``.
    """

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def T(self) -> SparseMatrix: ...

    @property
    def real(self) -> SparseMatrix: ...

    @property
    def imag(self) -> SparseMatrix: ...

    @overload
    def __matmul__(self, other: DenseMatrix) -> DenseMatrix: ...

    @overload
    def __matmul__(self, other: SparseMatrix) -> SparseMatrix: ...

    def __rmatmul__(self, other: object) -> DenseMatrix | SparseMatrix: ...

    def __add__(self, other: object) -> SparseMatrix: ...

    def __sub__(self, other: object) -> SparseMatrix: ...

    def __mul__(self, other: object) -> SparseMatrix: ...

    def __rmul__(self, other: object) -> SparseMatrix: ...

    def __neg__(self) -> SparseMatrix: ...

    def __getitem__(self, key: object) -> SparseMatrix: ...

    def tocsc(self, copy: bool = True) -> SparseMatrix: ...

    def tocsr(self, copy: bool = True) -> SparseMatrix: ...

    def getrow(self, i: int) -> SparseMatrix: ...

    def conjugate(self) -> SparseMatrix: ...

    def toarray(self) -> DenseMatrix: ...


type Matrix = DenseMatrix | SparseMatrix
type CaseMapping = MutableMapping[str, object]
type ReadOnlyCaseMapping = Mapping[str, object]

type SbusFunction = Callable[[FloatArray], ComplexArray | tuple[ComplexArray, Matrix]]


class BasicCaseData(TypedDict):
    """Required fields available on an internal MATPOWER case mapping."""

    baseMVA: float
    bus: FloatArray
    gen: FloatArray
    branch: FloatArray


class CaseData(BasicCaseData, total=False):
    """Optional fields commonly added during MATPOWER processing."""

    version: str
    gencost: FloatArray | None
    dcline: FloatArray | None
    dclinecost: FloatArray | None
    areas: FloatArray | None
    l: FloatArray
    u: FloatArray
    fparm: FloatArray
    H: Matrix
    Cw: FloatArray
    z0: FloatArray
    zl: FloatArray
    zu: FloatArray
    gentype: list[str]
    genfuel: list[str]
    bus_name: list[str]
    branch_name: list[str]
    comments: list[str]
    order: dict[str, Any]
    userfcn: list[Any]
    success: bool
    iterations: int | float
    et: float
    f: float | FloatArray
    Ybus: SparseMatrix
    Yf: SparseMatrix
    Yt: SparseMatrix
    A: Matrix
    N: Matrix


class RequiredResultFields(BasicCaseData):
    """Fields present on every completed PF/OPF result."""

    success: bool
    iterations: int | float
    et: float


class InternalResultData(RequiredResultFields, total=False):
    """Mutable result builder used before public result normalization."""

    version: str
    gencost: FloatArray | None
    dcline: FloatArray | None
    dclinecost: FloatArray | None
    areas: FloatArray | None
    order: dict[str, Any]
    userfcn: list[Any]
    f: float | FloatArray
    Ybus: SparseMatrix
    Yf: SparseMatrix
    Yt: SparseMatrix
    A: Matrix
    N: Matrix
    cpf: dict[str, object]
    raw: dict[str, Any]
    x: FloatArray | ComplexArray
    om: Any


class EventData(TypedDict, total=False):
    """Dynamic event record used by CPF event detection and callbacks."""

    name: str
    idx: IntArray
    zero: bool
    tol: float
    msg: str


class DoneData(TypedDict, total=False):
    flag: int | bool
    msg: str
