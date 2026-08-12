# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime normalization helpers for typed core inputs."""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy import sparse

from .types import ArrayLike, ComplexArray, DenseMatrix, FloatArray, IntArray, Matrix, SparseMatrix


def as_float_matrix(value: object, *, name: str = "value") -> FloatArray:
    """Convert an input to a two-dimensional floating-point matrix."""
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    return result


def as_float_scalar(value: object, *, name: str = "value") -> float:
    """Convert a scalar-like value to ``float`` and reject non-scalars."""
    result = np.asarray(value)
    if result.size != 1:
        raise ValueError(f"{name} must be scalar")
    return float(result.reshape(-1)[0])


def as_complex_vector(value: object, *, name: str = "value") -> ComplexArray:
    """Convert an input to a one-dimensional complex vector."""
    result = np.asarray(value, dtype=np.complex128).reshape(-1)
    return result


def as_index_vector(value: object, *, name: str = "value", one_based: bool = True) -> IntArray:
    """Convert an index input to an integer vector and validate its domain."""
    result = np.asarray(value, dtype=np.int64).reshape(-1)
    if one_based and np.any(result < 1):
        raise ValueError(f"{name} must contain positive one-based indices")
    if not one_based and np.any(result < 0):
        raise ValueError(f"{name} must contain non-negative indices")
    return result


def as_dense_matrix(value: Matrix) -> DenseMatrix:
    """Return a dense representation without leaking SciPy's untyped API."""
    if isinstance(value, np.ndarray):
        return value
    return value.toarray()


def as_csc_matrix(value: object) -> SparseMatrix:
    """Normalize a value to a CSC matrix behind the SciPy typing boundary."""
    return cast(SparseMatrix, sparse.csc_matrix(value))


def as_csr_matrix(value: object) -> SparseMatrix:
    """Normalize a value to a CSR matrix behind the SciPy typing boundary."""
    return cast(SparseMatrix, sparse.csr_matrix(value))


def matrix_real(value: Matrix) -> Matrix:
    """Return the real part of a dense or sparse matrix."""
    if isinstance(value, np.ndarray):
        return np.real(value)
    return as_csc_matrix(value.real)


def matrix_imag(value: Matrix) -> Matrix:
    """Return the imaginary part of a dense or sparse matrix."""
    if isinstance(value, np.ndarray):
        return np.imag(value)
    return as_csc_matrix(value.imag)


def subtract_matrices(left: Matrix, right: Matrix) -> Matrix:
    """Subtract matrices while preserving sparse storage when needed."""
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left - right
    return as_csc_matrix(sparse.csc_matrix(left) - sparse.csc_matrix(right))


def add_matrices(left: Matrix, right: Matrix) -> Matrix:
    """Add matrices while preserving sparse storage when needed."""
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left + right
    return as_csc_matrix(sparse.csc_matrix(left) + sparse.csc_matrix(right))


def complex_matvec(matrix: Matrix, vector: ArrayLike) -> ComplexArray:
    """Multiply a dense/sparse matrix by a vector and normalize the result."""
    vector_array = np.asarray(vector, dtype=np.complex128).reshape(-1)
    return np.asarray(matrix @ vector_array, dtype=np.complex128).reshape(-1)
