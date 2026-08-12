# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from .io import from_pypower, read_mat, to_pypower, write_mat
from .mpc import MatpowerCase


def from_matpower_mat(filename: str | PathLike[str], name: str = "mpc") -> MatpowerCase:
    """Load a MATPOWER case from a .mat file.

    Parameters
    ----------
    filename : str
        Path to the .mat file containing the MATPOWER case struct.
    name : str, optional
        Name of the variable in the .mat file that contains the case struct. Defaults to 'mpc'.

    Returns
    -------
    MatpowerCase
        The loaded case struct as a MatpowerCase instance.
    """
    return read_mat(filename, variable_name=name)


def to_matpower_mat(
    fname: str | PathLike[str],
    mpc: MatpowerCase | Mapping[str, Any],
    name: str = "mpc",
) -> Path:
    """Save a MATPOWER case struct to a MAT-file.

    Parameters
    ----------
    fname : str
        Output MAT-file path.
    mpc : dict
        MATPOWER case struct to save.
    """
    return write_mat(mpc, fname, variable_name=name)


__all__ = ["from_matpower_mat", "from_pypower", "to_matpower_mat", "to_pypower"]
