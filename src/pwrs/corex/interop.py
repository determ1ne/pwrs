# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path

from scipy.io import savemat

from .mpc import MatpowerCase


def from_matpower_mat(filename: str, name: str = "mpc") -> MatpowerCase:
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
    from scipy.io import loadmat

    try:
        mat_data = loadmat(filename, struct_as_record=False, squeeze_me=True)
        if name not in mat_data:
            raise ValueError(f"from_matpower_mat: '{name}' variable not found in {filename}")
        return MatpowerCase.from_dict(mat_data[name])
    except Exception as e:
        raise ValueError(f"from_matpower_mat: error loading case from {filename}") from e


def to_matpower_mat(fname: str, mpc: MatpowerCase):
    """Save a MATPOWER case struct to a MAT-file.

    Parameters
    ----------
    fname : str
        Output MAT-file path.
    mpc : dict
        MATPOWER case struct to save.
    """
    baseMVA = float(mpc.baseMVA)
    bus = mpc.bus
    gen = mpc.gen
    branch = mpc.branch
    areas = mpc.areas
    gencost = mpc.gencost

    path = Path(str(fname))
    fname_out = str(path if path.suffix.lower() == ".mat" else path.with_suffix(".mat"))

    mpc_payload = {}
    mpc_payload["version"] = mpc.version
    mpc_payload["baseMVA"] = baseMVA
    mpc_payload["bus"] = bus
    mpc_payload["gen"] = gen
    mpc_payload["branch"] = branch
    if mpc.areas is not None:
        mpc_payload["areas"] = areas
    if mpc.gencost is not None:
        mpc_payload["gencost"] = gencost
    payload = {"mpc": mpc_payload}

    savemat(fname_out, payload, do_compression=False)


def from_pypower(ppc: dict) -> MatpowerCase:
    """Convert a PYPOWER case dict to a MatpowerCase instance.

    Parameters
    ----------
    ppc : dict
        PYPOWER case dict.

    Returns
    -------
    MatpowerCase
        The converted case as a MatpowerCase instance.
    """
    return MatpowerCase.from_dict(ppc)


def to_pypower(mpc: MatpowerCase) -> dict:
    """Convert a MatpowerCase instance to a PYPOWER case dict.

    Parameters
    ----------
    mpc : MatpowerCase
        The case to convert.

    Returns
    -------
    dict
        The converted case as a PYPOWER case dict.
    """
    return mpc.to_dict()
