# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def gentypes():
    """Return the standard MATPOWER generator type codes.

    Provides the canonical list of generator technology/type strings used by
    MATPOWER case data and helper utilities.

    Returns
    -------
    numpy.ndarray
        Column vector of generator type code strings.
    """
    gt = [
        "BA",
        "CE",
        "CP",
        "FW",
        "PS",
        "ES",
        "ST",
        "GT",
        "IC",
        "CA",
        "CT",
        "CS",
        "CC",
        "HA",
        "HB",
        "HK",
        "HY",
        "BT",
        "PV",
        "WT",
        "WS",
        "FC",
        "OT",
        "UN",
        "JE",
        "NB",
        "NG",
        "NH",
        "NP",
        "IT",
        "SC",
        "DC",
        "MP",
        "W1",
        "W2",
        "W3",
        "W4",
        "SV",
        "DL",
    ]
    return np.asarray(gt, dtype=object).reshape(-1, 1)
