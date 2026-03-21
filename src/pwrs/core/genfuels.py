# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def genfuels(*, nargout=None):
    """Return the standard MATPOWER generator fuel codes.

    Provides the canonical list of generator fuel strings used by MATPOWER
    case data and helper utilities.

    Returns
    -------
    numpy.ndarray
        Column vector of generator fuel code strings.
    """
    gf = [
        "biomass",
        "coal",
        "dfo",
        "geothermal",
        "hydro",
        "hydrops",
        "jetfuel",
        "lng",
        "ng",
        "nuclear",
        "oil",
        "refuse",
        "rfo",
        "solar",
        "syncgen",
        "wasteheat",
        "wind",
        "wood",
        "other",
        "unknown",
        "dl",
        "ess",
    ]
    return np.asarray(gf, dtype=object).reshape(-1, 1)
