# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from ..mp_opt_model.qps_master import qps_master


def qps_matpower(*args, nargout=1):
    """Deprecated wrapper for :func:`qps_master`.

    Parameters
    ----------
    *args
        Positional arguments forwarded directly to :func:`qps_master`.
    nargout : int, optional
        MATLAB-style output count forwarded to :func:`qps_master`.

    Returns
    -------
    Any
        Whatever :func:`qps_master` returns for the requested ``nargout``.

    Notes
    -----
    This wrapper is preserved for MATPOWER compatibility only. New code
    should call :func:`qps_master` directly.
    """
    return qps_master(*args, nargout=nargout)
