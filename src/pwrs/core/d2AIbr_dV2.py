# Copyright (c) 2008-2019, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from .d2Abr_dV2 import d2Abr_dV2
from .d2Ibr_dV2 import d2Ibr_dV2_full


def d2AIbr_dV2(dIbr_dV1, dIbr_dV2, Ibr, Ybr, V, mu, vcart=0, nargout=1):
    """Compatibility wrapper for Hessians of squared branch-current magnitudes."""
    second_derivative = lambda voltage, multipliers: d2Ibr_dV2_full(Ybr, voltage, multipliers, vcart)
    return d2Abr_dV2(second_derivative, dIbr_dV1, dIbr_dV2, Ibr, V, mu, nargout)


def d2AIbr_dV2_full(dIbr_dV1, dIbr_dV2, Ibr, Ybr, V, mu, vcart=0):
    """Return all four Hessian blocks for squared branch-current magnitudes."""
    return d2AIbr_dV2(dIbr_dV1, dIbr_dV2, Ibr, Ybr, V, mu, vcart, nargout=4)
