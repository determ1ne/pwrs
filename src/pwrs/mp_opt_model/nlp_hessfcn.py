# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause


def nlp_hessfcn(om, x, lambda_, cost_mult: float = 1.0, Hs=None, nargout=1):
    _, _, d2f = om.eval_costfcn(x)
    d2f = d2f * cost_mult
    d2G = om.eval_nln_constraint_hess(x, lambda_["eqnonlin"], 1)
    d2H = om.eval_nln_constraint_hess(x, lambda_["ineqnonlin"], 0)
    Lxx = d2f + d2G + d2H
    if Hs is not None:
        Lxx = Lxx + Hs
    outputs = (Lxx,)
    return outputs[:nargout] if nargout > 1 else Lxx
