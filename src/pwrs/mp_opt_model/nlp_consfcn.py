# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from scipy import sparse


def nlp_consfcn(om, x, dhs=None, dgs=None, nargout=1):
    if nargout == 2:
        g = om.eval_nln_constraint(x, 1)[0]
        h = om.eval_nln_constraint(x, 0)[0]
        outputs = (h, g)
    else:
        g, dg = om.eval_nln_constraint(x, 1)
        h, dh = om.eval_nln_constraint(x, 0)
        dg = dg.T
        dh = dh.T
        if dhs is not None and dgs is not None:
            dg = dg + sparse.csc_matrix(dgs)
            dh = dh + sparse.csc_matrix(dhs)
        outputs = (h, g, dh, dg)
    return outputs[:nargout] if nargout > 1 else outputs[0]
