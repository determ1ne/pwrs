# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause


def nlp_costfcn(om, x, nargout=1):
    if nargout == 3:
        f, df, d2f = om.eval_nln_cost(x)
        if om.qdc.NS:
            fq, dfq, d2fq = om.eval_quad_cost(x)
            f = f + fq
            df = df + dfq
            d2f = d2f + d2fq
        outputs = (f, df, d2f)
    elif nargout == 2:
        f, df = om.eval_nln_cost(x)[:2]
        if om.qdc.NS:
            fq, dfq = om.eval_quad_cost(x)[:2]
            f = f + fq
            df = df + dfq
        outputs = (f, df)
    else:
        f = om.eval_nln_cost(x)[0]
        if om.qdc.NS:
            fq = om.eval_quad_cost(x)[0]
            f = f + fq
        outputs = (f,)
    return outputs[:nargout] if nargout > 1 else outputs[0]
