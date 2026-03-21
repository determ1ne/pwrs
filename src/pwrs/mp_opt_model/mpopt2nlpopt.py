# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from ..corex import MatpowerConfig


def mpopt2nlpopt(mpopt: MatpowerConfig, model="", alg="", nargout=1):
    """Create or modify an ``nlps_master`` options dict from ``mpopt``.

    Parameters
    ----------
    mpopt : MatpowerConfig
        MATPOWER options dict.
    model : str, optional
        Problem class used when resolving ``alg="DEFAULT"``. Typical values
        are ``"NLP"`` and ``"MINLP"``.
    alg : str, optional
        Either ``"opf.ac"`` to read ``mpopt["opf"]["ac"]["solver"]`` or a
        direct ``nlps_master`` algorithm name such as ``"MIPS"`` or
        ``"IPOPT"``.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    dict
        Options dict for use by :func:`nlps_master`.
    """
    if not model:
        model = "NLP"
    else:
        model = str(model).upper()

    if alg in ("opf.ac", ""):
        alg = str(mpopt.opf.ac.solver).upper()
    else:
        alg = str(alg).upper()

    if alg == "DEFAULT":
        alg = "MIPS"
        if model[0] == "M":
            raise ValueError(f"mpopt2nlpopt: Sorry, no solver available for {model} models")

    nlpopt = {"alg": alg, "verbose": mpopt.verbose}
    if alg == "MIPS":
        nlpopt["mips_opt"] = mpopt.mips
        if nlpopt["mips_opt"].feastol == 0:
            nlpopt["mips_opt"].feastol = mpopt.opf.violation
        if nlpopt["mips_opt"].cost_mult is None:
            nlpopt["mips_opt"].cost_mult = 1e-4
    elif alg == "FMINCON":
        nlpopt["fmincon_opt"] = dict(mpopt.fmincon)
        nlpopt["fmincon_opt"].setdefault("opts", {})
        nlpopt["fmincon_opt"]["opts"]["TolCon"] = mpopt.opf.violation
    elif alg == "IPOPT":
        nlpopt["ipopt_opt"] = dict(mpopt.ipopt.opts)
    elif alg == "KNITRO":
        nlpopt["knitro_opt"] = dict(mpopt.knitro)
        nlpopt["knitro_opt"].setdefault("opts", {})
        nlpopt["knitro_opt"]["opts"]["feastol"] = mpopt.opf.violation

    outputs = (nlpopt,)
    return outputs[:nargout] if nargout > 1 else nlpopt
