# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy

from ..corex import MatpowerConfig, NlpOptions


def mpopt2nlpopt(mpopt: MatpowerConfig, model: str = "", alg: str = "") -> NlpOptions:
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

    nlpopt: NlpOptions = {"alg": alg, "verbose": mpopt.verbose}
    if alg == "MIPS":
        mips_opt = deepcopy(mpopt.mips)
        if mips_opt.feastol == 0:
            mips_opt.feastol = mpopt.opf.violation
        if mips_opt.cost_mult is None:
            mips_opt.cost_mult = 1e-4
        nlpopt["mips_opt"] = mips_opt
    elif alg == "FMINCON":
        fmincon_opt = mpopt.fmincon.to_dict()
        native_options = dict(mpopt.fmincon.opts)
        native_options["TolCon"] = mpopt.opf.violation
        fmincon_opt["opts"] = native_options
        nlpopt["fmincon_opt"] = fmincon_opt
    elif alg == "IPOPT":
        nlpopt["ipopt_opt"] = {} if mpopt.ipopt is None else dict(mpopt.ipopt.opts)
    elif alg == "KNITRO":
        knitro_opt = mpopt.knitro.to_dict()
        native_options = dict(mpopt.knitro.opts)
        native_options["feastol"] = mpopt.opf.violation
        knitro_opt["opts"] = native_options
        nlpopt["knitro_opt"] = knitro_opt

    return nlpopt
