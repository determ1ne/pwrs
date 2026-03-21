# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from ..utils import get_nested


def _as_opt_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return dict(value or {})


def mpopt2qpopt(mpopt: dict[str, Any], model: str = "", alg: str = "") -> dict[str, Any]:
    """Translate MATPOWER options into QP solver options.

    Mirrors MP-Opt-Model's ``mpopt2qpopt`` helper by selecting the effective
    QP solver algorithm and extracting the corresponding backend option
    sub-struct from ``mpopt``.

    Parameters
    ----------
    mpopt : dict
        MATPOWER options struct.
    model : str, optional
        Optimization model type. Present for interface compatibility.
    alg : str, optional
        Explicit solver selection, or ``"opf.dc"``/empty to derive the
        solver from ``mpopt``.

    Returns
    -------
    dict
        Backend-neutral QP options dict containing the selected algorithm and
        its solver-specific options.
    """
    model = str(model or "MIQP").upper()
    if alg in {"", "opf.dc"}:
        alg = str(get_nested(mpopt, ["opf", "dc", "solver"], "DEFAULT")).upper()
    else:
        alg = str(alg).upper()
    if alg == "DEFAULT":
        alg = "MIPS"
    qpopt: dict[str, Any] = {"alg": alg, "verbose": int(get_nested(mpopt, ["verbose"], 0))}
    if alg == "MIPS":
        qpopt["mips_opt"] = _as_opt_dict(get_nested(mpopt, ["mips"], {}))
        if float(qpopt["mips_opt"].get("feastol", 0)) == 0:
            qpopt["mips_opt"]["feastol"] = float(get_nested(mpopt, ["opf", "violation"], 5e-6))
    elif alg == "GLPK":
        qpopt["glpk_opt"] = _as_opt_dict(get_nested(mpopt, ["glpk", "opts"], {}))
    elif alg == "IPOPT":
        qpopt["ipopt_opt"] = _as_opt_dict(get_nested(mpopt, ["ipopt", "opts"], {}))
    return qpopt
