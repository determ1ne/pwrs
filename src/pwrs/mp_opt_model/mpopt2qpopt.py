# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from ..corex import MatpowerConfig
from ..corex.mpoption import fetch_mpoption


def _as_opt_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return dict(value or {})


def mpopt2qpopt(mpopt: dict[str, Any] | MatpowerConfig, model: str = "", alg: str = "") -> dict[str, Any]:
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
    if isinstance(mpopt, MatpowerConfig):
        mpopt_cfg = mpopt
    else:
        from ..core.mpoption import mpoption

        mpopt_cfg = mpoption(mpopt)
    if alg in {"", "opf.dc"}:
        alg = mpopt_cfg.opf.dc.solver.upper()
    else:
        alg = str(alg).upper()
    if alg == "DEFAULT":
        alg = "MIPS"
    qpopt: dict[str, Any] = {"alg": alg, "verbose": int(mpopt_cfg.verbose)}
    if alg == "MIPS":
        qpopt["mips_opt"] = _as_opt_dict(mpopt_cfg.mips)
        if float(qpopt["mips_opt"].get("feastol", 0)) == 0:
            qpopt["mips_opt"]["feastol"] = float(mpopt_cfg.opf.violation)
    elif alg == "GLPK":
        qpopt["glpk_opt"] = _as_opt_dict(fetch_mpoption(mpopt_cfg, dict, "glpk.opts"))
    elif alg == "IPOPT":
        qpopt["ipopt_opt"] = _as_opt_dict(mpopt_cfg.ipopt.opts if mpopt_cfg.ipopt is not None else {})
    return qpopt
