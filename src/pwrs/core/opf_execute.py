# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, cast

import numpy as np

from ..corex import MatpowerConfig
from .dcopf_solver import dcopf_solver_full
from .idx_brch import MU_ANGMAX, MU_ANGMIN
from .idx_bus import MU_VMAX, MU_VMIN, VM
from .idx_gen import GEN_BUS, VG
from .mpoption import mpoption
from .nlpopf_solver import nlpopf_solver_full
from .update_mupq import update_mupq


def opf_execute(om, mpopt: MatpowerConfig, nargout=1):
    """Execute the OPF described by an OPF model object.

    Parameters
    ----------
    om : OPFModel
        OPF model object produced by :func:`opf_setup`.
    mpopt : dict
        MATPOWER options dict controlling AC/DC model selection, solver
        choice, and formulation details.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    dict or tuple
        Returns internal-order results, success flag, and raw solver output.
        The results are kept in internal indexing with in-service equipment
        only, matching MATLAB ``opf_execute``.
    """
    if not isinstance(mpopt, MatpowerConfig):
        mpopt = mpoption(mpopt)

    dc = mpopt.model.upper() == "DC"
    alg = mpopt.opf.ac.solver.upper()
    if alg == "DEFAULT":
        alg = "MIPS"
    sdp = alg == "SDPOPF"
    vcart = (not dc) and bool(float(mpopt.opf.v_cartesian))

    vv, ll, nne, nni = om.get_idx("var", "lin", "nle", "nli")

    if dc:
        results, success, raw = cast(tuple[dict[str, Any], float, dict[str, Any]], dcopf_solver_full(om, mpopt))
    else:
        results, success, raw = cast(tuple[dict[str, Any], float, dict[str, Any]], nlpopf_solver_full(om, mpopt))

    if raw.get("output", {}).get("alg") in (None, ""):
        raw.setdefault("output", {})["alg"] = alg

    if success and (not dc) and (not sdp):
        results["gen"][:, VG - 1] = results["bus"][results["gen"][:, GEN_BUS - 1].astype(int) - 1, VM - 1]
        if vcart:
            results["bus"][:, MU_VMIN - 1] = results["bus"][:, MU_VMIN - 1] * results["bus"][:, VM - 1] * 2
            results["bus"][:, MU_VMAX - 1] = results["bus"][:, MU_VMAX - 1] * results["bus"][:, VM - 1] * 2
        if ll.N.get("PQh", 0) > 0 or ll.N.get("PQl", 0) > 0:
            mu_PQh = (
                results["mu"]["lin"]["l"][ll.i1["PQh"] - 1 : ll.iN["PQh"]]
                - results["mu"]["lin"]["u"][ll.i1["PQh"] - 1 : ll.iN["PQh"]]
            )
            mu_PQl = (
                results["mu"]["lin"]["l"][ll.i1["PQl"] - 1 : ll.iN["PQl"]]
                - results["mu"]["lin"]["u"][ll.i1["PQl"] - 1 : ll.iN["PQl"]]
            )
            results["gen"] = update_mupq(
                results["baseMVA"], results["gen"], mu_PQh, mu_PQl, om.get_userdata("Apqdata")
            )
        iang = np.asarray(om.get_userdata("iang")).reshape(-1).astype(int)
        if iang.size:
            if vcart:
                results["branch"][iang - 1, MU_ANGMIN - 1] = (
                    results["mu"]["nli"][nni.i1["angL"] - 1 : nni.iN["angL"]] * np.pi / 180
                )
                results["branch"][iang - 1, MU_ANGMAX - 1] = (
                    results["mu"]["nli"][nni.i1["angU"] - 1 : nni.iN["angU"]] * np.pi / 180
                )
            else:
                results["branch"][iang - 1, MU_ANGMIN - 1] = (
                    results["mu"]["lin"]["l"][ll.i1["ang"] - 1 : ll.iN["ang"]] * np.pi / 180
                )
                results["branch"][iang - 1, MU_ANGMAX - 1] = (
                    results["mu"]["lin"]["u"][ll.i1["ang"] - 1 : ll.iN["ang"]] * np.pi / 180
                )

    results["var"] = {"val": {}, "mu": {"l": {}, "u": {}}}
    for entry in om.get("var", "order"):
        name = entry["name"]
        if om.getN("var", name, entry["idx"]):
            i1 = vv.i1[name] if not entry["idx"] else vv.i1[name][tuple(i - 1 for i in entry["idx"])]
            iN = vv.iN[name] if not entry["idx"] else vv.iN[name][tuple(i - 1 for i in entry["idx"])]
            sl = slice(i1 - 1, iN)
            results["var"]["val"][name] = results["x"][sl]
            results["var"]["mu"]["l"][name] = results["mu"]["var"]["l"][sl]
            results["var"]["mu"]["u"][name] = results["mu"]["var"]["u"][sl]

    results["lin"] = {"mu": {"l": {}, "u": {}}}
    for entry in om.get("lin", "order"):
        name = entry["name"]
        if om.getN("lin", name, entry["idx"]):
            i1 = ll.i1[name] if not entry["idx"] else ll.i1[name][tuple(i - 1 for i in entry["idx"])]
            iN = ll.iN[name] if not entry["idx"] else ll.iN[name][tuple(i - 1 for i in entry["idx"])]
            sl = slice(i1 - 1, iN)
            results["lin"]["mu"]["l"][name] = results["mu"]["lin"]["l"][sl]
            results["lin"]["mu"]["u"][name] = results["mu"]["lin"]["u"][sl]

    if not dc:
        results["nle"] = {"lambda": {}}
        for entry in om.get("nle", "order"):
            name = entry["name"]
            if om.getN("nle", name, entry["idx"]):
                i1 = nne.i1[name] if not entry["idx"] else nne.i1[name][tuple(i - 1 for i in entry["idx"])]
                iN = nne.iN[name] if not entry["idx"] else nne.iN[name][tuple(i - 1 for i in entry["idx"])]
                results["nle"]["lambda"][name] = results["mu"]["nle"][i1 - 1 : iN]
        results["nli"] = {"mu": {}}
        for entry in om.get("nli", "order"):
            name = entry["name"]
            if om.getN("nli", name, entry["idx"]):
                i1 = nni.i1[name] if not entry["idx"] else nni.i1[name][tuple(i - 1 for i in entry["idx"])]
                iN = nni.iN[name] if not entry["idx"] else nni.iN[name][tuple(i - 1 for i in entry["idx"])]
                results["nli"]["mu"][name] = results["mu"]["nli"][i1 - 1 : iN]

    if om.getN("qdc"):
        results["qdc"] = {}
        for entry in om.get("qdc", "order"):
            name = entry["name"]
            if om.getN("qdc", name, entry["idx"]):
                results["qdc"][name] = om.eval_quad_cost(results["x"], name, entry["idx"] or None)[0]
    if om.getN("nlc"):
        results["nlc"] = {}
        for entry in om.get("nlc", "order"):
            name = entry["name"]
            if om.getN("nlc", name, entry["idx"]):
                results["nlc"][name] = om.eval_nln_cost(results["x"], name, entry["idx"] or None)[0]
    if om.getN("cost"):
        results["cost"] = {}
        for entry in om.get("cost", "order"):
            name = entry["name"]
            if om.getN("cost", name, entry["idx"]):
                results["cost"][name] = om.eval_legacy_cost(results["x"], name, entry["idx"] or None)

    pwl1 = np.asarray(om.get_userdata("pwl1")).reshape(-1).astype(int)
    if pwl1.size:
        nx = vv.iN["Pg"] if dc else vv.iN["Qg"]
        y = np.zeros(len(pwl1))
        raw["xr"] = np.r_[raw["xr"][:nx], y, raw["xr"][nx:]]
        results["x"] = np.r_[results["x"][:nx], y, results["x"][nx:]]

    outputs = (results, success, raw)
    return outputs[:nargout] if nargout > 1 else results


def opf_execute_full(om, mpopt: MatpowerConfig) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Return OPF execution results, success flag and raw data."""
    return cast(tuple[dict[str, Any], float, dict[str, Any]], opf_execute(om, mpopt, nargout=3))
