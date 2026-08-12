# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from typing import Any

import numpy as np
from scipy import sparse

from .add_userfcn import add_userfcn
from .e2i_field import e2i_field
from .i2e_data import i2e_data
from .i2e_field import i2e_field
from .idx_gen import PMAX, RAMP_10
from .remove_userfcn import remove_userfcn


def _status(mpc: dict[str, Any]) -> bool:
    return bool(mpc.get("userfcn", {}).get("status", {}).get("reserves", 0))


def toggle_reserves(mpc: dict[str, Any], on_off: str) -> dict[str, Any] | bool:
    """Enable, disable, or query fixed zonal reserve callbacks."""
    action = on_off.upper()
    if action == "STATUS":
        return _status(mpc)
    if action not in {"ON", "OFF"}:
        raise ValueError("toggle_reserves: second argument must be 'on', 'off' or 'status'")

    if action == "ON":
        reserve = mpc.get("reserves")
        if not isinstance(reserve, dict) or not {"zones", "req", "cost"} <= reserve.keys():
            raise ValueError(
                "toggle_reserves: case must contain a 'reserves' mapping defining 'zones', 'req' and 'cost'"
            )
        callbacks = (
            ("ext2int", userfcn_reserves_ext2int),
            ("formulation", userfcn_reserves_formulation),
            ("int2ext", userfcn_reserves_int2ext),
            ("printpf", userfcn_reserves_printpf),
            ("savecase", userfcn_reserves_savecase),
        )
        result = mpc
        for stage, callback in callbacks:
            result = add_userfcn(result, stage, callback)
        result.setdefault("userfcn", {}).setdefault("status", {})["reserves"] = 1
        return result

    result = deepcopy(mpc)
    callbacks = (
        ("savecase", userfcn_reserves_savecase),
        ("printpf", userfcn_reserves_printpf),
        ("int2ext", userfcn_reserves_int2ext),
        ("formulation", userfcn_reserves_formulation),
        ("ext2int", userfcn_reserves_ext2int),
    )
    if "userfcn" in result:
        for stage, callback in callbacks:
            if stage in result["userfcn"]:
                result = remove_userfcn(result, stage, callback)
    result.setdefault("userfcn", {}).setdefault("status", {})["reserves"] = 0
    return result


def userfcn_reserves_ext2int(mpc: dict[str, Any], mpopt: object, args: object) -> dict[str, Any]:
    """Validate reserve data and convert generator-indexed fields to internal order."""
    del mpopt, args
    reserve = mpc["reserves"]
    zones = np.atleast_2d(np.asarray(reserve["zones"], dtype=float))
    requirements = np.asarray(reserve["req"], dtype=float).reshape(-1)
    cost = np.asarray(reserve["cost"], dtype=float).reshape(-1)
    original_gen_count = np.asarray(mpc["order"]["ext"]["gen"]).shape[0]
    reserve_generators = np.any(zones != 0, axis=0)
    reserve_indices = np.flatnonzero(reserve_generators)

    if zones.shape[0] != requirements.size:
        raise ValueError("userfcn_reserves_ext2int: reserves.req and reserves.zones row counts must match")
    if zones.shape[1] != original_gen_count:
        raise ValueError("userfcn_reserves_ext2int: reserves.zones must have one column per generator")
    if cost.size not in {original_gen_count, reserve_indices.size}:
        raise ValueError(
            "userfcn_reserves_ext2int: reserves.cost must have one row per generator or reserve generator"
        )
    if "qty" in reserve and np.asarray(reserve["qty"]).size != cost.size:
        raise ValueError("userfcn_reserves_ext2int: reserves.cost and reserves.qty must have equal lengths")

    reserve["zones"] = zones
    reserve["req"] = requirements
    reserve["rgens"] = reserve_generators.astype(float)
    reserve["cost"] = cost
    if cost.size < original_gen_count:
        reserve["original"] = {"cost": cost.copy()}
        expanded_cost = np.zeros(original_gen_count)
        expanded_cost[reserve_indices] = cost
        reserve["cost"] = expanded_cost
        if "qty" in reserve:
            qty = np.asarray(reserve["qty"], dtype=float).reshape(-1)
            reserve["original"]["qty"] = qty.copy()
            expanded_qty = np.zeros(original_gen_count)
            expanded_qty[reserve_indices] = qty
            reserve["qty"] = expanded_qty
    elif "qty" in reserve:
        reserve["qty"] = np.asarray(reserve["qty"], dtype=float).reshape(-1)

    if "qty" in reserve:
        mpc = e2i_field(mpc, ["reserves", "qty"], "gen")
    mpc = e2i_field(mpc, ["reserves", "cost"], "gen")
    mpc = e2i_field(mpc, ["reserves", "zones"], "gen", 2)
    mpc = e2i_field(mpc, ["reserves", "rgens"], "gen", 1)
    mpc["order"]["ext"].setdefault("reserves", {})["igr"] = reserve_indices + 1
    mpc["reserves"]["igr"] = np.flatnonzero(np.asarray(mpc["reserves"]["rgens"]).reshape(-1)) + 1
    return mpc


def userfcn_reserves_formulation(om: Any, mpopt: object, args: object) -> Any:
    """Add reserve variables, capability constraints, requirements, and costs."""
    del mpopt, args
    mpc = om.get_mpc()
    reserve = mpc["reserves"]
    indices = np.asarray(reserve["igr"], dtype=int).reshape(-1) - 1
    reserve_count = indices.size
    generator_count = np.asarray(mpc["gen"]).shape[0]
    base_mva = float(mpc["baseMVA"])

    lower = np.zeros(reserve_count)
    upper = np.full(reserve_count, np.inf)
    ramp = np.asarray(mpc["gen"])[indices, RAMP_10]
    positive_ramp = ramp != 0
    upper[positive_ramp] = ramp[positive_ramp]
    if "qty" in reserve:
        quantity = np.asarray(reserve["qty"], dtype=float).reshape(-1)[indices]
        upper = np.minimum(upper, quantity)
    upper /= base_mva

    selector = sparse.csc_matrix(
        (np.ones(reserve_count), (np.arange(reserve_count), indices)),
        shape=(reserve_count, generator_count),
    )
    capability = sparse.hstack([selector, sparse.eye(reserve_count, format="csc")], format="csc")
    capability_upper = np.asarray(mpc["gen"])[indices, PMAX] / base_mva
    requirements = np.asarray(reserve["req"], dtype=float).reshape(-1) / base_mva
    zones = sparse.csc_matrix(np.asarray(reserve["zones"], dtype=float)[:, indices])
    cost = np.asarray(reserve["cost"], dtype=float).reshape(-1)[indices] * base_mva

    om.add_var("R", reserve_count, [], lower, upper)
    om.add_lin_constraint("Pg_plus_R", capability, [], capability_upper, ["Pg", "R"])
    om.add_lin_constraint("Rreq", zones, requirements, [], ["R"])
    om.add_quad_cost("Rcost", sparse.csc_matrix((0, 0)), cost, 0, ["R"])
    return om


def userfcn_reserves_int2ext(results: dict[str, Any], mpopt: object, args: object) -> dict[str, Any]:
    """Convert reserve data and package reserve quantities, prices, and multipliers."""
    del mpopt, args
    reserve = results["reserves"]
    internal_indices = np.asarray(reserve["igr"], dtype=int).reshape(-1) - 1
    internal_gen_count = np.asarray(results["gen"]).shape[0]

    if "qty" in reserve:
        results = i2e_field(results, ["reserves", "qty"], "gen")
    results = i2e_field(results, ["reserves", "cost"], "gen")
    results = i2e_field(results, ["reserves", "zones"], "gen", 2)
    results = i2e_field(results, ["reserves", "rgens"], "gen", 1)
    results["order"].setdefault("int", {}).setdefault("reserves", {})["igr"] = internal_indices + 1
    results["reserves"]["igr"] = np.asarray(results["order"]["ext"]["reserves"]["igr"], dtype=int)
    reserve = results["reserves"]
    external_indices = np.asarray(reserve["igr"], dtype=int).reshape(-1) - 1
    external_gen_count = np.asarray(results["order"]["ext"]["gen"]).shape[0]
    base_mva = float(results["baseMVA"])

    _, reserve_lower, reserve_upper, _ = results["om"].params_var("R")
    values = np.zeros(internal_gen_count)
    minimum = np.zeros(internal_gen_count)
    maximum = np.zeros(internal_gen_count)
    mu_lower = np.zeros(internal_gen_count)
    mu_upper = np.zeros(internal_gen_count)
    mu_pmax = np.zeros(internal_gen_count)
    values[internal_indices] = np.asarray(results["var"]["val"]["R"]) * base_mva
    minimum[internal_indices] = reserve_lower * base_mva
    maximum[internal_indices] = reserve_upper * base_mva
    mu_lower[internal_indices] = np.asarray(results["var"]["mu"]["l"]["R"]) / base_mva
    mu_upper[internal_indices] = np.asarray(results["var"]["mu"]["u"]["R"]) / base_mva
    mu_pmax[internal_indices] = np.asarray(results["lin"]["mu"]["u"]["Pg_plus_R"]) / base_mva
    zero_external = np.zeros(external_gen_count)

    reserve["R"] = np.asarray(i2e_data(results, values, zero_external, "gen")).reshape(-1)
    reserve["Rmin"] = np.asarray(i2e_data(results, minimum, zero_external, "gen")).reshape(-1)
    reserve["Rmax"] = np.asarray(i2e_data(results, maximum, zero_external, "gen")).reshape(-1)
    reserve["mu"] = {
        "l": np.asarray(i2e_data(results, mu_lower, zero_external, "gen")).reshape(-1),
        "u": np.asarray(i2e_data(results, mu_upper, zero_external, "gen")).reshape(-1),
        "Pmax": np.asarray(i2e_data(results, mu_pmax, zero_external, "gen")).reshape(-1),
    }
    reserve["prc"] = np.zeros(external_gen_count)
    zones = np.asarray(reserve["zones"], dtype=float)
    requirement_prices = np.asarray(results["lin"]["mu"]["l"]["Rreq"], dtype=float)
    for generator in external_indices:
        memberships = np.flatnonzero(zones[:, generator] != 0)
        reserve["prc"][generator] = np.sum(requirement_prices[memberships]) / base_mva
    reserve["totalcost"] = float(np.sum(results["qdc"]["Rcost"]))

    original = reserve.pop("original", None)
    if isinstance(original, dict):
        reserve["cost"] = original["cost"]
        if "qty" in original:
            reserve["qty"] = original["qty"]
    return results


def userfcn_reserves_printpf(results: dict[str, Any], fd: object, mpopt: object, args: object) -> dict[str, Any]:
    """Retain the MATPOWER print callback stage without duplicating ``printpf`` output."""
    del fd, mpopt, args
    return results


def userfcn_reserves_savecase(mpc: dict[str, Any], lines: object, prefix: object, args: object) -> dict[str, Any]:
    """Retain the callback stage; full legacy ``savecase`` output is out of scope."""
    del lines, prefix, args
    return mpc
