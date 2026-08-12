# Copyright (c) 1996-2018, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from typing import Any

import numpy as np
from scipy import sparse

from ..corex import MatpowerConfig, SparseMatrix
from .add_userfcn import add_userfcn
from .idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_STATUS,
    F_BUS,
    MU_ANGMAX,
    MU_ANGMIN,
    MU_SF,
    MU_ST,
    RATE_A,
    T_BUS,
)
from .idx_bus import BUS_I, MU_VMAX, MU_VMIN, VMAX, VMIN
from .idx_gen import (
    GEN_STATUS,
    MU_PMAX,
    MU_PMIN,
    MU_QMAX,
    MU_QMIN,
    PMAX,
    PMIN,
    QMAX,
    QMIN,
)
from .isload import isload
from .makeYbus import makeYbus_full
from .margcost import margcost
from .opf_branch_flow_fcn import opf_branch_flow_fcn_with_jacobian
from .opf_branch_flow_hess import opf_branch_flow_hess
from .remove_userfcn import remove_userfcn

_LIMITS_AC = ("VMAX", "VMIN", "PMAX", "PMIN", "QMAX", "QMIN", "RATE_A", "ANGMAX", "ANGMIN")
_LIMITS_DC = ("PMAX", "PMIN", "RATE_A", "ANGMAX", "ANGMIN")


def _status(mpc: dict[str, Any]) -> bool:
    return bool(mpc.get("userfcn", {}).get("status", {}).get("softlims", 0))


def toggle_softlims(mpc: dict[str, Any], on_off: str) -> dict[str, Any] | bool:
    """Enable, disable, or query MATPOWER soft-limit callbacks."""
    action = on_off.upper()
    if action == "STATUS":
        return _status(mpc)
    if action not in {"ON", "OFF"}:
        raise ValueError("toggle_softlims: second argument must be 'on', 'off' or 'status'")

    callbacks = (
        ("ext2int", userfcn_softlims_ext2int),
        ("formulation", userfcn_softlims_formulation),
        ("int2ext", userfcn_softlims_int2ext),
        ("printpf", userfcn_softlims_printpf),
        ("savecase", userfcn_softlims_savecase),
    )
    if action == "ON":
        result = mpc
        for stage, callback in callbacks:
            result = add_userfcn(result, stage, callback)
        result.setdefault("userfcn", {}).setdefault("status", {})["softlims"] = 1
        return result

    result = deepcopy(mpc)
    if "userfcn" in result:
        for stage, callback in reversed(callbacks):
            if stage in result["userfcn"]:
                result = remove_userfcn(result, stage, callback)
    result.setdefault("userfcn", {}).setdefault("status", {})["softlims"] = 0
    return result


def _is_dc(mpopt: MatpowerConfig) -> bool:
    return mpopt.model.upper() == "DC"


def _limit_metadata(mpopt: MatpowerConfig) -> dict[str, tuple[str, int, int]]:
    metadata = {
        "VMAX": ("bus", VMAX, 1),
        "VMIN": ("bus", VMIN, -1),
        "PMAX": ("gen", PMAX, 1),
        "PMIN": ("gen", PMIN, -1),
        "QMAX": ("gen", QMAX, 1),
        "QMIN": ("gen", QMIN, -1),
        "RATE_A": ("branch", RATE_A, 1),
        "ANGMAX": ("branch", ANGMAX, 1),
        "ANGMIN": ("branch", ANGMIN, -1),
    }
    limits = _LIMITS_DC if _is_dc(mpopt) else _LIMITS_AC
    return {limit: metadata[limit] for limit in limits}


def _default_indices(matrix: np.ndarray, limit: str, column: int) -> np.ndarray:
    if limit in {"VMAX", "VMIN"}:
        return np.arange(1, matrix.shape[0] + 1, dtype=int)
    if limit in {"ANGMAX", "ANGMIN"}:
        values = matrix[:, column]
        return np.flatnonzero((matrix[:, BR_STATUS] > 0) & (values != 0) & (np.abs(values) < 360)) + 1
    if limit == "RATE_A":
        return np.flatnonzero((matrix[:, BR_STATUS] > 0) & (matrix[:, RATE_A] > 0)) + 1
    online_generator = (matrix[:, GEN_STATUS] > 0) & ~isload(matrix)
    if limit != "PMIN":
        online_generator &= ~np.isinf(matrix[:, column])
    return np.flatnonzero(online_generator) + 1


def _expanded_vector(value: object, size: int, field: str, limit: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size == 1:
        return np.full(size, float(vector[0]))
    if vector.size != size:
        raise ValueError(f"softlims_defaults: softlims.{limit}.{field} must be scalar or match idx")
    return vector.copy()


def _softlims_defaults(mpc: dict[str, Any], mpopt: MatpowerConfig) -> dict[str, dict[str, Any]]:
    supplied = mpc.get("softlims", {})
    if not isinstance(supplied, dict):
        raise TypeError("softlims_defaults: mpc.softlims must be a mapping")
    output = deepcopy(supplied)
    metadata = _limit_metadata(mpopt)
    generator_count = np.asarray(mpc["gen"]).shape[0]
    costs = np.asarray(mpc["gencost"], dtype=float)[:generator_count]
    pmax = np.asarray(mpc["gen"], dtype=float)[:, PMAX]
    marginal = np.asarray(margcost(costs, pmax), dtype=float).reshape(-1)
    max_generator_cost = float(np.max(marginal)) if marginal.size else 0.0
    base_cost = max(1000.0, 2.0 * max_generator_cost)
    cost_scale = {"VMAX": 100.0, "VMIN": 100.0}
    shift_default = {"VMAX": 0.25, "VMIN": 0.25}

    for limit, (matrix_name, column, direction) in metadata.items():
        specified = limit in supplied
        if not specified and not mpopt.opf.softlims.default:
            output[limit] = {"hl_mod": "none"}
            continue
        raw = supplied.get(limit, {})
        if not isinstance(raw, dict):
            raise TypeError(f"softlims_defaults: softlims.{limit} must be a mapping")
        spec = deepcopy(raw)
        if spec.get("hl_mod") == "none":
            output[limit] = spec
            continue

        external_matrix = np.asarray(mpc["order"]["ext"][matrix_name], dtype=float)
        default_idx = _default_indices(external_matrix, limit, column)
        if limit in {"VMAX", "VMIN"} and "idx" not in spec and np.asarray(spec.get("busnum", [])).size:
            bus_numbers = np.asarray(spec["busnum"], dtype=int).reshape(-1)
            spec["idx"] = np.flatnonzero(np.isin(external_matrix[:, BUS_I].astype(int), bus_numbers)) + 1
        indices = np.asarray(spec.get("idx", default_idx), dtype=int).reshape(-1)
        spec["idx"] = indices
        if indices.size == 0:
            spec["hl_mod"] = "none"
            output[limit] = spec
            continue

        spec["cost"] = _expanded_vector(
            spec.get("cost", cost_scale.get(limit, 1.0) * base_cost), indices.size, "cost", limit
        )
        mode = str(spec.get("hl_mod", "")).lower()
        original = external_matrix[indices - 1, column]
        if not mode:
            if limit == "PMIN":
                spec["hl_mod"] = "replace"
                replacement = np.zeros(indices.size)
                replacement[original < 0] = -np.inf
                spec["hl_val"] = replacement
            elif limit == "VMIN":
                spec["hl_mod"] = "replace"
                spec["hl_val"] = 0.0
            elif limit in {"QMIN", "ANGMIN"}:
                spec["hl_mod"] = "remove"
                spec["hl_val"] = -np.inf
            else:
                spec["hl_mod"] = "remove"
                spec["hl_val"] = np.inf
        elif mode == "scale" and "hl_val" not in spec:
            scale = np.full(indices.size, 2.0)
            scale[direction * original < 0] = 0.5
            spec["hl_val"] = scale
        elif mode == "shift" and "hl_val" not in spec:
            spec["hl_val"] = shift_default.get(limit, 10.0)
        elif mode == "replace" and "hl_val" not in spec:
            raise ValueError(f"softlims_defaults: softlims.{limit}.hl_val is required for replace")
        elif mode not in {"remove", "replace", "scale", "shift"}:
            raise ValueError(f"softlims_defaults: unknown hard limit modification {mode!r}")
        output[limit] = spec
    return output


def _softlims_init(mpc: dict[str, Any], mpopt: MatpowerConfig) -> dict[str, dict[str, Any]]:
    output = deepcopy(mpc["softlims"])
    for limit, (matrix_name, column, direction) in _limit_metadata(mpopt).items():
        spec = output[limit]
        if spec.get("hl_mod") == "none":
            continue
        matrix = np.asarray(mpc["order"]["ext"][matrix_name], dtype=float)
        entered_indices = np.asarray(spec["idx"], dtype=int).reshape(-1)
        valid = np.isin(entered_indices, _default_indices(matrix, limit, column))
        indices = entered_indices[valid]
        spec["idx"] = indices
        spec["cost"] = np.asarray(spec["cost"], dtype=float).reshape(-1)[valid]
        if indices.size == 0:
            output[limit] = spec
            continue
        saved = matrix[indices - 1, column].copy()
        spec["sav"] = saved
        mode = spec["hl_mod"]
        if mode == "remove":
            upper = np.full(indices.size, np.inf)
        else:
            values = _expanded_vector(spec["hl_val"], entered_indices.size, "hl_val", limit)[valid]
            spec["hl_val"] = values
            if mode == "scale":
                upper = direction * saved * (values - 1)
            elif mode == "shift":
                upper = values
            else:
                upper = direction * (values - saved)
        if np.any(upper < 0):
            raise ValueError(f"softlims_init: some hl_val for {limit} makes the hard limit more restrictive")
        spec["ub"] = upper
        if limit in {"ANGMAX", "ANGMIN", "RATE_A"}:
            spec["rval"] = 0.0
        elif limit in {"VMAX", "PMAX", "QMAX"}:
            spec["rval"] = np.inf
        else:
            spec["rval"] = -np.inf
        output[limit] = spec
    return output


def _external_to_internal_rows(order: dict[str, Any], matrix_name: str, internal_count: int) -> np.ndarray:
    external_count = np.asarray(order["ext"][matrix_name]).shape[0]
    mapping = np.zeros(external_count, dtype=int)
    online = np.asarray(order[matrix_name]["status"]["on"], dtype=int).reshape(-1) - 1
    if matrix_name == "gen":
        permutation = np.asarray(order["gen"]["i2e"], dtype=int).reshape(-1) - 1
        mapping[online[permutation]] = np.arange(1, internal_count + 1)
    else:
        mapping[online] = np.arange(1, internal_count + 1)
    return mapping


def userfcn_softlims_ext2int(mpc: dict[str, Any], mpopt: MatpowerConfig, args: object) -> dict[str, Any]:
    """Apply defaults, convert soft-limit indices, and remove original hard limits."""
    del args
    if mpopt.opf.v_cartesian:
        raise ValueError("userfcn_softlims_ext2int: soft limits do not support cartesian voltages")
    mpc["softlims"] = _softlims_defaults(mpc, mpopt)
    softlims = _softlims_init(mpc, mpopt)
    mpc["order"]["ext"]["softlims"] = deepcopy(softlims)

    for matrix_name in {item[0] for item in _limit_metadata(mpopt).values()}:
        mapping = _external_to_internal_rows(mpc["order"], matrix_name, np.asarray(mpc[matrix_name]).shape[0])
        for limit, (owner, _, _) in _limit_metadata(mpopt).items():
            if owner != matrix_name or softlims[limit].get("hl_mod") == "none":
                continue
            spec = softlims[limit]
            internal = mapping[np.asarray(spec["idx"], dtype=int) - 1]
            keep = internal != 0
            spec["idx"] = internal[keep]
            for field in ("cost", "sav", "ub"):
                spec[field] = np.asarray(spec[field]).reshape(-1)[keep]
            if "hl_val" in spec and np.asarray(spec["hl_val"]).size > 1:
                spec["hl_val"] = np.asarray(spec["hl_val"]).reshape(-1)[keep]

    for limit, (matrix_name, column, _) in _limit_metadata(mpopt).items():
        spec = softlims[limit]
        if spec.get("hl_mod") != "none" and np.asarray(spec["idx"]).size:
            mpc[matrix_name][np.asarray(spec["idx"], dtype=int) - 1, column] = spec["rval"]
    mpc["softlims"] = softlims
    mpc["order"].setdefault("int", {})["softlims"] = deepcopy(softlims)
    return mpc


def _selector(rows: np.ndarray, column_count: int) -> sparse.csc_matrix:
    return sparse.csc_matrix((np.ones(rows.size), (np.arange(rows.size), rows)), shape=(rows.size, column_count))


def _split_soft_flow_variables(x: object, bus_count: int, soft_count: int) -> list[np.ndarray]:
    vector = np.asarray(x, dtype=float).reshape(-1)
    return [vector[:bus_count], vector[bus_count : 2 * bus_count], vector[-soft_count:]]


def _soft_flow_fcn(
    x: object,
    mpc: dict[str, Any],
    yf: SparseMatrix,
    yt: SparseMatrix,
    indices: np.ndarray,
    mpopt: MatpowerConfig,
    maximum: np.ndarray,
) -> tuple[np.ndarray, sparse.csc_matrix]:
    blocks = _split_soft_flow_variables(x, np.asarray(mpc["bus"]).shape[0], indices.size)
    h, jacobian = opf_branch_flow_fcn_with_jacobian(blocks[:2], mpc, yf, yt, indices + 1, mpopt)
    violation = blocks[2]
    if mpopt.opf.flow_lim.upper().startswith("P"):
        adjusted_limit = maximum + violation
        slack_jacobian = -sparse.eye(indices.size, format="csc")
    else:
        adjusted_limit = (maximum + violation) ** 2
        slack_jacobian = sparse.diags(-2 * (maximum + violation), format="csc")
    values = np.asarray(h, dtype=float).reshape(-1) - np.r_[adjusted_limit, adjusted_limit]
    full_jacobian = sparse.hstack([jacobian, sparse.vstack([slack_jacobian, slack_jacobian])], format="csc")
    return values, sparse.csc_matrix(full_jacobian)


def _soft_flow_hess(
    x: object,
    multipliers: object,
    mpc: dict[str, Any],
    yf: SparseMatrix,
    yt: SparseMatrix,
    indices: np.ndarray,
    mpopt: MatpowerConfig,
) -> sparse.csc_matrix:
    blocks = _split_soft_flow_variables(x, np.asarray(mpc["bus"]).shape[0], indices.size)
    lam = np.asarray(multipliers, dtype=float).reshape(-1)
    voltage_hessian = opf_branch_flow_hess(blocks[:2], lam, mpc, yf, yt, indices + 1, mpopt)
    if mpopt.opf.flow_lim.upper().startswith("P"):
        slack_hessian = sparse.csc_matrix((indices.size, indices.size))
    else:
        slack_hessian = sparse.diags(-2 * (lam[: indices.size] + lam[indices.size :]), format="csc")
    return sparse.csc_matrix(sparse.block_diag((voltage_hessian, slack_hessian), format="csc"))


def userfcn_softlims_formulation(om: Any, mpopt: MatpowerConfig, args: object) -> Any:
    """Add violation variables, costs, and soft-limit constraints to an OPF model."""
    del args
    mpc = om.get_mpc()
    base_mva = float(mpc["baseMVA"])
    bus_count = np.asarray(mpc["bus"]).shape[0]
    generator_count = np.asarray(mpc["gen"]).shape[0]
    om.userdata["mpopt"] = mpopt

    for limit, spec in mpc["softlims"].items():
        if limit not in _limit_metadata(mpopt) or spec.get("hl_mod") == "none":
            continue
        rows = np.asarray(spec["idx"], dtype=int).reshape(-1) - 1
        if rows.size == 0:
            continue
        name = f"s_{limit.lower()}"
        upper = np.asarray(spec["ub"], dtype=float).reshape(-1).copy()
        cost = np.asarray(spec["cost"], dtype=float).reshape(-1).copy()
        if limit in {"RATE_A", "PMIN", "PMAX", "QMIN", "QMAX"}:
            upper /= base_mva
            cost *= base_mva
        elif limit in {"ANGMIN", "ANGMAX"}:
            upper *= np.pi / 180
            cost *= 180 / np.pi
        om.add_var(name, rows.size, np.zeros(rows.size), np.zeros(rows.size), upper)
        om.add_quad_cost(f"cs_{limit.lower()}", sparse.csc_matrix((0, 0)), cost, 0, [name])

        saved = np.asarray(spec["sav"], dtype=float).reshape(-1)
        identity = sparse.eye(rows.size, format="csc")
        if limit in {"ANGMIN", "ANGMAX"}:
            branch = np.asarray(mpc["branch"])
            angle = sparse.csc_matrix(
                (
                    np.r_[np.ones(rows.size), -np.ones(rows.size)],
                    (
                        np.r_[np.arange(rows.size), np.arange(rows.size)],
                        np.r_[branch[rows, F_BUS], branch[rows, T_BUS]].astype(int) - 1,
                    ),
                ),
                shape=(rows.size, bus_count),
            )
            matrix = sparse.hstack([angle, identity if limit == "ANGMIN" else -identity], format="csc")
            lower = saved * np.pi / 180 if limit == "ANGMIN" else np.full(rows.size, -np.inf)
            bound = np.full(rows.size, np.inf) if limit == "ANGMIN" else saved * np.pi / 180
            om.add_lin_constraint(f"soft_{limit.lower()}", matrix, lower, bound, ["Va", name])
        elif limit in {"PMIN", "PMAX"}:
            matrix = sparse.hstack(
                [_selector(rows, generator_count), identity if limit == "PMIN" else -identity], format="csc"
            )
            lower = saved / base_mva if limit == "PMIN" else np.full(rows.size, -np.inf)
            bound = np.full(rows.size, np.inf) if limit == "PMIN" else saved / base_mva
            om.add_lin_constraint(f"soft_{limit.lower()}", matrix, lower, bound, ["Pg", name])

    dc = _is_dc(mpopt)
    for limit in ("RATE_A",) if dc else ("VMIN", "VMAX", "QMIN", "QMAX", "RATE_A"):
        spec = mpc["softlims"][limit]
        if spec.get("hl_mod") == "none":
            continue
        rows = np.asarray(spec["idx"], dtype=int).reshape(-1) - 1
        if rows.size == 0:
            continue
        name = f"s_{limit.lower()}"
        saved = np.asarray(spec["sav"], dtype=float).reshape(-1)
        identity = sparse.eye(rows.size, format="csc")
        if limit == "RATE_A" and dc:
            bf = sparse.csc_matrix(om.get_userdata("Bf"))[rows]
            pfinj = np.asarray(om.get_userdata("Pfinj"), dtype=float).reshape(-1)[rows]
            om.add_lin_constraint("softPf", sparse.hstack([bf, -identity]), [], -pfinj + saved / base_mva, ["Va", name])
            om.add_lin_constraint("softPt", sparse.hstack([-bf, -identity]), [], pfinj + saved / base_mva, ["Va", name])
        elif limit == "RATE_A":
            _, yf, yt = makeYbus_full(base_mva, mpc["bus"], mpc["branch"])
            clean_mpc = {
                key: value for key, value in mpc.items() if not key.startswith("_opf_flow") and key != "_opf_branch_il"
            }
            soft_yf = yf[rows]
            soft_yt = yt[rows]
            maximum = saved / base_mva
            fcn = lambda x, cm=clean_mpc, syf=soft_yf, syt=soft_yt, r=rows, mx=maximum: _soft_flow_fcn(
                x, cm, syf, syt, r, mpopt, mx
            )
            hess = lambda x, lam, cm=clean_mpc, syf=soft_yf, syt=soft_yt, r=rows: _soft_flow_hess(
                x, lam, cm, syf, syt, r, mpopt
            )
            om.add_nln_constraint(
                ["softSf", "softSt"], np.array([rows.size, rows.size]), 0, fcn, hess, ["Va", "Vm", name]
            )
        else:
            column_count = bus_count if limit.startswith("V") else generator_count
            selector = _selector(rows, column_count)
            lower_limit = limit.endswith("MIN")
            matrix = sparse.hstack([selector, identity if lower_limit else -identity], format="csc")
            scale = 1.0 if limit.startswith("V") else base_mva
            lower = saved / scale if lower_limit else np.full(rows.size, -np.inf)
            bound = np.full(rows.size, np.inf) if lower_limit else saved / scale
            variable = "Vm" if limit.startswith("V") else "Qg"
            om.add_lin_constraint(f"soft_{limit.lower()}", matrix, lower, bound, [variable, name])
    return om


def _internal_to_external_rows(order: dict[str, Any], matrix_name: str, internal_rows: np.ndarray) -> np.ndarray:
    online = np.asarray(order[matrix_name]["status"]["on"], dtype=int).reshape(-1)
    if matrix_name == "gen":
        permutation = np.asarray(order["gen"]["i2e"], dtype=int).reshape(-1)
        return online[permutation[internal_rows] - 1] - 1
    return online[internal_rows] - 1


def userfcn_softlims_int2ext(results: dict[str, Any], mpopt: object, args: object) -> dict[str, Any]:
    """Restore hard limits and package overloads, costs, and shadow prices."""
    del args
    config = mpopt if isinstance(mpopt, MatpowerConfig) else results["om"].get_userdata("mpopt")
    if not isinstance(config, MatpowerConfig):
        raise TypeError("userfcn_softlims_int2ext: missing MATPOWER options in OPF model")
    results["om"].userdata.pop("mpopt", None)
    internal_softlims = results["softlims"]
    results["softlims"] = deepcopy(results["order"]["ext"]["softlims"])
    base_mva = float(results["baseMVA"])
    dc = _is_dc(config)

    for limit, (matrix_name, column, _) in _limit_metadata(config).items():
        spec = internal_softlims[limit]
        if spec.get("hl_mod") == "none":
            continue
        internal_rows = np.asarray(spec["idx"], dtype=int).reshape(-1) - 1
        if internal_rows.size:
            results[matrix_name][internal_rows, column] = np.asarray(spec["sav"], dtype=float)
        external_spec = results["softlims"][limit]
        for field in ("sav", "rval", "ub"):
            external_spec.pop(field, None)
        external_count = np.asarray(results["order"]["ext"][matrix_name]).shape[0]
        overload = np.zeros(external_count)
        overload_cost = np.zeros(external_count)
        if internal_rows.size and f"s_{limit.lower()}" in results["var"]["val"]:
            violation = np.asarray(results["var"]["val"][f"s_{limit.lower()}"], dtype=float).reshape(-1)
            if limit in {"RATE_A", "PMIN", "PMAX", "QMIN", "QMAX"}:
                violation *= base_mva
            elif limit in {"ANGMIN", "ANGMAX"}:
                violation *= 180 / np.pi
            violation[violation < 1e-8] = 0
            external_rows = _internal_to_external_rows(results["order"], matrix_name, internal_rows)
            overload[external_rows] = violation
            overload_cost[external_rows] = violation * np.asarray(spec["cost"], dtype=float).reshape(-1)
        external_spec["overload"] = overload
        external_spec["ovl_cost"] = overload_cost

        if not internal_rows.size:
            continue
        constraint = f"soft_{limit.lower()}"
        if limit in {"ANGMAX", "PMAX", "VMAX", "QMAX"}:
            mu = np.asarray(results["lin"]["mu"]["u"][constraint], dtype=float)
        elif limit in {"ANGMIN", "PMIN", "VMIN", "QMIN"}:
            mu = np.asarray(results["lin"]["mu"]["l"][constraint], dtype=float)
        else:
            mu = np.zeros(internal_rows.size)
        if limit == "ANGMAX":
            results["branch"][internal_rows, MU_ANGMAX] = mu * np.pi / 180
        elif limit == "ANGMIN":
            results["branch"][internal_rows, MU_ANGMIN] = mu * np.pi / 180
        elif limit == "PMAX":
            results["gen"][internal_rows, MU_PMAX] = mu / base_mva
        elif limit == "PMIN":
            results["gen"][internal_rows, MU_PMIN] = mu / base_mva
        elif limit == "VMAX" and not dc:
            results["bus"][internal_rows, MU_VMAX] = mu
        elif limit == "VMIN" and not dc:
            results["bus"][internal_rows, MU_VMIN] = mu
        elif limit == "QMAX" and not dc:
            results["gen"][internal_rows, MU_QMAX] = mu / base_mva
        elif limit == "QMIN" and not dc:
            results["gen"][internal_rows, MU_QMIN] = mu / base_mva
        elif limit == "RATE_A":
            if dc:
                mu_from = np.asarray(results["lin"]["mu"]["u"]["softPf"], dtype=float)
                mu_to = np.asarray(results["lin"]["mu"]["u"]["softPt"], dtype=float)
            else:
                mu_from = np.asarray(results["nli"]["mu"]["softSf"], dtype=float)
                mu_to = np.asarray(results["nli"]["mu"]["softSt"], dtype=float)
            results["branch"][internal_rows, MU_SF] = mu_from / base_mva
            results["branch"][internal_rows, MU_ST] = mu_to / base_mva
            if not dc and not config.opf.flow_lim.upper().startswith("P"):
                external_rows = _internal_to_external_rows(results["order"], matrix_name, internal_rows)
                conversion = 2 * (np.asarray(spec["sav"]) + overload[external_rows]) / base_mva
                results["branch"][internal_rows, MU_SF] *= conversion
                results["branch"][internal_rows, MU_ST] *= conversion
    return results


def userfcn_softlims_printpf(
    results: dict[str, Any], fd: object, mpopt: MatpowerConfig, args: object
) -> dict[str, Any]:
    """Retain the MATPOWER callback stage without duplicating print output."""
    del fd, mpopt, args
    return results


def userfcn_softlims_savecase(mpc: dict[str, Any], lines: object, prefix: object, args: object) -> dict[str, Any]:
    """Retain the callback stage; full legacy ``savecase`` output is out of scope."""
    del lines, prefix, args
    return mpc
