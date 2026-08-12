# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from typing import Any

import numpy as np
from scipy import sparse

from .add_userfcn import add_userfcn
from .idx_brch import PF
from .makeBdc import makeBdc
from .remove_userfcn import remove_userfcn


def _status(mpc: dict[str, Any]) -> bool:
    return bool(mpc.get("userfcn", {}).get("status", {}).get("iflims", 0))


def toggle_iflims(mpc: dict[str, Any], on_off: str) -> dict[str, Any] | bool:
    """Enable, disable, or query MATPOWER interface-flow-limit callbacks."""
    action = on_off.upper()
    if action == "STATUS":
        return _status(mpc)
    if action not in {"ON", "OFF"}:
        raise ValueError("toggle_iflims: second argument must be 'on', 'off' or 'status'")

    if action == "ON":
        interface = mpc.get("if")
        if not isinstance(interface, dict) or "map" not in interface or "lims" not in interface:
            raise ValueError("toggle_iflims: case must contain an 'if' mapping defining 'map' and 'lims'")
        callbacks = (
            ("ext2int", userfcn_iflims_ext2int),
            ("formulation", userfcn_iflims_formulation),
            ("int2ext", userfcn_iflims_int2ext),
            ("printpf", userfcn_iflims_printpf),
            ("savecase", userfcn_iflims_savecase),
        )
        result = mpc
        for stage, callback in callbacks:
            result = add_userfcn(result, stage, callback)
        result.setdefault("userfcn", {}).setdefault("status", {})["iflims"] = 1
        return result

    result = deepcopy(mpc)
    callbacks = (
        ("savecase", userfcn_iflims_savecase),
        ("printpf", userfcn_iflims_printpf),
        ("int2ext", userfcn_iflims_int2ext),
        ("formulation", userfcn_iflims_formulation),
        ("ext2int", userfcn_iflims_ext2int),
    )
    if "userfcn" in result:
        for stage, callback in callbacks:
            if stage in result["userfcn"]:
                result = remove_userfcn(result, stage, callback)
    result.setdefault("userfcn", {}).setdefault("status", {})["iflims"] = 0
    return result


def userfcn_iflims_ext2int(mpc: dict[str, Any], mpopt: object, args: object) -> dict[str, Any]:
    """Convert signed external branch indices in ``if.map`` to internal indices."""
    del mpopt, args
    ifmap = np.asarray(mpc["if"]["map"], dtype=float).copy()
    order = mpc["order"]
    branch_count = np.asarray(order["ext"]["branch"]).shape[0]
    online = np.asarray(order["branch"]["status"]["on"], dtype=int).reshape(-1) - 1
    external_to_internal = np.zeros(branch_count, dtype=int)
    external_to_internal[online] = np.arange(1, online.size + 1)

    order["ext"]["ifmap"] = ifmap.copy()
    direction = np.sign(ifmap[:, 1]).astype(int)
    branches = np.abs(ifmap[:, 1]).astype(int) - 1
    if np.any((branches < 0) | (branches >= branch_count)):
        raise ValueError("userfcn_iflims_ext2int: if.map contains an invalid branch index")
    ifmap[:, 1] = direction * external_to_internal[branches]
    mpc["if"]["map"] = ifmap[ifmap[:, 1] != 0]
    return mpc


def userfcn_iflims_formulation(om: Any, mpopt: object, args: object) -> Any:
    """Add DC interface-flow constraints to an OPF model."""
    del mpopt, args
    mpc = om.get_mpc()
    base_mva = float(mpc["baseMVA"])
    _, bf, _, pfinj = makeBdc(base_mva, mpc["bus"], mpc["branch"])
    ifmap = np.asarray(mpc["if"]["map"], dtype=float)
    limits = np.asarray(mpc["if"]["lims"], dtype=float)
    interface_ids = np.unique(limits[:, 0].astype(int))
    bus_count = np.asarray(mpc["bus"]).shape[0]
    aif = sparse.lil_matrix((interface_ids.size, bus_count), dtype=float)
    lower = np.zeros(interface_ids.size)
    upper = np.zeros(interface_ids.size)
    pfinj_vector = np.asarray(pfinj, dtype=float).reshape(-1)

    for row, interface_id in enumerate(interface_ids):
        signed = ifmap[ifmap[:, 0].astype(int) == interface_id, 1].astype(int)
        if signed.size == 0:
            raise ValueError(f"userfcn_iflims_formulation: interface {interface_id} has no in-service branches")
        direction = np.sign(signed)
        branches = np.abs(signed) - 1
        aif[row, :] = direction @ bf[branches, :]
        offset = float(direction @ pfinj_vector[branches])
        limit_rows = limits[limits[:, 0].astype(int) == interface_id]
        if limit_rows.shape[0] != 1:
            raise ValueError(f"userfcn_iflims_formulation: interface {interface_id} must have exactly one limit row")
        lower[row] = limit_rows[0, 1] / base_mva - offset
        upper[row] = limit_rows[0, 2] / base_mva - offset

    om.add_lin_constraint("iflims", aif.tocsc(), lower, upper, ["Va"])
    return om


def userfcn_iflims_int2ext(results: dict[str, Any], mpopt: object, args: object) -> dict[str, Any]:
    """Package interface flows and shadow prices into OPF results."""
    del mpopt, args
    ifmap = np.asarray(results["if"]["map"], dtype=float)
    limits = np.asarray(results["if"]["lims"], dtype=float)
    interface_ids = np.unique(limits[:, 0].astype(int))
    flows = np.zeros(interface_ids.size)
    for row, interface_id in enumerate(interface_ids):
        signed = ifmap[ifmap[:, 0].astype(int) == interface_id, 1].astype(int)
        direction = np.sign(signed)
        branches = np.abs(signed) - 1
        flows[row] = float(direction @ results["branch"][branches, PF])

    results["if"]["P"] = flows
    results["if"]["mu"] = {
        "l": np.asarray(results["lin"]["mu"]["l"]["iflims"], dtype=float) / results["baseMVA"],
        "u": np.asarray(results["lin"]["mu"]["u"]["iflims"], dtype=float) / results["baseMVA"],
    }
    results["if"]["map"] = np.asarray(results["order"]["ext"]["ifmap"], dtype=float)
    return results


def userfcn_iflims_printpf(results: dict[str, Any], fd: object, mpopt: object, args: object) -> dict[str, Any]:
    """Retain the MATPOWER print callback stage without duplicating ``printpf`` output."""
    del fd, mpopt, args
    return results


def userfcn_iflims_savecase(mpc: dict[str, Any], lines: object, prefix: object, args: object) -> dict[str, Any]:
    """Retain the callback stage; full legacy ``savecase`` output is out of scope."""
    del lines, prefix, args
    return mpc
