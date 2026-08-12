# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""MATPOWER result mapping shared by PowerModels formulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.idx_brch import MU_ANGMAX, MU_ANGMIN, MU_SF, MU_ST, PF, PT, QF, QT
from ..core.idx_bus import LAM_P, LAM_Q, MU_VMAX, MU_VMIN, VA, VM
from ..core.idx_dcline import MU_PMAX as DC_MU_PMAX
from ..core.idx_dcline import MU_PMIN as DC_MU_PMIN
from ..core.idx_dcline import MU_QMAXF as DC_MU_QMAXF
from ..core.idx_dcline import MU_QMAXT as DC_MU_QMAXT
from ..core.idx_dcline import MU_QMINF as DC_MU_QMINF
from ..core.idx_dcline import MU_QMINT as DC_MU_QMINT
from ..core.idx_dcline import PF as DC_PF
from ..core.idx_dcline import PT as DC_PT
from ..core.idx_dcline import QF as DC_QF
from ..core.idx_dcline import QT as DC_QT
from ..core.idx_dcline import VF as DC_VF
from ..core.idx_dcline import VT as DC_VT
from ..core.idx_gen import MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PG, QG, VG
from .network import PowerNetwork, pad_matrix


@dataclass(frozen=True)
class PowerModelSolution:
    """A solver result indexed by formulation-level semantic component names."""

    vector: np.ndarray
    variables: dict[str, np.ndarray]
    constraint_multipliers: dict[str, np.ndarray]
    lower_bound_multipliers: dict[str, np.ndarray]
    upper_bound_multipliers: dict[str, np.ndarray]


def _required(mapping: dict[str, np.ndarray], name: str) -> np.ndarray:
    try:
        return np.asarray(mapping[name], dtype=float)
    except KeyError as exc:
        raise KeyError(f"result mapping requires the semantic component group {name!r}") from exc


def _optional(mapping: dict[str, np.ndarray], name: str, size: int) -> np.ndarray:
    return np.asarray(mapping.get(name, np.zeros(size)), dtype=float)


def build_matpower_result(
    network: PowerNetwork,
    solution: PowerModelSolution,
    objective: float,
    success: int,
    info: dict[str, Any],
    *,
    formulation: str,
    implementation: str,
    branch_limit: str = "apparent_power",
    angle_limit: str = "bus_pair",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map semantic OPF values and multipliers into MATPOWER result matrices."""
    bus = pad_matrix(network.bus_external, MU_VMIN)
    gen = pad_matrix(network.gen_external, MU_QMIN)
    branch = pad_matrix(network.branch_external, MU_ANGMAX)
    values = solution.variables
    va = _required(values, "va")
    vm = _required(values, "vm")
    pg = _required(values, "pg")
    qg = _required(values, "qg")
    pf = _required(values, "pf").copy()
    qf = _required(values, "qf").copy()
    pt = _required(values, "pt").copy()
    qt = _required(values, "qt").copy()

    bus[network.bus_rows, VA - 1] = np.rad2deg(va)
    bus[network.bus_rows, VM - 1] = vm
    gen[network.gen_rows, PG - 1] = pg * network.base_mva
    gen[network.gen_rows, QG - 1] = qg * network.base_mva
    gen[network.gen_rows, VG - 1] = vm[network.gen_bus]
    reversed_branches = network.branch_reversed
    pf[reversed_branches], pt[reversed_branches] = pt[reversed_branches], pf[reversed_branches]
    qf[reversed_branches], qt[reversed_branches] = qt[reversed_branches], qf[reversed_branches]
    branch[network.branch_rows, PF - 1] = pf * network.base_mva
    branch[network.branch_rows, QF - 1] = qf * network.base_mva
    branch[network.branch_rows, PT - 1] = pt * network.base_mva
    branch[network.branch_rows, QT - 1] = qt * network.base_mva

    dcline = None
    if len(network.dcline_external):
        dcline = pad_matrix(network.dcline_external, DC_MU_QMAXT)
    if len(network.dcline):
        assert dcline is not None
        ndc = len(network.dcline)
        pdcf = _required(values, "pdcf")
        pdct = _required(values, "pdct")
        qdcf = _optional(values, "qdcf", ndc)
        qdct = _optional(values, "qdct", ndc)
        rows = network.dcline_rows
        dcline[rows, DC_PF - 1] = pdcf * network.base_mva
        dcline[rows, DC_PT - 1] = -pdct * network.base_mva
        dcline[rows, DC_QF - 1] = -qdcf * network.base_mva
        dcline[rows, DC_QT - 1] = -qdct * network.base_mva
        dcline[rows, DC_VF - 1] = vm[network.dc_f_bus]
        dcline[rows, DC_VT - 1] = vm[network.dc_t_bus]

    nb, ng, nl = len(network.bus), len(network.gen), len(network.branch)
    multipliers = solution.constraint_multipliers
    lower = solution.lower_bound_multipliers
    upper = solution.upper_bound_multipliers
    bus[network.bus_rows, LAM_P - 1] = _optional(multipliers, "active_balance", nb) / network.base_mva
    bus[network.bus_rows, LAM_Q - 1] = _optional(multipliers, "reactive_balance", nb) / network.base_mva

    finite_rate = np.flatnonzero(np.isfinite(network.rate))
    mu_sf = np.zeros(nl)
    mu_st = np.zeros(nl)
    if branch_limit == "apparent_power":
        mu_sf[finite_rate] = (
            2 * _optional(multipliers, "thermal_from", len(finite_rate)) * network.rate[finite_rate] / network.base_mva
        )
        mu_st[finite_rate] = (
            2 * _optional(multipliers, "thermal_to", len(finite_rate)) * network.rate[finite_rate] / network.base_mva
        )
    elif branch_limit == "conic_apparent_power":
        mu_sf[finite_rate] = _optional(multipliers, "thermal_from", len(finite_rate)) / network.base_mva
        mu_st[finite_rate] = _optional(multipliers, "thermal_to", len(finite_rate)) / network.base_mva
    elif branch_limit == "active_power":
        # PowerModels reports the lossless from/to limits through the lower and
        # upper bound references, respectively.
        mu_sf[finite_rate] = _optional(lower, "p", nl)[finite_rate] / network.base_mva
        mu_st[finite_rate] = _optional(upper, "p", nl)[finite_rate] / network.base_mva
    elif branch_limit == "directed_active_power":
        pf_lower = _optional(lower, "pf", nl)
        pf_upper = _optional(upper, "pf", nl)
        pt_lower = _optional(lower, "pt", nl)
        pt_upper = _optional(upper, "pt", nl)
        mu_sf[finite_rate] = np.maximum(pf_lower, pf_upper)[finite_rate] / network.base_mva
        mu_st[finite_rate] = np.maximum(pt_lower, pt_upper)[finite_rate] / network.base_mva
    else:
        raise ValueError(f"unsupported branch-limit result mapping: {branch_limit}")
    mu_sf[reversed_branches], mu_st[reversed_branches] = mu_st[reversed_branches], mu_sf[reversed_branches]
    branch[network.branch_rows, MU_SF - 1] = mu_sf
    branch[network.branch_rows, MU_ST - 1] = mu_st

    if angle_limit == "bus_pair":
        angle_upper = _optional(multipliers, "angle_upper", len(network.angle_pairs))
        angle_lower = _optional(multipliers, "angle_lower", len(network.angle_pairs))
        for pair_i, (f_bus, t_bus) in enumerate(network.angle_pairs):
            matches = np.flatnonzero((network.f_bus == f_bus) & (network.t_bus == t_bus))
            if not matches.size:
                continue
            branch_i = matches[0]
            external = network.branch_rows[branch_i]
            if network.branch_reversed[branch_i]:
                branch[external, MU_ANGMIN - 1] = angle_upper[pair_i] * np.pi / 180
                branch[external, MU_ANGMAX - 1] = -angle_lower[pair_i] * np.pi / 180
            else:
                branch[external, MU_ANGMAX - 1] = angle_upper[pair_i] * np.pi / 180
                branch[external, MU_ANGMIN - 1] = -angle_lower[pair_i] * np.pi / 180
    elif angle_limit == "branch":
        angle_upper = _optional(multipliers, "angle_upper", nl)
        angle_lower = _optional(multipliers, "angle_lower", nl)
        for branch_i, external in enumerate(network.branch_rows):
            if network.branch_reversed[branch_i]:
                branch[external, MU_ANGMIN - 1] = angle_upper[branch_i] * np.pi / 180
                branch[external, MU_ANGMAX - 1] = -angle_lower[branch_i] * np.pi / 180
            else:
                branch[external, MU_ANGMAX - 1] = angle_upper[branch_i] * np.pi / 180
                branch[external, MU_ANGMIN - 1] = -angle_lower[branch_i] * np.pi / 180
    else:
        raise ValueError(f"unsupported angle-limit result mapping: {angle_limit}")

    bus[network.bus_rows, MU_VMIN - 1] = _optional(lower, "vm", nb)
    bus[network.bus_rows, MU_VMAX - 1] = _optional(upper, "vm", nb)
    gen[network.gen_rows, MU_PMIN - 1] = _optional(lower, "pg", ng) / network.base_mva
    gen[network.gen_rows, MU_PMAX - 1] = _optional(upper, "pg", ng) / network.base_mva
    gen[network.gen_rows, MU_QMIN - 1] = _optional(lower, "qg", ng) / network.base_mva
    gen[network.gen_rows, MU_QMAX - 1] = _optional(upper, "qg", ng) / network.base_mva

    if len(network.dcline):
        assert dcline is not None
        ndc = len(network.dcline)
        rows = network.dcline_rows
        pdcf_lower = _optional(lower, "pdcf", ndc)
        pdcf_upper = _optional(upper, "pdcf", ndc)
        pdct_lower = _optional(lower, "pdct", ndc)
        pdct_upper = _optional(upper, "pdct", ndc)
        dcline[rows, DC_MU_PMIN - 1] = (pdcf_lower + pdct_upper) / network.base_mva
        dcline[rows, DC_MU_PMAX - 1] = (pdcf_upper + pdct_lower) / network.base_mva
        # MATPOWER QF/QT are terminal injections, the negative of the
        # PowerModels outgoing-flow convention, so their bound duals swap.
        dcline[rows, DC_MU_QMINF - 1] = _optional(upper, "qdcf", ndc) / network.base_mva
        dcline[rows, DC_MU_QMAXF - 1] = _optional(lower, "qdcf", ndc) / network.base_mva
        dcline[rows, DC_MU_QMINT - 1] = _optional(upper, "qdct", ndc) / network.base_mva
        dcline[rows, DC_MU_QMAXT - 1] = _optional(lower, "qdct", ndc) / network.base_mva

    result = network.source.copy()
    result.update(
        {
            "bus": bus,
            "gen": gen,
            "branch": branch,
            "f": objective,
            "x": solution.vector,
            "success": success,
        }
    )
    if dcline is not None:
        result["dcline"] = dcline
    status_msg = info.get("status_msg", "")
    if isinstance(status_msg, (bytes, bytearray)):
        status_msg = status_msg.decode(errors="replace")
    algorithm = f"POWER_MODELS/{formulation}/{implementation}/{info.get('solver_name', 'IPOPT')}"
    raw = {
        "xr": solution.vector,
        "pimul": np.array([]),
        "info": int(info.get("status", -1)),
        "output": {
            "alg": algorithm,
            "backend": "POWER_MODELS",
            "formulation": formulation,
            "implementation": implementation,
            "status": int(info.get("status", -1)),
            "status_msg": status_msg,
            "iterations": info.get("iter_count", info.get("iter")),
        },
    }
    return result, raw


__all__ = ["PowerModelSolution", "build_matpower_result"]
