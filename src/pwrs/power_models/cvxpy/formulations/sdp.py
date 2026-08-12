# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Dense and chordal sparse semidefinite W-space OPF relaxations."""

from __future__ import annotations

from typing import Any

from ...network import PowerNetwork
from ...options import validate_conic_ac_power_opf_options
from ..common import (
    add_branch_power_equations,
    add_power_balance_constraints,
    add_power_variables,
    add_thermal_cones,
    add_w_angle_constraints,
    register_w_bounds,
)
from ..context import CvxpyPowerModel
from ..opf import CvxpyOpfSpec, build_opf_problem
from ..results import build_w_result


def _validate_sdpwrm_options(mpopt: Any) -> None:
    validate_conic_ac_power_opf_options(mpopt, "SDPWRM")


def _validate_sparse_sdpwrm_options(mpopt: Any) -> None:
    validate_conic_ac_power_opf_options(mpopt, "SPARSESDPWRM")


def _build_sdpwrm_result(network, solution, objective, success, info):
    return build_w_result(network, solution, objective, success, info, "SDPWRM")


def _build_sparse_sdpwrm_result(network, solution, objective, success, info):
    return build_w_result(network, solution, objective, success, info, "SPARSESDPWRM")


def _add_common_w_constraints(problem: CvxpyPowerModel, cp: Any) -> None:
    add_power_balance_constraints(problem, cp)
    add_branch_power_equations(problem, cp)
    add_w_angle_constraints(problem, cp)
    add_thermal_cones(problem, cp)


def _populate_sdpwrm(problem: CvxpyPowerModel, cp: Any) -> None:
    """Add dense SDPWRM variables and network constraints."""
    network = problem.network
    nb = len(network.bus)
    wr_matrix = cp.Variable((nb, nb), symmetric=True, name="WR")
    wi_matrix = cp.Variable((nb, nb), name="WI")
    problem.register_variable("WR", wr_matrix)
    problem.register_variable("WI", wi_matrix)
    add_power_variables(problem, cp)

    w = cp.diag(wr_matrix)
    wr = cp.hstack([wr_matrix[int(f), int(t)] for f, t in network.angle_pairs])
    wi = cp.hstack([wi_matrix[int(f), int(t)] for f, t in network.angle_pairs])
    problem.register_expression("w", w)
    problem.register_expression("wr", wr)
    problem.register_expression("wi", wi)
    register_w_bounds(problem, w, wr, wi)

    problem.register_constraints("imaginary_skew_symmetry", wi_matrix + wi_matrix.T == 0)
    block = cp.bmat([[wr_matrix, wi_matrix], [-wi_matrix, wr_matrix]])
    problem.register_constraints("voltage_psd", block >> 0)
    _add_common_w_constraints(problem, cp)
    problem.metadata.update({"psd_clique_count": 1, "psd_max_clique_size": nb})


_SDPWRM_OPF = CvxpyOpfSpec(
    name="SDPWRM",
    cone_kind="SDP",
    populate=_populate_sdpwrm,
    result_builder=_build_sdpwrm_result,
    option_validator=_validate_sdpwrm_options,
    variable_order=("WR", "WI", "pg", "qg", "pf", "qf", "pt", "qt"),
)


def build_sdpwrm_power_model(mpc: Any) -> CvxpyPowerModel:
    """Build the dense real-block representation of PowerModels' SDPWRM."""
    return build_opf_problem(mpc, _SDPWRM_OPF)


def _maximal_chordal_cliques(network: PowerNetwork) -> tuple[tuple[int, ...], ...]:
    """Compute a deterministic minimum-degree chordal extension and its cliques."""
    nb = len(network.bus)
    adjacency = [set() for _ in range(nb)]
    for f_bus, t_bus in network.angle_pairs:
        f, t = int(f_bus), int(t_bus)
        adjacency[f].add(t)
        adjacency[t].add(f)
    remaining = set(range(nb))
    candidates: list[frozenset[int]] = []
    while remaining:
        bus = min(remaining, key=lambda i: (len(adjacency[i] & remaining), i))
        neighbors = sorted(adjacency[bus] & remaining)
        candidates.append(frozenset((bus, *neighbors)))
        for position, first in enumerate(neighbors):
            for second in neighbors[position + 1 :]:
                adjacency[first].add(second)
                adjacency[second].add(first)
        remaining.remove(bus)
    maximal = [clique for clique in candidates if not any(clique < other for other in candidates)]
    return tuple(sorted((tuple(sorted(clique)) for clique in maximal), key=lambda value: (value[0], len(value), value)))


def _clique_edges(cliques: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()
    for clique in cliques:
        for position, first in enumerate(clique):
            for second in clique[position + 1 :]:
                edges.add((first, second))
    return tuple(sorted(edges))


def _sparse_voltage_products(
    problem: CvxpyPowerModel,
    cp: Any,
    cliques: tuple[tuple[int, ...], ...],
) -> tuple[Any, Any, Any]:
    network = problem.network
    edges = _clique_edges(cliques)
    edge_index = {edge: index for index, edge in enumerate(edges)}
    w = cp.Variable(len(network.bus), name="w")
    wr_edge = cp.Variable(len(edges), name="chordal_wr")
    wi_edge = cp.Variable(len(edges), name="chordal_wi")
    problem.register_variable("w", w)
    problem.register_variable("chordal_wr", wr_edge)
    problem.register_variable("chordal_wi", wi_edge)

    wr_values = []
    wi_values = []
    for f_bus, t_bus in network.angle_pairs:
        f, t = int(f_bus), int(t_bus)
        edge = (min(f, t), max(f, t))
        wr_values.append(wr_edge[edge_index[edge]])
        sign = 1 if f < t else -1
        wi_values.append(sign * wi_edge[edge_index[edge]])
    wr = cp.hstack(wr_values)
    wi = cp.hstack(wi_values)
    problem.register_expression("w", w)
    problem.register_expression("wr", wr)
    problem.register_expression("wi", wi)
    register_w_bounds(problem, w, wr, wi)

    cone_constraints = []
    zero = cp.Constant(0.0)
    for clique in cliques:
        if len(clique) == 1:
            continue
        if len(clique) == 2:
            first, second = clique
            edge_i = edge_index[(first, second)]
            cone_constraints.append(
                cp.SOC(
                    w[first] + w[second],
                    cp.hstack(
                        [
                            w[first] - w[second],
                            2 * wr_edge[edge_i],
                            2 * wi_edge[edge_i],
                        ]
                    ),
                )
            )
            continue
        real_rows = []
        imaginary_rows = []
        for first in clique:
            real_row = []
            imaginary_row = []
            for second in clique:
                if first == second:
                    real_row.append(w[first])
                    imaginary_row.append(zero)
                else:
                    edge = (min(first, second), max(first, second))
                    edge_i = edge_index[edge]
                    real_row.append(wr_edge[edge_i])
                    imaginary_row.append(wi_edge[edge_i] if first < second else -wi_edge[edge_i])
            real_rows.append(real_row)
            imaginary_rows.append(imaginary_row)
        real_matrix = cp.bmat(real_rows)
        imaginary_matrix = cp.bmat(imaginary_rows)
        block = cp.bmat([[real_matrix, imaginary_matrix], [-imaginary_matrix, real_matrix]])
        cone_constraints.append(block >> 0)
    problem.register_constraints("voltage_psd", tuple(cone_constraints))
    problem.metadata.update(
        {
            "psd_cliques": cliques,
            "psd_clique_count": len(cliques),
            "psd_max_clique_size": max(map(len, cliques), default=0),
            "chordal_edge_count": len(edges),
        }
    )
    return w, wr, wi


def _populate_sparse_sdpwrm(problem: CvxpyPowerModel, cp: Any) -> None:
    """Add sparse SDPWRM variables and network constraints."""
    network = problem.network
    add_power_variables(problem, cp)
    cliques = _maximal_chordal_cliques(network)
    _sparse_voltage_products(problem, cp, cliques)
    _add_common_w_constraints(problem, cp)


_SPARSE_SDPWRM_OPF = CvxpyOpfSpec(
    name="SPARSESDPWRM",
    cone_kind="SDP",
    populate=_populate_sparse_sdpwrm,
    result_builder=_build_sparse_sdpwrm_result,
    option_validator=_validate_sparse_sdpwrm_options,
    variable_order=("w", "chordal_wr", "chordal_wi", "pg", "qg", "pf", "qf", "pt", "qt"),
)


def build_sparse_sdpwrm_power_model(mpc: Any) -> CvxpyPowerModel:
    """Build a chordal sparse SDPWRM with shared overlap variables."""
    return build_opf_problem(mpc, _SPARSE_SDPWRM_OPF)


__all__ = ["build_sdpwrm_power_model", "build_sparse_sdpwrm_power_model"]
