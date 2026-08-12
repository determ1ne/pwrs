# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Current-voltage rectangular (IVR) Pyomo OPF formulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ....core.idx_brch import BR_R, BR_X
from ....core.idx_bus import BS, GS, PD, QD, VMAX, VMIN
from ....core.idx_gen import PMAX, PMIN, QMAX, QMIN
from ...network import PowerNetwork
from ...options import validate_ac_opf_options
from ...results import PowerModelSolution, build_matpower_result
from ..common import add_apparent_power_limits
from ..context import PyomoPowerModel
from ..opf import PyomoOpfSpec, build_opf_problem
from ..rectangular import (
    add_rectangular_angle_constraints,
    add_rectangular_branch_voltage_expressions,
    add_rectangular_reference_constraints,
    add_rectangular_voltage_magnitude_constraints,
    add_rectangular_voltage_variables,
)


def _validate_ivr_options(mpopt: Any) -> None:
    validate_ac_opf_options(mpopt, "IVR")


def _build_ivr_result(
    network: PowerNetwork,
    solution: PowerModelSolution,
    objective: float,
    success: int,
    info: dict[str, Any],
):
    values = solution.variables
    vr, vi = values["vr"], values["vi"]
    vm_squared = vr**2 + vi**2
    vm = np.sqrt(vm_squared)
    crf, cif = values["crf"], values["cif"]
    crt, cit = values["crt"], values["cit"]
    crg, cig = values["crg"], values["cig"]
    variables = dict(values)
    variables.update(
        {
            "va": np.arctan2(vi, vr),
            "vm": vm,
            "pg": vr[network.gen_bus] * crg + vi[network.gen_bus] * cig,
            "qg": vi[network.gen_bus] * crg - vr[network.gen_bus] * cig,
            "pf": vr[network.f_bus] * crf + vi[network.f_bus] * cif,
            "qf": vi[network.f_bus] * crf - vr[network.f_bus] * cif,
            "pt": vr[network.t_bus] * crt + vi[network.t_bus] * cit,
            "qt": vi[network.t_bus] * crt - vr[network.t_bus] * cit,
        }
    )
    if len(network.dcline):
        crdcf, cidcf = values["crdcf"], values["cidcf"]
        crdct, cidct = values["crdct"], values["cidct"]
        variables.update(
            {
                "pdcf": vr[network.dc_f_bus] * crdcf + vi[network.dc_f_bus] * cidcf,
                "qdcf": vi[network.dc_f_bus] * crdcf - vr[network.dc_f_bus] * cidcf,
                "pdct": vr[network.dc_t_bus] * crdct + vi[network.dc_t_bus] * cidct,
                "qdct": vi[network.dc_t_bus] * crdct - vr[network.dc_t_bus] * cidct,
            }
        )

    multipliers = dict(solution.constraint_multipliers)
    current_real = multipliers["current_real_balance"]
    current_imaginary = multipliers["current_imaginary_balance"]
    multipliers["active_balance"] = (vr * current_real + vi * current_imaginary) / vm_squared
    multipliers["reactive_balance"] = (vi * current_real - vr * current_imaginary) / vm_squared
    branch_real = vr[network.f_bus] * vr[network.t_bus] + vi[network.f_bus] * vi[network.t_bus]
    multipliers["angle_upper"] = multipliers["angle_upper"] * branch_real / np.cos(network.branch_angle_max) ** 2
    multipliers["angle_lower"] = -multipliers["angle_lower"] * branch_real / np.cos(network.branch_angle_min) ** 2

    lower = dict(solution.lower_bound_multipliers)
    upper = dict(solution.upper_bound_multipliers)
    lower["vm"] = -2 * vm * multipliers["voltage_lower"]
    upper["vm"] = (
        2 * vm * multipliers["voltage_upper"]
        + solution.lower_bound_multipliers["vr"]
        + solution.upper_bound_multipliers["vr"]
        + solution.lower_bound_multipliers["vi"]
        + solution.upper_bound_multipliers["vi"]
    )
    lower["pg"] = -multipliers["gen_active_lower"]
    upper["pg"] = multipliers["gen_active_upper"]
    lower["qg"] = -multipliers["gen_reactive_lower"]
    upper["qg"] = multipliers["gen_reactive_upper"]
    if len(network.dcline):
        ndc = len(network.dcline)

        def constraint_bounds(name: str, finite: np.ndarray, sign: float = 1.0) -> np.ndarray:
            values = np.zeros(ndc)
            indices = np.flatnonzero(np.isfinite(finite))
            values[indices] = sign * multipliers.get(name, np.zeros(len(indices)))
            return values

        lower["pdcf"] = constraint_bounds("dcline_active_from_lower", network.dc_pmin_from, -1.0)
        upper["pdcf"] = constraint_bounds("dcline_active_from_upper", network.dc_pmax_from)
        lower["pdct"] = constraint_bounds("dcline_active_to_lower", network.dc_pmin_to, -1.0)
        upper["pdct"] = constraint_bounds("dcline_active_to_upper", network.dc_pmax_to)
        lower["qdcf"] = constraint_bounds("dcline_reactive_from_lower", network.dc_qmin_from, -1.0)
        upper["qdcf"] = constraint_bounds("dcline_reactive_from_upper", network.dc_qmax_from)
        lower["qdct"] = constraint_bounds("dcline_reactive_to_lower", network.dc_qmin_to, -1.0)
        upper["qdct"] = constraint_bounds("dcline_reactive_to_upper", network.dc_qmax_to)
    mapped = PowerModelSolution(
        vector=solution.vector,
        variables=variables,
        constraint_multipliers=multipliers,
        lower_bound_multipliers=lower,
        upper_bound_multipliers=upper,
    )
    return build_matpower_result(
        network,
        mapped,
        objective,
        success,
        info,
        formulation="IVR",
        implementation="PYOMO",
        angle_limit="branch",
    )


def _add_current_variables(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network

    def terminal_bounds(_: Any, i: int) -> tuple[float | None, float | None]:
        rating = float(network.rate[i])
        if not np.isfinite(rating):
            return None, None
        f_bus, t_bus = int(network.f_bus[i]), int(network.t_bus[i])
        upper = max(
            rating * network.tap[i] / network.bus[f_bus, VMIN],
            rating / network.bus[t_bus, VMIN],
        )
        return -float(upper), float(upper)

    def series_bounds(_: Any, i: int) -> tuple[float | None, float | None]:
        rating = float(network.rate[i])
        if not np.isfinite(rating):
            return None, None
        f_bus, t_bus = int(network.f_bus[i]), int(network.t_bus[i])
        shunt_current = max(
            abs(network.b_fr[i]) * network.bus[f_bus, VMAX] ** 2,
            abs(network.b_to[i]) * network.bus[t_bus, VMAX] ** 2,
        )
        series_current = max(
            rating * network.tap[i] / network.bus[f_bus, VMIN],
            rating * network.tap[i] / network.bus[t_bus, VMIN],
        )
        upper = series_current + shunt_current
        return -float(upper), float(upper)

    def generator_bounds(_: Any, i: int) -> tuple[float, float]:
        bus = int(network.gen_bus[i])
        active = max(abs(network.gen[i, PMAX]), abs(network.gen[i, PMIN])) / network.base_mva
        reactive = max(abs(network.gen[i, QMAX]), abs(network.gen[i, QMIN])) / network.base_mva
        upper = np.hypot(active, reactive) / network.bus[bus, VMIN]
        return -float(upper), float(upper)

    def dcline_bounds(_: Any, i: int) -> tuple[float | None, float | None]:
        s_from = np.hypot(
            max(abs(network.dc_pmax_from[i]), abs(network.dc_pmin_from[i])),
            max(abs(network.dc_qmax_from[i]), abs(network.dc_qmin_from[i])),
        )
        s_to = np.hypot(
            max(abs(network.dc_pmax_to[i]), abs(network.dc_pmin_to[i])),
            max(abs(network.dc_qmax_to[i]), abs(network.dc_qmin_to[i])),
        )
        vmin = min(
            network.bus[int(network.dc_f_bus[i]), VMIN],
            network.bus[int(network.dc_t_bus[i]), VMIN],
        )
        upper = max(s_from, s_to) / vmin
        if not np.isfinite(upper):
            return None, None
        return -float(upper), float(upper)

    for name in ("crf", "cif", "crt", "cit"):
        setattr(model, name, pyo.Var(model.BRANCH, bounds=terminal_bounds, initialize=0.0))
        component = getattr(model, name)
        problem.register_variables(name, tuple(component[i] for i in model.BRANCH))
    for name in ("csr", "csi"):
        setattr(model, name, pyo.Var(model.BRANCH, bounds=series_bounds, initialize=0.0))
        component = getattr(model, name)
        problem.register_variables(name, tuple(component[i] for i in model.BRANCH))
    for name in ("crg", "cig"):
        setattr(model, name, pyo.Var(model.GEN, bounds=generator_bounds, initialize=0.0))
        component = getattr(model, name)
        problem.register_variables(name, tuple(component[i] for i in model.GEN))
    if len(network.dcline):
        for name in ("crdcf", "cidcf", "crdct", "cidct"):
            setattr(model, name, pyo.Var(model.DCLINE, bounds=dcline_bounds, initialize=0.0))
            component = getattr(model, name)
            problem.register_variables(name, tuple(component[i] for i in model.DCLINE))


def _add_power_expressions(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network
    model.pg = pyo.Expression(
        model.GEN,
        rule=lambda m, i: m.vr[int(network.gen_bus[i])] * m.crg[i] + m.vi[int(network.gen_bus[i])] * m.cig[i],
    )
    model.qg = pyo.Expression(
        model.GEN,
        rule=lambda m, i: m.vi[int(network.gen_bus[i])] * m.crg[i] - m.vr[int(network.gen_bus[i])] * m.cig[i],
    )
    model.pf = pyo.Expression(
        model.BRANCH,
        rule=lambda m, i: m.vr[int(network.f_bus[i])] * m.crf[i] + m.vi[int(network.f_bus[i])] * m.cif[i],
    )
    model.qf = pyo.Expression(
        model.BRANCH,
        rule=lambda m, i: m.vi[int(network.f_bus[i])] * m.crf[i] - m.vr[int(network.f_bus[i])] * m.cif[i],
    )
    model.pt = pyo.Expression(
        model.BRANCH,
        rule=lambda m, i: m.vr[int(network.t_bus[i])] * m.crt[i] + m.vi[int(network.t_bus[i])] * m.cit[i],
    )
    model.qt = pyo.Expression(
        model.BRANCH,
        rule=lambda m, i: m.vi[int(network.t_bus[i])] * m.crt[i] - m.vr[int(network.t_bus[i])] * m.cit[i],
    )
    for name in ("pg", "qg", "pf", "qf", "pt", "qt"):
        problem.register_expression(name, getattr(model, name))
    if len(network.dcline):
        model.pdcf = pyo.Expression(
            model.DCLINE,
            rule=lambda m, i: m.vr[int(network.dc_f_bus[i])] * m.crdcf[i] + m.vi[int(network.dc_f_bus[i])] * m.cidcf[i],
        )
        model.qdcf = pyo.Expression(
            model.DCLINE,
            rule=lambda m, i: m.vi[int(network.dc_f_bus[i])] * m.crdcf[i] - m.vr[int(network.dc_f_bus[i])] * m.cidcf[i],
        )
        model.pdct = pyo.Expression(
            model.DCLINE,
            rule=lambda m, i: m.vr[int(network.dc_t_bus[i])] * m.crdct[i] + m.vi[int(network.dc_t_bus[i])] * m.cidct[i],
        )
        model.qdct = pyo.Expression(
            model.DCLINE,
            rule=lambda m, i: m.vi[int(network.dc_t_bus[i])] * m.crdct[i] - m.vr[int(network.dc_t_bus[i])] * m.cidct[i],
        )
        for name in ("pdcf", "qdcf", "pdct", "qdct"):
            problem.register_expression(name, getattr(model, name))


def _add_dcline_power_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network
    if not len(network.dcline):
        return
    bounds = (
        ("dcline_active_from_lower", model.pdcf, network.dc_pmin_from, "lower"),
        ("dcline_active_from_upper", model.pdcf, network.dc_pmax_from, "upper"),
        ("dcline_active_to_lower", model.pdct, network.dc_pmin_to, "lower"),
        ("dcline_active_to_upper", model.pdct, network.dc_pmax_to, "upper"),
        ("dcline_reactive_from_lower", model.qdcf, network.dc_qmin_from, "lower"),
        ("dcline_reactive_from_upper", model.qdcf, network.dc_qmax_from, "upper"),
        ("dcline_reactive_to_lower", model.qdct, network.dc_qmin_to, "lower"),
        ("dcline_reactive_to_upper", model.qdct, network.dc_qmax_to, "upper"),
    )
    for name, expression, values, sense in bounds:
        indices = tuple(int(i) for i in np.flatnonzero(np.isfinite(values)))
        if not indices:
            continue
        index_name = name.upper()
        setattr(model, index_name, pyo.Set(initialize=indices, ordered=True))
        index = getattr(model, index_name)
        if sense == "lower":
            rule = lambda _, i, expression=expression, values=values: expression[i] >= float(values[i])
        else:
            rule = lambda _, i, expression=expression, values=values: expression[i] <= float(values[i])
        setattr(model, name, pyo.Constraint(index, rule=rule))
        component = getattr(model, name)
        problem.register_constraints(name, tuple(component[i] for i in indices))
    model.dcline_loss = pyo.Constraint(
        model.DCLINE,
        rule=lambda m, i: (
            (1.0 - float(network.dc_loss1[i])) * m.pdcf[i] + m.pdct[i] - float(network.dc_loss0[i]) == 0.0
        ),
    )
    problem.register_constraints("dcline_loss", tuple(model.dcline_loss[i] for i in model.DCLINE))


def _add_generator_power_bounds(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network
    model.gen_active_lower = pyo.Constraint(
        model.GEN,
        rule=lambda m, i: m.pg[i] >= float(network.gen[i, PMIN] / network.base_mva),
    )
    model.gen_active_upper = pyo.Constraint(
        model.GEN,
        rule=lambda m, i: m.pg[i] <= float(network.gen[i, PMAX] / network.base_mva),
    )
    model.gen_reactive_lower = pyo.Constraint(
        model.GEN,
        rule=lambda m, i: m.qg[i] >= float(network.gen[i, QMIN] / network.base_mva),
    )
    model.gen_reactive_upper = pyo.Constraint(
        model.GEN,
        rule=lambda m, i: m.qg[i] <= float(network.gen[i, QMAX] / network.base_mva),
    )
    for name in ("gen_active_lower", "gen_active_upper", "gen_reactive_lower", "gen_reactive_upper"):
        component = getattr(model, name)
        problem.register_constraints(name, tuple(component[i] for i in model.GEN))


def _add_current_balance_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network

    def real_balance(m: Any, i: int) -> Any:
        voltage_squared = m.voltage_magnitude_squared[i]
        return (
            pyo.quicksum(m.crf[j] for j in network.from_branches_at_bus[i])
            + pyo.quicksum(m.crt[j] for j in network.to_branches_at_bus[i])
            + pyo.quicksum(m.crdcf[j] for j in network.from_dclines_at_bus[i])
            + pyo.quicksum(m.crdct[j] for j in network.to_dclines_at_bus[i])
            - pyo.quicksum(m.crg[j] for j in network.generators_at_bus[i])
            + (network.bus[i, PD] * m.vr[i] + network.bus[i, QD] * m.vi[i]) / network.base_mva / voltage_squared
            + network.bus[i, GS] / network.base_mva * m.vr[i]
            - network.bus[i, BS] / network.base_mva * m.vi[i]
            == 0.0
        )

    def imaginary_balance(m: Any, i: int) -> Any:
        voltage_squared = m.voltage_magnitude_squared[i]
        return (
            pyo.quicksum(m.cif[j] for j in network.from_branches_at_bus[i])
            + pyo.quicksum(m.cit[j] for j in network.to_branches_at_bus[i])
            + pyo.quicksum(m.cidcf[j] for j in network.from_dclines_at_bus[i])
            + pyo.quicksum(m.cidct[j] for j in network.to_dclines_at_bus[i])
            - pyo.quicksum(m.cig[j] for j in network.generators_at_bus[i])
            + (network.bus[i, PD] * m.vi[i] - network.bus[i, QD] * m.vr[i]) / network.base_mva / voltage_squared
            + network.bus[i, GS] / network.base_mva * m.vi[i]
            + network.bus[i, BS] / network.base_mva * m.vr[i]
            == 0.0
        )

    model.current_real_balance = pyo.Constraint(model.BUS, rule=real_balance)
    model.current_imaginary_balance = pyo.Constraint(model.BUS, rule=imaginary_balance)
    problem.register_constraints("current_real_balance", tuple(model.current_real_balance[i] for i in model.BUS))
    problem.register_constraints(
        "current_imaginary_balance",
        tuple(model.current_imaginary_balance[i] for i in model.BUS),
    )


def _add_branch_current_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network
    resistance = network.branch[:, BR_R]
    reactance = network.branch[:, BR_X]
    tr = network.tap * np.cos(network.shift)
    ti = network.tap * np.sin(network.shift)
    tap_squared = network.tap**2

    model.current_from_real = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.crf[i]
            == (tr[i] * m.csr[i] - ti[i] * m.csi[i] - network.b_fr[i] * m.vi[int(network.f_bus[i])]) / tap_squared[i]
        ),
    )
    model.current_from_imaginary = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.cif[i]
            == (tr[i] * m.csi[i] + ti[i] * m.csr[i] + network.b_fr[i] * m.vr[int(network.f_bus[i])]) / tap_squared[i]
        ),
    )
    model.current_to_real = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: m.crt[i] == -m.csr[i] - network.b_to[i] * m.vi[int(network.t_bus[i])],
    )
    model.current_to_imaginary = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: m.cit[i] == -m.csi[i] + network.b_to[i] * m.vr[int(network.t_bus[i])],
    )
    model.voltage_drop_real = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.vr[int(network.t_bus[i])]
            == (m.vr[int(network.f_bus[i])] * tr[i] + m.vi[int(network.f_bus[i])] * ti[i]) / tap_squared[i]
            - resistance[i] * m.csr[i]
            + reactance[i] * m.csi[i]
        ),
    )
    model.voltage_drop_imaginary = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.vi[int(network.t_bus[i])]
            == (m.vi[int(network.f_bus[i])] * tr[i] - m.vr[int(network.f_bus[i])] * ti[i]) / tap_squared[i]
            - resistance[i] * m.csi[i]
            - reactance[i] * m.csr[i]
        ),
    )
    for name in (
        "current_from_real",
        "current_from_imaginary",
        "current_to_real",
        "current_to_imaginary",
        "voltage_drop_real",
        "voltage_drop_imaginary",
    ):
        component = getattr(model, name)
        problem.register_constraints(name, tuple(component[i] for i in model.BRANCH))


def _populate_ivr(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add IVR variables and network constraints to an OPF problem."""
    add_rectangular_voltage_variables(problem, pyo)
    add_rectangular_voltage_magnitude_constraints(problem, pyo)
    _add_current_variables(problem, pyo)
    _add_power_expressions(problem, pyo)
    _add_dcline_power_constraints(problem, pyo)
    _add_generator_power_bounds(problem, pyo)
    add_rectangular_reference_constraints(problem, pyo)
    _add_current_balance_constraints(problem, pyo)
    _add_branch_current_constraints(problem, pyo)
    add_rectangular_branch_voltage_expressions(problem, pyo)
    add_rectangular_angle_constraints(problem, pyo, per_branch=True)
    add_apparent_power_limits(problem, pyo)


_IVR_OPF = PyomoOpfSpec(
    name="IVR",
    populate=_populate_ivr,
    result_builder=_build_ivr_result,
    option_validator=_validate_ivr_options,
    variable_order=("vr", "vi", "crf", "cif", "crt", "cit", "csr", "csi", "crg", "cig"),
    constraint_order=(
        "voltage_lower",
        "voltage_upper",
        "gen_active_lower",
        "gen_active_upper",
        "gen_reactive_lower",
        "gen_reactive_upper",
        "reference_angle",
        "current_real_balance",
        "current_imaginary_balance",
        "current_from_real",
        "current_from_imaginary",
        "current_to_real",
        "current_to_imaginary",
        "voltage_drop_real",
        "voltage_drop_imaginary",
        "angle_upper",
        "angle_lower",
        "thermal_from",
        "thermal_to",
    ),
)


def build_ivr_power_model(mpc: Any, pyo: Any) -> PyomoPowerModel:
    """Build IVR as a standard single-network OPF."""
    return build_opf_problem(mpc, pyo, _IVR_OPF)


__all__ = ["build_ivr_power_model"]
