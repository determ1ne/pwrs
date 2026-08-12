# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Standard single-network OPF problem builder for Pyomo formulations."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..network import PowerNetwork, prepare_network
from .common import add_component_sets, add_generator_cost_objective, add_solver_suffixes
from .context import OptionValidator, PyomoPowerModel, ResultBuilder

type FormulationBuilder = Callable[[PyomoPowerModel, Any], None]


@dataclass(frozen=True)
class PyomoOpfSpec:
    """Formulation hooks and result layout required by the OPF builder."""

    name: str
    populate: FormulationBuilder
    result_builder: ResultBuilder
    option_validator: OptionValidator
    variable_order: tuple[str, ...]
    constraint_order: tuple[str, ...]


def build_opf_problem(mpc: Any, pyo: Any, spec: PyomoOpfSpec) -> PyomoPowerModel:
    """Build one standard Pyomo OPF from formulation-specific components."""
    network_start = time.perf_counter()
    network = mpc if isinstance(mpc, PowerNetwork) else prepare_network(mpc)
    network_build_time = time.perf_counter() - network_start

    model_start = time.perf_counter()
    problem = PyomoPowerModel(
        model=pyo.ConcreteModel(name=f"pwrs_POWER_MODELS_{spec.name}"),
        network=network,
        formulation=spec.name,
        result_builder=spec.result_builder,
        option_validator=spec.option_validator,
        network_build_time=network_build_time,
    )
    add_component_sets(problem, pyo)
    spec.populate(problem, pyo)
    add_generator_cost_objective(problem, pyo)
    add_solver_suffixes(problem, pyo)
    problem.set_result_order(spec.variable_order, spec.constraint_order)
    problem.model_build_time = time.perf_counter() - model_start
    return problem


__all__ = ["PyomoOpfSpec", "build_opf_problem"]
