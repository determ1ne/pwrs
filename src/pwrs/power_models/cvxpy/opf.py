# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Standard single-network OPF problem builder for CVXPY formulations."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..network import PowerNetwork, prepare_network
from .common import generator_cost_expression, import_cvxpy
from .context import CvxpyPowerModel, OptionValidator, ResultBuilder

type FormulationBuilder = Callable[[CvxpyPowerModel, Any], None]


@dataclass(frozen=True)
class CvxpyOpfSpec:
    """Formulation hooks and result layout required by the OPF builder."""

    name: str
    cone_kind: str
    populate: FormulationBuilder
    result_builder: ResultBuilder
    option_validator: OptionValidator
    variable_order: tuple[str, ...]


def build_opf_problem(mpc: Any, spec: CvxpyOpfSpec) -> CvxpyPowerModel:
    """Build one standard CVXPY OPF from formulation-specific components."""
    cp = import_cvxpy()
    network_start = time.perf_counter()
    network = mpc if isinstance(mpc, PowerNetwork) else prepare_network(mpc)
    network_build_time = time.perf_counter() - network_start

    model_start = time.perf_counter()
    problem = CvxpyPowerModel(
        network=network,
        formulation=spec.name,
        cone_kind=spec.cone_kind,
        objective=None,
        result_builder=spec.result_builder,
        option_validator=spec.option_validator,
        network_build_time=network_build_time,
    )
    spec.populate(problem, cp)
    problem.objective = generator_cost_expression(problem, cp)
    problem.set_result_order(spec.variable_order)
    problem.model_build_time = time.perf_counter() - model_start
    return problem


__all__ = ["CvxpyOpfSpec", "build_opf_problem"]
