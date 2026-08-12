# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Shared state and semantic registries for CVXPY power models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..network import PowerNetwork
from ..results import PowerModelSolution

type ResultBuilder = Callable[
    [PowerNetwork, PowerModelSolution, float, int, dict[str, Any]],
    tuple[dict[str, Any], dict[str, Any]],
]
type OptionValidator = Callable[[Any], None]


@dataclass(frozen=True)
class BoundConstraint:
    """A CVXPY bound constraint and its positions in a semantic variable."""

    constraint: Any
    size: int
    indices: np.ndarray


@dataclass
class CvxpyPowerModel:
    """A built CVXPY formulation plus solver and result metadata."""

    network: PowerNetwork
    formulation: str
    cone_kind: str
    objective: Any
    result_builder: ResultBuilder
    option_validator: OptionValidator
    network_build_time: float = 0.0
    model_build_time: float = 0.0
    variables: dict[str, Any] = field(default_factory=dict)
    expressions: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    constraint_senses: dict[str, tuple[str, ...]] = field(default_factory=dict)
    objective_terms: dict[str, Any] = field(default_factory=dict)
    lower_bounds: dict[str, BoundConstraint] = field(default_factory=dict)
    upper_bounds: dict[str, BoundConstraint] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    _variable_result_order: tuple[str, ...] = ()
    base_variable_groups: tuple[str, ...] = ()
    base_constraint_groups: tuple[str, ...] = ()
    extension_records: tuple[Any, ...] = ()
    extensions_applied: bool = False
    extension_signature: tuple[tuple[str, int], ...] = ()
    base_problem_class: str | None = None

    def register_variable(self, name: str, value: Any) -> None:
        if name in self.variables:
            raise KeyError(f"duplicate CVXPY variable group: {name}")
        self.variables[name] = value

    def register_expression(self, name: str, value: Any) -> None:
        if name in self.expressions:
            raise KeyError(f"duplicate CVXPY expression group: {name}")
        self.expressions[name] = value

    @property
    def backend(self) -> str:
        return "cvxpy"

    @property
    def available_quantities(self) -> frozenset[str]:
        return frozenset((*self.variables, *self.expressions))

    def has_quantity(self, name: str) -> bool:
        return name in self.variables or name in self.expressions

    def register_constraints(self, name: str, values: Any | tuple[Any, ...]) -> None:
        if name in self.constraints:
            raise KeyError(f"duplicate CVXPY constraint group: {name}")
        self.constraints[name] = values if isinstance(values, tuple) else (values,)

    def register_bound(
        self,
        name: str,
        constraint: Any,
        size: int,
        indices: np.ndarray,
        *,
        upper: bool,
    ) -> None:
        registry = self.upper_bounds if upper else self.lower_bounds
        if name in registry:
            raise KeyError(f"duplicate CVXPY bound group: {name}")
        registry[name] = BoundConstraint(constraint, size, np.asarray(indices, dtype=int))

    def set_result_order(self, variables: tuple[str, ...]) -> None:
        missing = [name for name in variables if name not in self.variables]
        if missing:
            raise KeyError(f"unregistered CVXPY variable groups: {', '.join(missing)}")
        self._variable_result_order = (*variables, *(name for name in self.variables if name not in variables))
        self.base_variable_groups = tuple(self.variables)
        self.base_constraint_groups = tuple(self.constraints)

    def add_constraint(
        self,
        name: str,
        constraint: Any | tuple[Any, ...],
        *,
        senses: str | tuple[str, ...] | None = None,
    ) -> Any | tuple[Any, ...]:
        """Register a named custom constraint before solving."""
        if name in self.constraints:
            raise KeyError(f"duplicate CVXPY constraint group: {name}")
        values = constraint if isinstance(constraint, tuple) else (constraint,)
        inferred = tuple(
            "equality" if value.__class__.__name__ in {"Equality", "Zero"} else "native"
            for value in values
        )
        normalized_senses = self._normalize_senses(senses, len(values), inferred)
        self.register_constraints(name, values)
        self.constraint_senses[name] = normalized_senses
        return constraint

    def add_variable(self, name: str, variable: Any) -> Any:
        """Register an auxiliary CVXPY variable or variable expression."""
        self.register_variable(name, variable)
        return variable

    def add_objective_term(self, name: str, expression: Any) -> Any:
        """Add a named expression to the minimization objective."""
        if name in self.objective_terms:
            raise KeyError(f"duplicate CVXPY objective term: {name}")
        if tuple(expression.shape) != ():
            raise ValueError("a CVXPY objective term must be scalar")
        self.objective_terms[name] = expression
        self.objective = self.objective + expression
        return expression

    def add_bounded_constraint(
        self,
        name: str,
        expression: Any,
        *,
        lower: float | None = None,
        upper: float | None = None,
    ) -> tuple[Any, ...]:
        """Create and register one semantic group of scalar bounds."""
        return self.add_bounded_constraints(name, ((expression, lower, upper),))

    def add_bounded_constraints(
        self,
        name: str,
        bounds: tuple[tuple[Any, float | None, float | None], ...],
    ) -> tuple[Any, ...]:
        """Create and register a group of scalar lower/upper bounds."""
        constraints = []
        senses = []
        for expression, lower, upper in bounds:
            if lower is None and upper is None:
                raise ValueError("a bounded constraint requires a lower or upper bound")
            if lower is not None:
                constraints.append(expression >= lower)
                senses.append("lower")
            if upper is not None:
                constraints.append(expression <= upper)
                senses.append("upper")
        if not constraints:
            raise ValueError("a bounded constraint group cannot be empty")
        constraints = tuple(constraints)
        self.add_constraint(name, constraints)
        self.constraint_senses[name] = tuple(senses)
        return constraints

    def add_equality_constraints(
        self,
        name: str,
        equalities: tuple[tuple[Any, float], ...],
    ) -> tuple[Any, ...]:
        """Create and register a group of scalar equalities."""
        if not equalities:
            raise ValueError("an equality constraint group cannot be empty")
        constraints = tuple(expression == value for expression, value in equalities)
        self.add_constraint(name, constraints)
        self.constraint_senses[name] = ("equality",) * len(equalities)
        return constraints

    def quantity(self, name: str, index: int | None = None) -> Any:
        """Return a semantic variable/expression group or one indexed value."""
        value = self.variables.get(name, self.expressions.get(name))
        if value is None:
            raise KeyError(f"{self.formulation} does not define semantic quantity {name!r}")
        return value if index is None else value[index]

    def variable(self, name: str, index: int | None = None) -> Any:
        """Compatibility alias for :meth:`quantity`."""
        return self.quantity(name, index)

    @staticmethod
    def _normalize_senses(
        senses: str | tuple[str, ...] | None,
        count: int,
        inferred: tuple[str, ...],
    ) -> tuple[str, ...]:
        if senses is None:
            return inferred
        values = (senses,) * count if isinstance(senses, str) else tuple(senses)
        allowed = {"lower", "upper", "equality", "bounded", "native"}
        if len(values) != count or any(value not in allowed for value in values):
            raise ValueError(
                "constraint senses must match the constraint count and use "
                "lower/upper/equality/bounded/native"
            )
        return values

    def unregistered_active_constraints(self) -> tuple[str, ...]:
        """CVXPY constraints can only enter the solve through the registry."""
        return ()

    @property
    def variable_result_groups(self) -> tuple[str, ...]:
        return self._variable_result_order

    @property
    def all_constraints(self) -> list[Any]:
        return [
            *(constraint for values in self.constraints.values() for constraint in values),
            *(bound.constraint for bound in self.lower_bounds.values()),
            *(bound.constraint for bound in self.upper_bounds.values()),
        ]


__all__ = ["BoundConstraint", "CvxpyPowerModel", "OptionValidator", "ResultBuilder"]
