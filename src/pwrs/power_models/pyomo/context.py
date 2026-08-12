# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Shared state and semantic component registry for Pyomo formulations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..network import PowerNetwork
from ..results import PowerModelSolution

type ResultBuilder = Callable[
    [PowerNetwork, PowerModelSolution, float, int, dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]
]
type OptionValidator = Callable[[Any], None]


@dataclass
class PyomoPowerModel:
    """A built Pyomo formulation plus the metadata required to solve it."""

    model: Any
    network: PowerNetwork
    formulation: str
    result_builder: ResultBuilder
    option_validator: OptionValidator
    network_build_time: float = 0.0
    model_build_time: float = 0.0
    variables: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    constraints: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    constraint_senses: dict[str, tuple[str, ...]] = field(default_factory=dict)
    expressions: dict[str, Any] = field(default_factory=dict)
    objective_terms: dict[str, Any] = field(default_factory=dict)
    _variable_result_order: tuple[str, ...] = ()
    _constraint_result_order: tuple[str, ...] = ()
    base_variable_groups: tuple[str, ...] = ()
    base_constraint_groups: tuple[str, ...] = ()
    extension_records: tuple[Any, ...] = ()
    extensions_applied: bool = False
    extension_signature: tuple[tuple[str, int], ...] = ()
    base_problem_class: str | None = None

    def register_variables(self, name: str, values: tuple[Any, ...]) -> None:
        self._register(self.variables, name, values)

    def register_constraints(self, name: str, values: tuple[Any, ...]) -> None:
        self._register(self.constraints, name, values)

    def register_expression(self, name: str, value: Any) -> None:
        if name in self.expressions:
            raise KeyError(f"duplicate Pyomo expression group: {name}")
        self.expressions[name] = value

    @property
    def backend(self) -> str:
        return "pyomo"

    @property
    def available_quantities(self) -> frozenset[str]:
        return frozenset((*self.variables, *self.expressions))

    def has_quantity(self, name: str) -> bool:
        return name in self.variables or name in self.expressions

    def add_constraint(
        self,
        name: str,
        constraint: Any,
        *,
        senses: str | tuple[str, ...] | None = None,
    ) -> Any:
        """Attach and register a named custom Pyomo constraint component."""
        if name in self.constraints or self.model.component(name) is not None:
            raise KeyError(f"duplicate Pyomo constraint group: {name}")
        self.model.add_component(name, constraint)
        values = tuple(constraint.values())
        if not values:
            self.model.del_component(constraint)
            raise ValueError(f"custom Pyomo constraint group {name!r} is empty")
        inferred = tuple(
            "equality"
            if value.equality
            else "bounded"
            if value.lower is not None and value.upper is not None
            else "lower"
            if value.lower is not None
            else "upper"
            if value.upper is not None
            else "native"
            for value in values
        )
        try:
            normalized_senses = self._normalize_senses(senses, len(values), inferred)
        except ValueError:
            self.model.del_component(constraint)
            raise
        self.register_constraints(name, values)
        self.constraint_senses[name] = normalized_senses
        return constraint

    def add_variable(self, name: str, variable: Any) -> Any:
        """Attach and register an auxiliary Pyomo variable component."""
        if name in self.variables or self.model.component(name) is not None:
            raise KeyError(f"duplicate Pyomo variable group: {name}")
        self.model.add_component(name, variable)
        values = tuple(variable.values())
        if not values:
            self.model.del_component(variable)
            raise ValueError(f"custom Pyomo variable group {name!r} is empty")
        self.register_variables(name, values)
        return variable

    def add_objective_term(self, name: str, expression: Any) -> Any:
        """Attach a named expression and add it to the minimization objective."""
        import pyomo.environ as pyo

        if name in self.objective_terms or self.model.component(name) is not None:
            raise KeyError(f"duplicate Pyomo objective term: {name}")
        objectives = tuple(self.model.component_data_objects(pyo.Objective, active=True))
        if len(objectives) != 1:
            raise RuntimeError("custom objective terms require exactly one active objective")
        component = pyo.Expression(expr=expression)
        self.model.add_component(name, component)
        objectives[0].set_value(objectives[0].expr + component)
        self.objective_terms[name] = component
        return component

    def add_bounded_constraint(
        self,
        name: str,
        expression: Any,
        *,
        lower: float | None = None,
        upper: float | None = None,
    ) -> Any:
        """Create and register one semantic group of scalar bounds."""
        return self.add_bounded_constraints(name, ((expression, lower, upper),))

    def add_bounded_constraints(
        self,
        name: str,
        bounds: tuple[tuple[Any, float | None, float | None], ...],
    ) -> Any:
        """Create and register a group of scalar lower/upper bounds."""
        from pyomo.environ import Constraint

        expressions = []
        senses = []
        for expression, lower, upper in bounds:
            if lower is None and upper is None:
                raise ValueError("a bounded constraint requires a lower or upper bound")
            if lower is not None:
                expressions.append(expression >= lower)
                senses.append("lower")
            if upper is not None:
                expressions.append(expression <= upper)
                senses.append("upper")
        if not expressions:
            raise ValueError("a bounded constraint group cannot be empty")
        component = Constraint(range(len(expressions)), rule=lambda _, index: expressions[index])
        result = self.add_constraint(name, component)
        self.constraint_senses[name] = tuple(senses)
        return result

    def add_equality_constraints(
        self,
        name: str,
        equalities: tuple[tuple[Any, float], ...],
    ) -> Any:
        """Create and register a group of scalar equalities."""
        if not equalities:
            raise ValueError("an equality constraint group cannot be empty")
        from pyomo.environ import Constraint

        component = Constraint(
            range(len(equalities)),
            rule=lambda _, index: equalities[index][0] == equalities[index][1],
        )
        result = self.add_constraint(name, component)
        self.constraint_senses[name] = ("equality",) * len(equalities)
        return result

    def quantity(self, name: str, index: int | None = None) -> Any:
        """Return a semantic variable/expression group or one indexed value."""
        values = self.variables.get(name, self.expressions.get(name))
        if values is None:
            raise KeyError(f"{self.formulation} does not define semantic quantity {name!r}")
        return values if index is None else values[index]

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
        """Return native active constraint names missing from the registry."""
        import pyomo.environ as pyo

        registered = {id(value) for values in self.constraints.values() for value in values}
        return tuple(
            value.name
            for value in self.model.component_data_objects(pyo.Constraint, active=True)
            if id(value) not in registered
        )

    def set_result_order(self, variables: tuple[str, ...], constraints: tuple[str, ...]) -> None:
        self._require_groups(self.variables, variables, "variable")
        self._require_groups(self.constraints, constraints, "constraint")
        self._variable_result_order = (*variables, *(name for name in self.variables if name not in variables))
        self._constraint_result_order = (
            *constraints,
            *(name for name in self.constraints if name not in constraints),
        )
        self.base_variable_groups = tuple(self.variables)
        self.base_constraint_groups = tuple(self.constraints)

    @property
    def variable_order(self) -> tuple[Any, ...]:
        return tuple(value for name in self._variable_result_order for value in self.variables[name])

    @property
    def constraint_order(self) -> tuple[Any, ...]:
        return tuple(value for name in self._constraint_result_order for value in self.constraints[name])

    @property
    def variable_result_groups(self) -> tuple[str, ...]:
        return self._variable_result_order

    @property
    def constraint_result_groups(self) -> tuple[str, ...]:
        return self._constraint_result_order

    @staticmethod
    def _register(registry: dict[str, tuple[Any, ...]], name: str, values: tuple[Any, ...]) -> None:
        if name in registry:
            raise KeyError(f"duplicate Pyomo component group: {name}")
        registry[name] = values

    @staticmethod
    def _require_groups(registry: dict[str, tuple[Any, ...]], names: tuple[str, ...], kind: str) -> None:
        missing = [name for name in names if name not in registry]
        if missing:
            raise KeyError(f"unregistered Pyomo {kind} groups: {', '.join(missing)}")


__all__ = ["OptionValidator", "PyomoPowerModel", "ResultBuilder"]
