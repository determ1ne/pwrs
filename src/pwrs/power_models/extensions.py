# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Lifecycle and metadata for user extensions to built power models."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from .network import PowerNetwork


@runtime_checkable
class PowerModel(Protocol):
    """Backend-neutral interface available to extension callbacks."""

    network: PowerNetwork
    formulation: str
    variables: Mapping[str, Any]
    constraints: Mapping[str, tuple[Any, ...]]
    expressions: Mapping[str, Any]
    objective_terms: Mapping[str, Any]

    @property
    def backend(self) -> str: ...

    @property
    def available_quantities(self) -> frozenset[str]: ...

    def has_quantity(self, name: str) -> bool: ...

    def quantity(self, name: str, index: int | None = None) -> Any: ...

    def add_constraint(
        self,
        name: str,
        constraint: Any,
        *,
        senses: str | tuple[str, ...] | None = None,
    ) -> Any: ...

    def add_bounded_constraint(
        self,
        name: str,
        expression: Any,
        *,
        lower: float | None = None,
        upper: float | None = None,
    ) -> Any: ...

    def add_bounded_constraints(
        self,
        name: str,
        bounds: tuple[tuple[Any, float | None, float | None], ...],
    ) -> Any: ...

    def add_equality_constraints(
        self,
        name: str,
        equalities: tuple[tuple[Any, float], ...],
    ) -> Any: ...

    def add_variable(self, name: str, variable: Any) -> Any: ...

    def add_objective_term(self, name: str, expression: Any) -> Any: ...


type PowerModelExtensionCallback = Callable[[PowerModel], None]


@dataclass(frozen=True)
class PowerModelExtension:
    """A stable name paired with a callback that mutates a built model."""

    name: str
    callback: PowerModelExtensionCallback

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError("PowerModelExtension.name must be a non-empty Python identifier")
        if not callable(self.callback):
            raise TypeError("PowerModelExtension.callback must be callable")
        if self.name == "__built_model__":
            raise ValueError("PowerModelExtension.name '__built_model__' is reserved")


@dataclass(frozen=True)
class ExtensionRecord:
    """The constraints and construction time contributed by one extension."""

    name: str
    constraint_groups: tuple[str, ...]
    variable_groups: tuple[str, ...]
    objective_terms: tuple[str, ...]
    build_time: float


def _normalize_extensions(value: Any) -> tuple[PowerModelExtension, ...]:
    if value is None or value == () or value == []:
        return ()
    items = value if isinstance(value, (tuple, list)) else (value,)
    extensions = []
    for item in items:
        if isinstance(item, PowerModelExtension):
            extension = item
        elif callable(item):
            name = getattr(item, "__name__", "")
            if not name or name == "<lambda>":
                raise ValueError("anonymous extensions require PowerModelExtension(name, callback)")
            extension = PowerModelExtension(name, cast(PowerModelExtensionCallback, item))
        else:
            raise TypeError("PowerModels extensions must be callables or PowerModelExtension values")
        extensions.append(extension)
    names = [extension.name for extension in extensions]
    if len(names) != len(set(names)):
        raise ValueError("PowerModels extension names must be unique")
    return tuple(extensions)


def apply_extensions(problem: Any, value: Any) -> tuple[ExtensionRecord, ...]:
    """Apply configured extensions once, after build and before solver selection."""
    extensions = _normalize_extensions(value)
    signature = tuple((extension.name, id(extension.callback)) for extension in extensions)
    if problem.extensions_applied:
        if signature != problem.extension_signature:
            raise RuntimeError("a built PowerModels model cannot be solved with a different extension set")
        return problem.extension_records

    records = []
    for extension in extensions:
        constraints_before = tuple(problem.constraints)
        variables_before = tuple(problem.variables)
        objective_terms_before = tuple(problem.objective_terms)
        start = time.perf_counter()
        try:
            returned = extension.callback(problem)
        except Exception as exc:
            raise RuntimeError(f"PowerModels extension {extension.name!r} failed") from exc
        elapsed = time.perf_counter() - start
        if returned is not None:
            raise TypeError(f"PowerModels extension {extension.name!r} must return None")
        missing = problem.unregistered_active_constraints()
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                f"PowerModels extension {extension.name!r} added unregistered active constraints: {joined}; "
                "use problem.add_constraint()"
            )
        added = tuple(name for name in problem.constraints if name not in constraints_before)
        variables_added = tuple(name for name in problem.variables if name not in variables_before)
        objective_terms_added = tuple(
            name for name in problem.objective_terms if name not in objective_terms_before
        )
        inactive = tuple(
            getattr(constraint, "name", name)
            for name in added
            for constraint in problem.constraints[name]
            if getattr(constraint, "active", True) is False
        )
        if inactive:
            joined = ", ".join(inactive)
            raise RuntimeError(f"PowerModels extension {extension.name!r} registered inactive constraints: {joined}")
        records.append(
            ExtensionRecord(extension.name, added, variables_added, objective_terms_added, elapsed)
        )

    owned_constraints = {
        name for record in records for name in record.constraint_groups
    }
    owned_variables = {name for record in records for name in record.variable_groups}
    owned_objective_terms = {name for record in records for name in record.objective_terms}
    manual_constraints = tuple(
        name
        for name in problem.constraints
        if name not in problem.base_constraint_groups and name not in owned_constraints
    )
    manual_variables = tuple(
        name
        for name in problem.variables
        if name not in problem.base_variable_groups and name not in owned_variables
    )
    manual_objective_terms = tuple(
        name for name in problem.objective_terms if name not in owned_objective_terms
    )
    if manual_constraints or manual_variables or manual_objective_terms:
        records.append(
            ExtensionRecord(
                "__built_model__",
                manual_constraints,
                manual_variables,
                manual_objective_terms,
                0.0,
            )
        )

    problem.extension_records = tuple(records)
    problem.extension_signature = signature
    problem.extensions_applied = True
    return problem.extension_records


def extension_summary(problem: Any) -> list[dict[str, Any]]:
    """Return JSON-compatible construction metadata for applied extensions."""
    return [
        {
            "name": record.name,
            "constraint_groups": list(record.constraint_groups),
            "variable_groups": list(record.variable_groups),
            "objective_terms": list(record.objective_terms),
            "build_time": record.build_time,
        }
        for record in problem.extension_records
    ]


__all__ = [
    "ExtensionRecord",
    "PowerModelExtension",
    "PowerModelExtensionCallback",
    "PowerModel",
    "apply_extensions",
    "extension_summary",
]
