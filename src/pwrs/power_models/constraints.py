# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Backend-neutral domain constraints for PowerModels formulations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, overload

from .extensions import PowerModel
from .voltage import branch_pair_indices


def _validate_name(name: str) -> None:
    if not name or not name.isidentifier():
        raise ValueError("constraint name must be a non-empty Python identifier")


def _selection(value: int | Iterable[int] | None) -> Iterable[int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return (int(value),)
    return value


@overload
def _finite_values(value: None, count: int, label: str) -> tuple[None, ...]: ...


@overload
def _finite_values(value: float | Iterable[float], count: int, label: str) -> tuple[float, ...]: ...


def _finite_values(value: float | Iterable[float] | None, count: int, label: str) -> tuple[float | None, ...]:
    if value is None:
        return (None,) * count
    if isinstance(value, (int, float)):
        values = (float(value),) * count
    else:
        values = tuple(float(item) for item in value)
    if len(values) != count:
        raise ValueError(f"{label} must be scalar or contain {count} values")
    if any(not math.isfinite(item) for item in values):
        raise ValueError(f"{label} values must be finite")
    return values


def _bounded_values(
    lower: float | Iterable[float] | None,
    upper: float | Iterable[float] | None,
    count: int,
    lower_label: str,
    upper_label: str,
) -> tuple[tuple[float | None, ...], tuple[float | None, ...]]:
    if lower is None and upper is None:
        raise ValueError(f"a constraint requires {lower_label} or {upper_label}")
    lowers = _finite_values(lower, count, lower_label)
    uppers = _finite_values(upper, count, upper_label)
    if any(lo is not None and hi is not None and lo > hi for lo, hi in zip(lowers, uppers)):
        raise ValueError(f"{lower_label} cannot exceed {upper_label}")
    return lowers, uppers


def _quantity(problem: PowerModel, name: str, index: int) -> Any:
    try:
        return problem.quantity(name, index)
    except KeyError as exc:
        raise NotImplementedError(
            f"POWER_MODELS/{problem.formulation} does not support constraints on {name!r}"
        ) from exc


def _generator_bounds(
    problem: PowerModel,
    name: str,
    quantity: str,
    generator_rows: int | Iterable[int] | None,
    minimum: float | Iterable[float] | None,
    maximum: float | Iterable[float] | None,
) -> Any:
    _validate_name(name)
    indices = problem.network.generator_indices(_selection(generator_rows))
    if not indices:
        raise ValueError("generator bounds require at least one active generator")
    lower, upper = _bounded_values(minimum, maximum, len(indices), "minimum", "maximum")
    scale = problem.network.base_mva
    bounds = tuple(
        (
            _quantity(problem, quantity, index),
            None if lo is None else lo / scale,
            None if hi is None else hi / scale,
        )
        for index, lo, hi in zip(indices, lower, upper)
    )
    return problem.add_bounded_constraints(name, bounds)


def constraint_gen_setpoint_active(
    problem: PowerModel,
    name: str,
    *,
    generator_row: int,
    value_mw: float,
) -> Any:
    """Fix one active source generator row to an MW setpoint."""
    _validate_name(name)
    value = _finite_values(value_mw, 1, "value_mw")[0]
    index = problem.network.generator_indices((generator_row,))[0]
    return problem.add_equality_constraints(
        name,
        ((_quantity(problem, "pg", index), value / problem.network.base_mva),),
    )


def constraint_gen_setpoint_reactive(
    problem: PowerModel,
    name: str,
    *,
    generator_row: int,
    value_mvar: float,
) -> Any:
    """Fix one reactive source generator row to an MVAr setpoint."""
    _validate_name(name)
    value = _finite_values(value_mvar, 1, "value_mvar")[0]
    index = problem.network.generator_indices((generator_row,))[0]
    return problem.add_equality_constraints(
        name,
        ((_quantity(problem, "qg", index), value / problem.network.base_mva),),
    )


def constraint_gen_active_bounds(
    problem: PowerModel,
    name: str,
    *,
    generator_rows: int | Iterable[int] | None = None,
    minimum_mw: float | Iterable[float] | None = None,
    maximum_mw: float | Iterable[float] | None = None,
) -> Any:
    """Add per-generator active-power bounds in MW."""
    return _generator_bounds(problem, name, "pg", generator_rows, minimum_mw, maximum_mw)


def constraint_gen_reactive_bounds(
    problem: PowerModel,
    name: str,
    *,
    generator_rows: int | Iterable[int] | None = None,
    minimum_mvar: float | Iterable[float] | None = None,
    maximum_mvar: float | Iterable[float] | None = None,
) -> Any:
    """Add per-generator reactive-power bounds in MVAr."""
    return _generator_bounds(problem, name, "qg", generator_rows, minimum_mvar, maximum_mvar)


def constraint_active_generation_sum(
    problem: PowerModel,
    name: str,
    *,
    generator_rows: int | Iterable[int] | None = None,
    minimum_mw: float | None = None,
    maximum_mw: float | None = None,
) -> Any:
    """Bound total active generation for selected source generator rows."""
    _validate_name(name)
    lower, upper = _bounded_values(minimum_mw, maximum_mw, 1, "minimum_mw", "maximum_mw")
    indices = problem.network.generator_indices(_selection(generator_rows))
    if not indices:
        raise ValueError("active-generation sum requires at least one active generator")
    expression = sum(_quantity(problem, "pg", index) for index in indices)
    scale = problem.network.base_mva
    return problem.add_bounded_constraint(
        name,
        expression,
        lower=None if lower[0] is None else lower[0] / scale,
        upper=None if upper[0] is None else upper[0] / scale,
    )


def _bus_indices(problem: PowerModel, bus_ids: int | Iterable[int] | None) -> tuple[int, ...]:
    if bus_ids is None:
        return tuple(range(len(problem.network.bus)))
    values = (int(bus_ids),) if isinstance(bus_ids, int) else tuple(int(item) for item in bus_ids)
    if len(set(values)) != len(values):
        raise ValueError("duplicate bus IDs are not allowed")
    return tuple(problem.network.bus_index(bus_id) for bus_id in values)


def _voltage_quantity(problem: PowerModel, index: int) -> tuple[Any, bool]:
    for name in ("vm", "phi"):
        try:
            value = problem.quantity(name, index)
        except KeyError:
            continue
        return (1.0 + value, False) if name == "phi" else (value, False)
    for name in ("voltage_magnitude_squared", "w"):
        try:
            return problem.quantity(name, index), True
        except KeyError:
            continue
    raise NotImplementedError(f"POWER_MODELS/{problem.formulation} does not expose a voltage-magnitude quantity")


def constraint_voltage_magnitude_setpoint(
    problem: PowerModel,
    name: str,
    *,
    bus_id: int,
    value_pu: float,
) -> Any:
    """Fix the voltage magnitude at one active MATPOWER bus number."""
    _validate_name(name)
    value = _finite_values(value_pu, 1, "value_pu")[0]
    if value < 0:
        raise ValueError("value_pu cannot be negative")
    expression, squared = _voltage_quantity(problem, problem.network.bus_index(bus_id))
    return problem.add_equality_constraints(name, ((expression, value**2 if squared else value),))


def constraint_voltage_magnitude_bounds(
    problem: PowerModel,
    name: str,
    *,
    bus_ids: int | Iterable[int] | None = None,
    minimum_pu: float | Iterable[float] | None = None,
    maximum_pu: float | Iterable[float] | None = None,
) -> Any:
    """Add voltage-magnitude bounds at active MATPOWER bus numbers."""
    _validate_name(name)
    indices = _bus_indices(problem, bus_ids)
    if not indices:
        raise ValueError("voltage bounds require at least one active bus")
    lower, upper = _bounded_values(minimum_pu, maximum_pu, len(indices), "minimum_pu", "maximum_pu")
    if any(value is not None and value < 0 for value in (*lower, *upper)):
        raise ValueError("voltage-magnitude bounds cannot be negative")
    bounds = []
    for index, lo, hi in zip(indices, lower, upper):
        expression, squared = _voltage_quantity(problem, index)
        bounds.append(
            (
                expression,
                None if lo is None else lo**2 if squared else lo,
                None if hi is None else hi**2 if squared else hi,
            )
        )
    return problem.add_bounded_constraints(name, tuple(bounds))


def _branch_end_quantity(problem: PowerModel, index: int, end: str, quantity: str) -> Any:
    if end not in ("from", "to"):
        raise ValueError("branch end must be 'from' or 'to'")
    internal_end = end
    if bool(problem.network.branch_reversed[index]):
        internal_end = "to" if end == "from" else "from"
    direct_name = ("p" if quantity == "active" else "q") + internal_end[0]
    try:
        return problem.quantity(direct_name, index)
    except KeyError:
        if quantity != "active":
            raise NotImplementedError(
                f"POWER_MODELS/{problem.formulation} does not model reactive branch flow"
            ) from None
    try:
        flow = problem.quantity("p", index)
    except KeyError as exc:
        raise NotImplementedError(f"POWER_MODELS/{problem.formulation} does not model active branch flow") from exc
    return flow if internal_end == "from" else -flow


def _branch_power_bounds(
    problem: PowerModel,
    name: str,
    quantity: str,
    branch_rows: int | Iterable[int] | None,
    end: str,
    minimum: float | Iterable[float] | None,
    maximum: float | Iterable[float] | None,
) -> Any:
    _validate_name(name)
    indices = problem.network.branch_indices(_selection(branch_rows))
    if not indices:
        raise ValueError("branch-flow bounds require at least one active branch")
    lower, upper = _bounded_values(minimum, maximum, len(indices), "minimum", "maximum")
    scale = problem.network.base_mva
    return problem.add_bounded_constraints(
        name,
        tuple(
            (
                _branch_end_quantity(problem, index, end, quantity),
                None if lo is None else lo / scale,
                None if hi is None else hi / scale,
            )
            for index, lo, hi in zip(indices, lower, upper)
        ),
    )


def constraint_branch_active_power_bounds(
    problem: PowerModel,
    name: str,
    *,
    branch_rows: int | Iterable[int] | None = None,
    end: str = "from",
    minimum_mw: float | Iterable[float] | None = None,
    maximum_mw: float | Iterable[float] | None = None,
) -> Any:
    """Add active-power bounds at an original MATPOWER branch end."""
    return _branch_power_bounds(problem, name, "active", branch_rows, end, minimum_mw, maximum_mw)


def constraint_branch_reactive_power_bounds(
    problem: PowerModel,
    name: str,
    *,
    branch_rows: int | Iterable[int] | None = None,
    end: str = "from",
    minimum_mvar: float | Iterable[float] | None = None,
    maximum_mvar: float | Iterable[float] | None = None,
) -> Any:
    """Add reactive-power bounds at an original MATPOWER branch end."""
    return _branch_power_bounds(problem, name, "reactive", branch_rows, end, minimum_mvar, maximum_mvar)


def constraint_branch_apparent_power_limit(
    problem: PowerModel,
    name: str,
    *,
    branch_rows: int | Iterable[int] | None = None,
    end: str = "both",
    maximum_mva: float | Iterable[float],
) -> Any:
    """Add apparent-power limits at one or both original branch ends."""
    _validate_name(name)
    if end not in ("from", "to", "both"):
        raise ValueError("branch end must be 'from', 'to', or 'both'")
    indices = problem.network.branch_indices(_selection(branch_rows))
    if not indices:
        raise ValueError("apparent-power limits require at least one active branch")
    limits = _finite_values(maximum_mva, len(indices), "maximum_mva")
    if any(limit < 0 for limit in limits):
        raise ValueError("maximum_mva cannot be negative")
    scale = problem.network.base_mva
    bounds = []
    for index, limit in zip(indices, limits):
        for selected_end in ("from", "to") if end == "both" else (end,):
            active = _branch_end_quantity(problem, index, selected_end, "active")
            reactive = _branch_end_quantity(problem, index, selected_end, "reactive")
            bounds.append((active**2 + reactive**2, None, (limit / scale) ** 2))
    return problem.add_bounded_constraints(name, tuple(bounds))


def _angle_expression(problem: PowerModel, index: int) -> tuple[Any, Any | None]:
    network = problem.network
    sign = -1.0 if bool(network.branch_reversed[index]) else 1.0
    try:
        va = problem.quantity("va")
    except KeyError:
        va = None
    if va is not None:
        difference = va[int(network.f_bus[index])] - va[int(network.t_bus[index])]
        return sign * difference, None
    try:
        real = problem.quantity("branch_voltage_product_real", index)
        imaginary = problem.quantity("branch_voltage_product_imaginary", index)
    except KeyError:
        try:
            pair = int(branch_pair_indices(network)[index])
            real = problem.quantity("wr", pair)
            imaginary = problem.quantity("wi", pair)
        except KeyError as exc:
            raise NotImplementedError(
                f"POWER_MODELS/{problem.formulation} does not expose a voltage-angle quantity"
            ) from exc
    return real, sign * imaginary


def constraint_voltage_angle_difference(
    problem: PowerModel,
    name: str,
    *,
    branch_rows: int | Iterable[int] | None = None,
    minimum_degrees: float | Iterable[float] | None = None,
    maximum_degrees: float | Iterable[float] | None = None,
) -> Any:
    """Bound original-direction voltage-angle differences in degrees."""
    _validate_name(name)
    indices = problem.network.branch_indices(_selection(branch_rows))
    if not indices:
        raise ValueError("angle constraints require at least one active branch")
    lower, upper = _bounded_values(
        minimum_degrees,
        maximum_degrees,
        len(indices),
        "minimum_degrees",
        "maximum_degrees",
    )
    if any(value is not None and not -90.0 < value < 90.0 for value in (*lower, *upper)):
        raise ValueError("angle bounds must lie strictly between -90 and 90 degrees")
    bounds = []
    for index, lo, hi in zip(indices, lower, upper):
        first, imaginary = _angle_expression(problem, index)
        if imaginary is None:
            bounds.append(
                (
                    first,
                    None if lo is None else math.radians(lo),
                    None if hi is None else math.radians(hi),
                )
            )
        else:
            if lo is not None:
                bounds.append((imaginary - math.tan(math.radians(lo)) * first, 0.0, None))
            if hi is not None:
                bounds.append((imaginary - math.tan(math.radians(hi)) * first, None, 0.0))
    return problem.add_bounded_constraints(name, tuple(bounds))


def constraint_interface_active_power(
    problem: PowerModel,
    name: str,
    *,
    branch_rows: Iterable[int],
    weights: float | Iterable[float] = 1.0,
    end: str = "from",
    minimum_mw: float | None = None,
    maximum_mw: float | None = None,
) -> Any:
    """Bound a weighted sum of original-direction branch active flows."""
    _validate_name(name)
    indices = problem.network.branch_indices(_selection(branch_rows))
    if not indices:
        raise ValueError("an interface requires at least one active branch")
    coefficients = _finite_values(weights, len(indices), "weights")
    lower, upper = _bounded_values(minimum_mw, maximum_mw, 1, "minimum_mw", "maximum_mw")
    expression = sum(
        coefficient * _branch_end_quantity(problem, index, end, "active")
        for index, coefficient in zip(indices, coefficients)
    )
    scale = problem.network.base_mva
    return problem.add_bounded_constraint(
        name,
        expression,
        lower=None if lower[0] is None else lower[0] / scale,
        upper=None if upper[0] is None else upper[0] / scale,
    )


__all__ = [
    "constraint_active_generation_sum",
    "constraint_branch_active_power_bounds",
    "constraint_branch_apparent_power_limit",
    "constraint_branch_reactive_power_bounds",
    "constraint_gen_active_bounds",
    "constraint_gen_reactive_bounds",
    "constraint_gen_setpoint_active",
    "constraint_gen_setpoint_reactive",
    "constraint_interface_active_power",
    "constraint_voltage_angle_difference",
    "constraint_voltage_magnitude_bounds",
    "constraint_voltage_magnitude_setpoint",
]
