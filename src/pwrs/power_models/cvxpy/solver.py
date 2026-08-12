# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""CVXPY solver selection, execution, and semantic result extraction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import numpy as np

from ..extensions import extension_summary
from ..results import PowerModelSolution
from .common import import_cvxpy
from .context import BoundConstraint, CvxpyPowerModel


@dataclass(frozen=True)
class _SolverSelection:
    requested: str
    selected: str
    fallbacks: tuple[dict[str, str], ...]


def classify_cvxpy_power_model(problem: CvxpyPowerModel) -> str:
    """Return the declared cone class for a CVXPY formulation."""
    return problem.cone_kind


def _requested_solver(mpopt: Any) -> str:
    requested = str(mpopt.opf.power_models.solver).upper()
    legacy = str(mpopt.opf.ac.solver).upper()
    if requested != "DEFAULT" and legacy != "DEFAULT" and requested != legacy:
        raise ValueError(
            f"conflicting PowerModels solver options: opf.power_models.solver={requested!r}, legacy solver={legacy!r}"
        )
    return legacy if requested == "DEFAULT" and legacy != "DEFAULT" else requested


def _select_solver(problem: CvxpyPowerModel, mpopt: Any, cp: Any) -> _SolverSelection:
    requested = _requested_solver(mpopt)
    compatible = {"CLARABEL", "SCS", "MOSEK"}
    if requested != "DEFAULT" and requested not in compatible:
        raise ValueError(f"POWER_MODELS/{problem.formulation}/CVXPY requires a conic solver; got {requested!r}")
    candidates = (requested,) if requested != "DEFAULT" else ("CLARABEL", "SCS")
    installed = {str(name).upper() for name in cp.installed_solvers()}
    fallbacks: list[dict[str, str]] = []
    for candidate in candidates:
        if candidate in installed:
            return _SolverSelection(requested, candidate, tuple(fallbacks))
        if requested != "DEFAULT":
            raise RuntimeError(f"requested PowerModels solver {candidate} is unavailable")
        fallbacks.append({"solver": candidate, "reason": "solver is not installed or unavailable"})
    raise RuntimeError(f"no compatible {problem.cone_kind} solver is available")


def _solver_options(mpopt: Any, solver: str) -> dict[str, Any]:
    field = f"{solver.lower()}_options"
    return dict(getattr(mpopt.opf.power_models, field, {}) or {})


def _expression_value(expression: Any) -> np.ndarray:
    value = expression.value
    if value is None:
        raise RuntimeError(f"CVXPY did not return a value for {expression}")
    return np.asarray(value, dtype=float).reshape(-1)


def _dual_value(constraint: Any) -> np.ndarray:
    value = constraint.dual_value
    if value is None:
        return np.zeros(0)
    if isinstance(value, (list, tuple)):
        # For a SOC, the first item is the scalar dual paired with its radius.
        value = value[0]
    return np.asarray(value, dtype=float).reshape(-1)


def _constraint_duals(problem: CvxpyPowerModel) -> dict[str, np.ndarray]:
    return {
        name: np.concatenate([_dual_value(constraint) for constraint in constraints])
        for name, constraints in problem.constraints.items()
    }


def _bound_duals(bounds: dict[str, BoundConstraint]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for name, bound in bounds.items():
        result = np.zeros(bound.size)
        dual = _dual_value(bound.constraint)
        if dual.size:
            result[bound.indices] = dual
        values[name] = result
    return values


def _max_constraint_violation(problem: CvxpyPowerModel) -> float:
    maximum = 0.0
    for constraint in problem.all_constraints:
        try:
            value = np.asarray(constraint.violation(), dtype=float)
        except (ValueError, TypeError):
            continue
        if value.size:
            maximum = max(maximum, float(np.max(value)))
    return maximum


def _extension_results(problem: CvxpyPowerModel) -> dict[str, Any]:
    output = {}
    for record in problem.extension_records:
        groups = {}
        for name in record.constraint_groups:
            constraints = problem.constraints[name]
            native_dual = np.concatenate([_dual_value(constraint) for constraint in constraints]).tolist()
            senses = problem.constraint_senses.get(name, ())
            violation = []
            for constraint in constraints:
                try:
                    values = np.asarray(constraint.violation(), dtype=float).reshape(-1)
                except (ValueError, TypeError):
                    values = np.zeros(0)
                violation.extend(values.tolist())
            groups[name] = {
                "dual": native_dual,
                "native_dual": native_dual,
                "senses": list(senses),
                "violation": violation,
                "max_violation": max(violation, default=0.0),
            }
        output[record.name] = {
            "build_time": record.build_time,
            "constraints": groups,
            "variables": {
                name: _expression_value(problem.variables[name]).tolist()
                for name in record.variable_groups
            },
            "objective_terms": {
                name: float(_expression_value(problem.objective_terms[name])[0])
                for name in record.objective_terms
            },
        }
    return output


def _solver_version(name: str) -> str | None:
    try:
        if name == "CLARABEL":
            return version("clarabel")
        if name == "SCS":
            import scs

            return str(scs.__version__)
        if name == "MOSEK":
            import mosek

            return str(mosek.Env().getversion())
    except Exception:
        return None
    return None


def solve_cvxpy_power_model(problem: CvxpyPowerModel, mpopt: Any, nargout: int = 1):
    """Validate, solve, and map a conic PowerModels formulation."""
    problem.option_validator(mpopt)
    cp = import_cvxpy()
    selection = _select_solver(problem, mpopt, cp)
    model = cp.Problem(cp.Minimize(problem.objective), problem.all_constraints)
    solve_start = time.perf_counter()
    options = _solver_options(mpopt, selection.selected)
    options.setdefault("verbose", bool(int(mpopt.verbose)))
    try:
        objective = model.solve(solver=selection.selected, **options)
    except cp.error.SolverError as exc:
        if selection.requested != "DEFAULT" or selection.selected != "CLARABEL" or "SCS" not in cp.installed_solvers():
            raise
        fallbacks = (*selection.fallbacks, {"solver": "CLARABEL", "reason": f"solve failed: {exc}"})
        selection = _SolverSelection("DEFAULT", "SCS", fallbacks)
        options = _solver_options(mpopt, selection.selected)
        options.setdefault("verbose", bool(int(mpopt.verbose)))
        objective = model.solve(solver=selection.selected, **options)
    solve_time = time.perf_counter() - solve_start
    success = int(model.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
    if not success or objective is None:
        raise RuntimeError(f"{selection.selected} failed to solve POWER_MODELS/{problem.formulation}: {model.status}")

    variable_values = {name: _expression_value(value) for name, value in problem.variables.items()}
    variable_values.update({name: _expression_value(value) for name, value in problem.expressions.items()})
    vector = np.concatenate([variable_values[name] for name in problem.variable_result_groups])
    solution = PowerModelSolution(
        vector=vector,
        variables=variable_values,
        constraint_multipliers=_constraint_duals(problem),
        lower_bound_multipliers=_bound_duals(problem.lower_bounds),
        upper_bound_multipliers=_bound_duals(problem.upper_bounds),
    )
    stats = model.solver_stats
    info = {
        "status": 0 if success else -1,
        "status_msg": model.status,
        "termination_condition": model.status,
        "native_status": model.status,
        "iter_count": stats.num_iters,
        "solver_name": selection.selected,
    }
    result_start = time.perf_counter()
    result, raw = problem.result_builder(problem.network, solution, float(objective), success, info)
    result_time = time.perf_counter() - result_start
    extension_data = extension_summary(problem)
    raw["extensions"] = _extension_results(problem)
    raw["output"].update(
        {
            "alg": f"POWER_MODELS/{problem.formulation}/CVXPY/{selection.selected}",
            "formulation": problem.formulation,
            "implementation": "CVXPY",
            "solver_interface": "CVXPY",
            "solver": {
                "requested": selection.requested,
                "selected": selection.selected,
                "name": selection.selected,
                "interface": "CVXPY",
                "version": _solver_version(selection.selected),
                "problem_class": problem.cone_kind,
                "base_problem_class": problem.base_problem_class,
                "reclassified": problem.base_problem_class != problem.cone_kind,
                "status": info["status"],
                "termination_condition": model.status,
                "native_status": model.status,
                "fallbacks": list(selection.fallbacks),
            },
            "max_constraint_violation": _max_constraint_violation(problem),
            "extensions": extension_data,
            "timings": {
                "network_build": problem.network_build_time,
                "model_build": problem.model_build_time,
                "extensions": sum(record.build_time for record in problem.extension_records),
                "solver_call": solve_time,
                "solver_setup": stats.setup_time,
                "solve": stats.solve_time,
                "result": result_time,
            },
        }
    )
    outputs = (result, success, raw)
    return outputs[:nargout] if nargout > 1 else result


__all__ = ["classify_cvxpy_power_model", "solve_cvxpy_power_model"]
