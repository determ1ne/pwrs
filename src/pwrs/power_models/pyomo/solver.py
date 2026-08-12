# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Solver execution and solution loading for Pyomo power models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..extensions import extension_summary
from ..results import PowerModelSolution
from .context import PyomoPowerModel


@dataclass(frozen=True)
class _ModelCapabilities:
    problem_class: str
    highs_compatible: bool
    glpk_compatible: bool
    detail: str


@dataclass(frozen=True)
class _SolverSelection:
    requested: str
    selected: str
    factory_name: str
    capabilities: _ModelCapabilities
    fallbacks: tuple[dict[str, str], ...]


def import_pyomo() -> Any:
    try:
        import pyomo.environ as pyo
    except ImportError as exc:
        raise ImportError("this PowerModels formulation requires pyomo") from exc
    return pyo


def _split_groups(
    values: np.ndarray, names: tuple[str, ...], registry: dict[str, tuple[Any, ...]]
) -> dict[str, np.ndarray]:
    groups: dict[str, np.ndarray] = {}
    offset = 0
    for name in names:
        size = len(registry[name])
        groups[name] = values[offset : offset + size]
        offset += size
    return groups


def _ipopt_options(mpopt: Any) -> dict[str, Any]:
    options = dict(getattr(mpopt.ipopt, "opts", {}) or {})
    options.setdefault("print_level", min(12, int(mpopt.verbose) * 2 + 1) if int(mpopt.verbose) else 0)
    return options


def _solver_version(name: str) -> str | None:
    try:
        if name == "HIGHS":
            import highspy

            return str(highspy.Highs().version())
        if name == "IPOPT":
            import cyipopt

            return str(cyipopt.__version__)
    except Exception:
        return None
    return None


def _constraint_violation(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    violation = np.zeros_like(values)
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    violation[finite_lower] = np.maximum(violation[finite_lower], lower[finite_lower] - values[finite_lower])
    violation[finite_upper] = np.maximum(violation[finite_upper], values[finite_upper] - upper[finite_upper])
    return float(np.max(violation, initial=0.0))


def _extension_results(problem: PyomoPowerModel) -> dict[str, Any]:
    pyo = import_pyomo()
    output = {}
    for record in problem.extension_records:
        groups = {}
        for name in record.constraint_groups:
            constraints = problem.constraints[name]
            native_dual = [float(problem.model.dual.get(constraint, 0.0)) for constraint in constraints]
            senses = problem.constraint_senses.get(name, ())
            dual = (
                [-value if sense == "lower" else value for value, sense in zip(native_dual, senses)]
                if senses
                else native_dual
            )
            violation = []
            for constraint in constraints:
                body = float(pyo.value(constraint.body))
                lower = float(pyo.value(constraint.lower)) if constraint.lower is not None else -np.inf
                upper = float(pyo.value(constraint.upper)) if constraint.upper is not None else np.inf
                violation.append(max(0.0, lower - body, body - upper))
            groups[name] = {
                "dual": dual,
                "native_dual": native_dual,
                "senses": list(senses),
                "violation": violation,
                "max_violation": max(violation, default=0.0),
            }
        output[record.name] = {
            "build_time": record.build_time,
            "constraints": groups,
            "variables": {
                name: [float(pyo.value(value)) for value in problem.variables[name]]
                for name in record.variable_groups
            },
            "objective_terms": {
                name: float(pyo.value(problem.objective_terms[name]))
                for name in record.objective_terms
            },
        }
    return output


def _analyze_model(model: Any, pyo: Any) -> _ModelCapabilities:
    from pyomo.repn.standard_repn import generate_standard_repn

    if any(not variable.is_continuous() for variable in model.component_data_objects(pyo.Var, active=True)):
        return _ModelCapabilities("MIP", False, False, "integer variables are not yet supported")

    for constraint in model.component_data_objects(pyo.Constraint, active=True):
        repn: Any = generate_standard_repn(constraint.body, quadratic=True)
        if not repn.is_linear():
            return _ModelCapabilities(
                "NLP",
                False,
                False,
                f"nonlinear constraint {constraint.name!r}",
            )

    objectives = tuple(model.component_data_objects(pyo.Objective, active=True))
    if len(objectives) != 1:
        return _ModelCapabilities("NLP", False, False, "exactly one active objective is required")
    objective = objectives[0]
    repn: Any = generate_standard_repn(objective.expr, quadratic=True)
    if repn.is_linear():
        return _ModelCapabilities("LP", True, True, "linear objective and constraints")
    if not repn.is_quadratic() or objective.sense != pyo.minimize:
        return _ModelCapabilities("NLP", False, False, "non-quadratic or maximizing objective")

    tolerance = 1e-12
    for coefficient, variables in zip(repn.quadratic_coefs, repn.quadratic_vars):
        if variables[0] is not variables[1] or float(coefficient) < -tolerance:
            return _ModelCapabilities(
                "NLP",
                False,
                False,
                "quadratic objective convexity is not safely certifiable",
            )
    return _ModelCapabilities("QP", True, False, "convex separable quadratic objective")


def classify_pyomo_power_model(problem: PyomoPowerModel) -> str:
    """Return the solver problem class for the model in its current state."""
    return _analyze_model(problem.model, import_pyomo()).problem_class


def _requested_solver(mpopt: Any) -> str:
    requested = str(mpopt.opf.power_models.solver).upper()
    legacy = str(mpopt.opf.dc.solver if str(mpopt.model).upper() == "DC" else mpopt.opf.ac.solver).upper()
    if requested != "DEFAULT" and legacy != "DEFAULT" and requested != legacy:
        raise ValueError(
            f"conflicting PowerModels solver options: opf.power_models.solver={requested!r}, legacy solver={legacy!r}"
        )
    if requested == "DEFAULT" and legacy != "DEFAULT":
        requested = legacy
    return requested


def _solver_candidates(requested: str, capabilities: _ModelCapabilities) -> tuple[str, ...]:
    if requested != "DEFAULT":
        if requested == "HIGHS" and not capabilities.highs_compatible:
            raise ValueError(f"HiGHS cannot solve this {capabilities.problem_class} model: {capabilities.detail}")
        if requested == "GLPK" and not capabilities.glpk_compatible:
            raise ValueError(f"GLPK cannot solve this {capabilities.problem_class} model: {capabilities.detail}")
        if capabilities.problem_class == "MIP":
            raise ValueError(capabilities.detail)
        return (requested,)
    if capabilities.problem_class == "LP":
        return ("HIGHS", "GLPK", "IPOPT")
    if capabilities.problem_class == "QP":
        return ("HIGHS", "IPOPT")
    if capabilities.problem_class == "NLP":
        return ("IPOPT",)
    raise ValueError(capabilities.detail)


def _select_solver(problem: PyomoPowerModel, mpopt: Any, pyo: Any) -> _SolverSelection:
    requested = _requested_solver(mpopt)
    capabilities = _analyze_model(problem.model, pyo)
    factory_names = {"HIGHS": "highs", "GLPK": "glpk", "IPOPT": "cyipopt"}
    fallbacks: list[dict[str, str]] = []
    candidates = _solver_candidates(requested, capabilities)
    for candidate in candidates:
        factory_name = factory_names[candidate]
        try:
            available = bool(pyo.SolverFactory(factory_name).available(exception_flag=False))
        except Exception as exc:
            available = False
            reason = f"availability check failed: {exc}"
        else:
            reason = "solver is not installed or unavailable"
        if available:
            return _SolverSelection(requested, candidate, factory_name, capabilities, tuple(fallbacks))
        if requested != "DEFAULT":
            raise RuntimeError(f"requested PowerModels solver {candidate} is unavailable")
        fallbacks.append({"solver": candidate, "reason": reason})
    raise RuntimeError("no compatible PowerModels solver is available")


def _build_outputs(
    problem: PyomoPowerModel,
    solution: PowerModelSolution,
    objective: float,
    success: int,
    info: dict[str, Any],
    *,
    selection: _SolverSelection,
    solver_version: str | None,
    max_constraint_violation: float,
    timings: dict[str, float],
    nargout: int,
):
    result_start = time.perf_counter()
    result, raw = problem.result_builder(problem.network, solution, objective, success, info)
    timings["result"] = time.perf_counter() - result_start
    extension_data = extension_summary(problem)
    timings["extensions"] = sum(record.build_time for record in problem.extension_records)
    raw["extensions"] = _extension_results(problem)
    raw["output"].update(
        {
            "alg": f"POWER_MODELS/{problem.formulation}/PYOMO/{selection.selected}",
            "formulation": problem.formulation,
            "implementation": "PYOMO",
            "solver_interface": "PYOMO",
            "solver": {
                "requested": selection.requested,
                "selected": selection.selected,
                "name": selection.selected,
                "interface": "PYOMO",
                "version": solver_version,
                "problem_class": selection.capabilities.problem_class,
                "base_problem_class": problem.base_problem_class,
                "reclassified": problem.base_problem_class != selection.capabilities.problem_class,
                "status": int(info["status"]),
                "termination_condition": info.get("termination_condition"),
                "native_status": info.get("native_status"),
                "fallbacks": list(selection.fallbacks),
            },
            "max_constraint_violation": max_constraint_violation,
            "extensions": extension_data,
            "timings": {
                "network_build": problem.network_build_time,
                "model_build": problem.model_build_time,
                **timings,
            },
        }
    )
    outputs = (result, success, raw)
    return outputs[:nargout] if nargout > 1 else result


def _solve_with_cyipopt_factory(
    problem: PyomoPowerModel,
    mpopt: Any,
    selection: _SolverSelection,
    nargout: int,
):
    pyo = import_pyomo()
    solver: Any = pyo.SolverFactory(selection.factory_name)

    solve_start = time.perf_counter()
    results, nlp = solver.solve(
        problem.model,
        options=_ipopt_options(mpopt),
        tee=bool(int(mpopt.verbose)),
        load_solutions=False,
        return_nlp=True,
    )
    solver_call_time = time.perf_counter() - solve_start

    solution_record = results.solution(0)
    symbol_map_id = results._smap_id
    pyomo_variables = nlp.get_pyomo_variables()
    pyomo_constraints = nlp.get_pyomo_constraints()
    variable_records = tuple(solution_record.variable.values())
    constraint_records = tuple(solution_record.constraint.values())
    if len(variable_records) != len(pyomo_variables) or len(constraint_records) != len(pyomo_constraints):
        raise RuntimeError("Pyomo cyipopt result ordering does not match the returned NLP")
    x_all = np.asarray([record["Value"] for record in variable_records], dtype=float)
    mult_g_all = np.asarray([record.get("Dual", 0.0) for record in constraint_records], dtype=float)
    mult_x_l_all = np.asarray([record.get("ipopt_zL_out", 0.0) for record in variable_records], dtype=float)
    mult_x_u_all = np.asarray([record.get("ipopt_zU_out", 0.0) for record in variable_records], dtype=float)
    problem.model.dual.clear()
    problem.model.ipopt_zL_out.clear()
    problem.model.ipopt_zU_out.clear()
    try:
        for variable, value in zip(pyomo_variables, x_all):
            variable.set_value(float(value), skip_validation=True)
        problem.model.dual.update(zip(pyomo_constraints, mult_g_all))
        problem.model.ipopt_zL_out.update(zip(pyomo_variables, mult_x_l_all))
        problem.model.ipopt_zU_out.update(zip(pyomo_variables, mult_x_u_all))
    finally:
        problem.model.solutions.delete_symbol_map(symbol_map_id)

    base_variable_indices = np.asarray(nlp.get_primal_indices(list(problem.variable_order)), dtype=np.intp)
    base_constraint_indices = np.asarray(nlp.get_constraint_indices(list(problem.constraint_order)), dtype=np.intp)
    x = x_all[base_variable_indices]
    mult_g = mult_g_all[base_constraint_indices]
    mult_x_l = mult_x_l_all[base_variable_indices]
    mult_x_u = mult_x_u_all[base_variable_indices]
    solution = PowerModelSolution(
        vector=x,
        variables=_split_groups(x, problem.variable_result_groups, problem.variables),
        constraint_multipliers=_split_groups(mult_g, problem.constraint_result_groups, problem.constraints),
        lower_bound_multipliers=_split_groups(mult_x_l, problem.variable_result_groups, problem.variables),
        upper_bound_multipliers=_split_groups(mult_x_u, problem.variable_result_groups, problem.variables),
    )

    status = int(results.solver.return_code)
    success = int(status in (0, 1))
    objective_component = next(problem.model.component_data_objects(pyo.Objective, active=True))
    objective = float(pyo.value(objective_component))
    status_msg = results.solver.message
    info = {
        "status": status,
        "status_msg": status_msg,
        "termination_condition": str(results.solver.termination_condition),
        "native_status": str(results.solver.status),
        "mult_g": mult_g,
        "mult_x_L": mult_x_l,
        "mult_x_U": mult_x_u,
    }

    values = np.asarray(nlp.evaluate_constraints(), dtype=float)
    lower = np.asarray(nlp.constraints_lb(), dtype=float)
    upper = np.asarray(nlp.constraints_ub(), dtype=float)
    native_solve_time = float(results.solver.wallclock_time)
    return _build_outputs(
        problem,
        solution,
        objective,
        success,
        info,
        selection=selection,
        solver_version=_solver_version("IPOPT"),
        max_constraint_violation=_constraint_violation(values, lower, upper),
        timings={
            "solver_call": solver_call_time,
            "solver_setup": max(0.0, solver_call_time - native_solve_time),
            "ipopt": native_solve_time,
        },
        nargout=nargout,
    )


def _algebraic_solver_options(mpopt: Any, solver_name: str) -> dict[str, Any]:
    if solver_name == "HIGHS":
        return dict(mpopt.opf.power_models.highs_options)
    if solver_name == "GLPK":
        return dict(mpopt.opf.power_models.glpk_options)
    raise ValueError(f"unsupported algebraic solver: {solver_name}")


def _solve_with_algebraic_factory(
    problem: PyomoPowerModel,
    mpopt: Any,
    selection: _SolverSelection,
    nargout: int,
):
    pyo = import_pyomo()
    solver = pyo.SolverFactory(selection.factory_name)
    model = problem.model
    model.dual.clear()
    model.rc.clear()

    solve_start = time.perf_counter()
    solve_kwargs = {
        "options": _algebraic_solver_options(mpopt, selection.selected),
        "tee": bool(int(mpopt.verbose)),
    }
    if selection.selected == "HIGHS":
        solve_kwargs["load_solutions"] = False
    results = solver.solve(model, **solve_kwargs)
    solver_call_time = time.perf_counter() - solve_start

    termination = str(results.solver.termination_condition)
    success = int(termination in ("optimal", "locallyOptimal", "globallyOptimal"))
    if selection.selected == "HIGHS" and success:
        for variable, value in solver._get_primals().items():
            variable.set_value(float(value), skip_validation=True)
        raw_duals = {constraint: float(value) for constraint, value in solver._get_duals().items()}
        reduced_cost_map = solver._get_reduced_costs()
    else:
        raw_duals = {constraint: float(value) for constraint, value in model.dual.items()}
        reduced_cost_map = model.rc
    model.dual.clear()
    model.dual.update((constraint, -value) for constraint, value in raw_duals.items())
    if reduced_cost_map is not model.rc:
        model.rc.clear()
        model.rc.update(reduced_cost_map.items())
    variables = problem.variable_order
    constraints = problem.constraint_order
    x = np.asarray([pyo.value(variable) for variable in variables], dtype=float)
    mult_g = np.asarray([model.dual.get(constraint, 0.0) for constraint in constraints], dtype=float)
    reduced_costs = np.asarray([reduced_cost_map.get(variable, 0.0) for variable in variables], dtype=float)
    mult_x_l = np.maximum(reduced_costs, 0.0)
    mult_x_u = np.maximum(-reduced_costs, 0.0)
    model.ipopt_zL_out.clear()
    model.ipopt_zL_out.update(zip(variables, mult_x_l))
    model.ipopt_zU_out.clear()
    model.ipopt_zU_out.update(zip(variables, mult_x_u))
    solution = PowerModelSolution(
        vector=x,
        variables=_split_groups(x, problem.variable_result_groups, problem.variables),
        constraint_multipliers=_split_groups(mult_g, problem.constraint_result_groups, problem.constraints),
        lower_bound_multipliers=_split_groups(mult_x_l, problem.variable_result_groups, problem.variables),
        upper_bound_multipliers=_split_groups(mult_x_u, problem.variable_result_groups, problem.variables),
    )

    status = 0 if success else -1
    objective_component = next(model.component_data_objects(pyo.Objective, active=True))
    objective = float(pyo.value(objective_component))
    status_msg = str(getattr(results.solver, "message", termination))
    if status_msg == "<undefined>":
        status_msg = termination
    info = {
        "status": status,
        "status_msg": status_msg,
        "termination_condition": termination,
        "native_status": str(results.solver.status),
        "mult_g": mult_g,
        "mult_x_L": mult_x_l,
        "mult_x_U": mult_x_u,
    }
    active_constraints = tuple(model.component_data_objects(pyo.Constraint, active=True))
    values = np.asarray([pyo.value(constraint.body) for constraint in active_constraints], dtype=float)
    lower = np.asarray(
        [pyo.value(constraint.lower) if constraint.lower is not None else -np.inf for constraint in active_constraints],
        dtype=float,
    )
    upper = np.asarray(
        [pyo.value(constraint.upper) if constraint.upper is not None else np.inf for constraint in active_constraints],
        dtype=float,
    )
    return _build_outputs(
        problem,
        solution,
        objective,
        success,
        info,
        selection=selection,
        solver_version=_solver_version(selection.selected),
        max_constraint_violation=_constraint_violation(values, lower, upper),
        timings={"solver_call": solver_call_time},
        nargout=nargout,
    )


def solve_pyomo_power_model(problem: PyomoPowerModel, mpopt: Any, nargout: int = 1):
    """Solve a built and optionally extended Pyomo formulation."""
    problem.option_validator(mpopt)
    selection = _select_solver(problem, mpopt, import_pyomo())
    if selection.selected == "IPOPT":
        return _solve_with_cyipopt_factory(problem, mpopt, selection, nargout)
    return _solve_with_algebraic_factory(problem, mpopt, selection, nargout)


__all__ = [
    "classify_pyomo_power_model",
    "import_pyomo",
    "solve_pyomo_power_model",
]
