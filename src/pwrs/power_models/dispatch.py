# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Formulation registry and build/extend/solve lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from .extensions import apply_extensions


@dataclass(frozen=True)
class FormulationSpec:
    """Lazy entry points for one public PowerModels formulation."""

    name: str
    builder: str
    solver: str
    classifier: str


_PYOMO_SOLVER = "pwrs.power_models.pyomo:solve_pyomo_power_model"
_PYOMO_CLASSIFIER = "pwrs.power_models.pyomo:classify_pyomo_power_model"
_CVXPY_SOLVER = "pwrs.power_models.cvxpy:solve_cvxpy_power_model"
_CVXPY_CLASSIFIER = "pwrs.power_models.cvxpy:classify_cvxpy_power_model"


def _spec(name: str, module: str, builder: str, *, conic: bool = False) -> FormulationSpec:
    prefix = f"pwrs.power_models.{module}"
    return FormulationSpec(
        name,
        f"{prefix}:{builder}",
        _CVXPY_SOLVER if conic else _PYOMO_SOLVER,
        _CVXPY_CLASSIFIER if conic else _PYOMO_CLASSIFIER,
    )


FORMULATIONS = {
    spec.name: spec
    for spec in (
        _spec("ACP", "acp", "build_acp_model"),
        _spec("ACR", "acr", "build_acr_model"),
        _spec("ACT", "act", "build_act_model"),
        _spec("SOCWR", "socwr", "build_socwr_model"),
        _spec("DCP", "dcp", "build_dcp_model"),
        _spec("DCMP", "dcmp", "build_dcmp_model"),
        _spec("NFA", "dcp", "build_nfa_model"),
        _spec("DCPLL", "dcp", "build_dcpll_model"),
        _spec("LPACC", "lpacc", "build_lpacc_model"),
        _spec("BFA", "bf", "build_bfa_model"),
        _spec("SOCBF", "bf", "build_socbf_model"),
        _spec("IVR", "ivr", "build_ivr_model"),
        _spec("QCRM", "qc", "build_qcrm_model"),
        _spec("QCLS", "qc", "build_qcls_model"),
        _spec("SOCWRCONIC", "conic", "build_socwr_conic_model", conic=True),
        _spec("SOCBFCONIC", "conic", "build_socbf_conic_model", conic=True),
        _spec("SDPWRM", "conic", "build_sdpwrm_model", conic=True),
        _spec("SPARSESDPWRM", "conic", "build_sparse_sdpwrm_model", conic=True),
    )
}


def _resolve(path: str) -> Callable[..., Any]:
    module_name, attribute = path.split(":", 1)
    return getattr(import_module(module_name), attribute)


def formulation_spec(formulation: str) -> FormulationSpec:
    """Return a registered formulation or raise a stable public error."""
    name = str(formulation).upper()
    try:
        return FORMULATIONS[name]
    except KeyError as exc:
        raise NotImplementedError(f"unsupported PowerModels formulation: {name}") from exc


def build_power_model(mpc: Any, formulation: str) -> Any:
    """Build one registered formulation without applying extensions."""
    spec = formulation_spec(formulation)
    return _resolve(spec.builder)(mpc)


def solve_built_power_model(
    problem: Any,
    mpopt: Any,
    nargout: int = 1,
    *,
    expected_formulation: str | None = None,
):
    """Apply configured extensions and solve a previously built model."""
    if expected_formulation is not None and problem.formulation != expected_formulation:
        raise ValueError(f"expected a {expected_formulation} model, got {problem.formulation}")
    spec = formulation_spec(problem.formulation)
    classifier = _resolve(spec.classifier)
    if problem.base_problem_class is None:
        problem.base_problem_class = classifier(problem)
    apply_extensions(problem, mpopt.opf.power_models.extensions)
    return _resolve(spec.solver)(problem, mpopt, nargout)


def solve_formulation_opf(
    mpc: Any,
    mpopt: Any,
    formulation: str,
    nargout: int = 1,
):
    """Build, extend, and solve one explicitly selected formulation."""
    problem = build_power_model(mpc, formulation)
    return solve_built_power_model(
        problem,
        mpopt,
        nargout,
        expected_formulation=str(formulation).upper(),
    )


def solve_power_models_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Build, extend, and solve the formulation selected in ``mpopt``."""
    formulation = str(mpopt.opf.power_models.formulation).upper()
    return solve_formulation_opf(mpc, mpopt, formulation, nargout)


__all__ = [
    "FORMULATIONS",
    "FormulationSpec",
    "build_power_model",
    "formulation_spec",
    "solve_built_power_model",
    "solve_formulation_opf",
    "solve_power_models_opf",
]
