# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from .dataclass_util import DataclassDictMixin

type ModelType = Literal["AC", "DC"]
type PfAlg = Literal[
    "NR",
    "NR-SP",
    "NR-SC",
    "NR-SH",
    "NR-IP",
    "NR-IC",
    "NR-IH",
    "FDXB",
    "FDBX",
    "GS",
    "PQSUM",
    "ISUM",
    "YSUM",
]
type PfCurrentBalance = Literal[0, 1]
type PfVCartesian = Literal[0, 1, 2]
type PfEnforceQLims = Literal[0, 1, 2]
type CpfParameterization = Literal[1, 2, 3]
type CpfStopAt = Literal["NOSE", "FULL"] | float
type CpfBinaryFlag = Literal[0, 1]
type CpfPlotLevel = Literal[0, 1, 2, 3]
type OpfAcSolver = Literal[
    "DEFAULT",
    "MIPS",
    "FMINCON",
    "IPOPT",
    "KNITRO",
    "MINOPF",
    "PDIPM",
    "SDPOPF",
    "TRALM",
]
type OpfDcSolver = Literal[
    "DEFAULT",
    "MIPS",
    "BPMPD",
    "CLP",
    "CPLEX",
    "GLPK",
    "GUROBI",
    "IPOPT",
    "MOSEK",
    "OSQP",
    "OT",
]
type OpfCurrentBalance = Literal[0, 1]
type OpfVCartesian = Literal[0, 1]
type OpfFlowLim = Literal["S", "P", "2", "I"]
type OpfIgnoreAngleLim = Literal[0, 1]
type OpfSoftlimsDefault = Literal[0, 1]
type OpfInitFromMpc = Literal[-1, 0, 1]
type OpfStart = Literal[0, 1, 2, 3]
type OpfReturnRawDer = Literal[0, 1]
type OpfBackend = Literal["MATPOWER", "POWER_MODELS"]
type PowerModelsFormulation = Literal[
    "ACP",
    "ACR",
    "ACT",
    "SOCWR",
    "DCP",
    "DCMP",
    "NFA",
    "DCPLL",
    "LPACC",
    "BFA",
    "SOCBF",
    "IVR",
    "QCRM",
    "QCLS",
    "SOCWRCONIC",
    "SOCBFCONIC",
    "SDPWRM",
    "SPARSESDPWRM",
]
type PowerModelsSolver = Literal["DEFAULT", "HIGHS", "IPOPT", "GLPK", "CLARABEL", "SCS", "MOSEK"]
type VerboseLevel = Literal[0, 1, 2, 3]
type OutAll = Literal[-1, 0, 1]
type OutLimAll = Literal[-1, 0, 1, 2]
type OutLimDetail = Literal[0, 1, 2]
type OutBinaryOrAuto = Literal[-1, 0, 1]
type MipsLinsolver = Literal[
    "",
    "\\",
    "LU",
    "LU3",
    "LU3a",
    "LU3m",
    "LU3am",
    "LU4",
    "LU4m",
    "LU5",
    "LU5m",
    "SUPERLU",
    "UMFPACK",
    "PARDISO",
    "KLU",
]
type CplexLpMethod = Literal[0, 1, 2, 3, 4, 5, 6]
type CplexQpMethod = Literal[0, 1, 2, 3, 4]
type FminconAlg = Literal[1, 2, 3, 4, 5, 6]
type GurobiMethod = Literal[-1, 0, 1, 2, 3, 4]
type MosekLpAlg = Literal[0, 1, 2, 3, 4, 6]


@dataclass
class PfNrConfig(DataclassDictMixin):
    max_it: int = 10
    lin_solver: str = ""


@dataclass
class PfFdConfig(DataclassDictMixin):
    max_it: int = 30


@dataclass
class PfGsConfig(DataclassDictMixin):
    max_it: int = 1000


@dataclass
class PfRadialConfig(DataclassDictMixin):
    max_it: int = 20
    vcorr: Literal[0, 1] = 0


@dataclass
class PfConfig(DataclassDictMixin):
    """Power flow options.

    Attributes:
        alg (PfAlg): AC power flow algorithm. Default is ``"NR"``.
            - ``"NR"``: Newton's method (formulation depends on ``current_balance`` and ``v_cartesian``)
            - ``"NR-SP"``: Newton's method (power mismatch, polar)
            - ``"NR-SC"``: Newton's method (power mismatch, cartesian)
            - ``"NR-SH"``: Newton's method (power mismatch, hybrid)
            - ``"NR-IP"``: Newton's method (current mismatch, polar)
            - ``"NR-IC"``: Newton's method (current mismatch, cartesian)
            - ``"NR-IH"``: Newton's method (current mismatch, hybrid)
            - ``"FDXB"``: Fast-Decoupled (XB version)
            - ``"FDBX"``: Fast-Decoupled (BX version)
            - ``"GS"``: Gauss-Seidel
            - ``"PQSUM"``: Power Summation method (radial networks only)
            - ``"ISUM"``: Current Summation method (radial networks only)
            - ``"YSUM"``: Admittance Summation method (radial networks only)
        current_balance (PfCurrentBalance): Type of nodal balance equation. Default is ``0``.
            - ``0``: use complex power balance equations
            - ``1``: use complex current balance equations
        v_cartesian (PfVCartesian): Voltage representation. Default is ``0``.
            - ``0``: bus voltage variables represented in polar coordinates
            - ``1``: bus voltage variables represented in cartesian coordinates
            - ``2``: hybrid, polar updates computed via modified cartesian Jacobian
        tol (float): Termination tolerance on per-unit P & Q mismatch. Default is ``1e-8``.
        nr (PfNrConfig): Newton method options.
            - ``max_it`` (int): maximum number of iterations for Newton's method (default ``10``)
            - ``lin_solver`` (str): linear solver passed to MPLINSOLVE for Newton update step (default ``""``)
            - ``""``: default to ``"\"`` for small systems, ``"LU3"`` for larger ones
            - ``"\"``: built-in backslash operator
            - ``"LU"``: explicit default LU decomposition and back substitution
            - ``"LU3"``: 3-output LU, Gilbert-Peierls with AMD reordering
            - ``"LU4"``: 4-output LU, UMFPACK solver (same as ``"LU"``)
            - ``"LU5"``: 5-output LU, UMFPACK solver with row scaling
        fd (PfFdConfig): Fast-decoupled method options.
            - ``max_it`` (int): maximum number of iterations (default ``30``)
        gs (PfGsConfig): Gauss-Seidel method options.
            - ``max_it`` (int): maximum number of iterations (default ``1000``)
        radial (PfRadialConfig): Radial power flow method options.
            - ``max_it`` (int): maximum number of iterations (default ``20``)
            - ``vcorr`` (Literal[0, 1]): perform voltage correction procedure (default ``0``)
            - ``0``: do not perform voltage correction
            - ``1``: perform voltage correction
        enforce_q_lims (PfEnforceQLims): Enforce generator reactive power limits at expense of ``|V|``. Default is ``0``.
            - ``0``: do not enforce limits
            - ``1``: enforce limits with simultaneous bus type conversion
            - ``2``: enforce limits with one-at-a-time bus type conversion
    """

    alg: PfAlg = "NR"
    current_balance: PfCurrentBalance = 0
    v_cartesian: PfVCartesian = 0
    tol: float = 1e-8
    nr: PfNrConfig = field(default_factory=PfNrConfig)
    fd: PfFdConfig = field(default_factory=PfFdConfig)
    gs: PfGsConfig = field(default_factory=PfGsConfig)
    radial: PfRadialConfig = field(default_factory=PfRadialConfig)
    enforce_q_lims: PfEnforceQLims = 0


@dataclass
class CpfPlotConfig(DataclassDictMixin):
    level: CpfPlotLevel = 0
    bus: list[int] = field(default_factory=list)


@dataclass
class CpfConfig(DataclassDictMixin):
    """Continuation power flow options.

    Attributes:
    parameterization (CpfParameterization): Choice of parameterization. Default is ``3``.
        - ``1``: natural
        - ``2``: arc length
        - ``3``: pseudo arc length
    stop_at (CpfStopAt): Determines stopping criterion. Default is ``"NOSE"``.
        - ``"NOSE"``: stop when nose point is reached
        - ``"FULL"``: trace full nose curve
        - ``float``: stop upon reaching specified target lambda value
    enforce_p_lims (CpfBinaryFlag): Enforce generator active power limits. Default is ``0``.
        - ``0``: do not enforce limits
        - ``1``: enforce limits, simultaneous bus type conversion
    enforce_q_lims (CpfBinaryFlag): Enforce generator reactive power limits at expense of ``|V|``. Default is ``0``.
        - ``0``: do not enforce limits
        - ``1``: enforce limits, simultaneous bus type conversion
    enforce_v_lims (CpfBinaryFlag): Enforce bus voltage magnitude limits. Default is ``0``.
        - ``0``: do not enforce limits
        - ``1``: enforce limits, termination on detection
    enforce_flow_lims (CpfBinaryFlag): Enforce branch flow MVA limits. Default is ``0``.
        - ``0``: do not enforce limits
        - ``1``: enforce limits, termination on detection
    step (float): Continuation power flow step size. Default is ``0.05``.
    adapt_step (CpfBinaryFlag): Toggle adaptive step size feature. Default is ``0``.
        - ``0``: adaptive step size disabled
        - ``1``: adaptive step size enabled
    step_min (float): Minimum allowed step size. Default is ``1e-4``.
    step_max (float): Maximum allowed step size. Default is ``0.2``.
    adapt_step_damping (float): Damping factor for adaptive step sizing. Default is ``0.7``.
    adapt_step_tol (float): Tolerance for adaptive step sizing. Default is ``1e-3``.
    target_lam_tol (float): Tolerance for target lambda detection. Default is ``1e-5``.
    nose_tol (float): Tolerance for nose point detection in p.u. Default is ``1e-5``.
    p_lims_tol (float): Tolerance for generator active power limit enforcement in MW. Default is ``0.01``.
    q_lims_tol (float): Tolerance for generator reactive power limit enforcement in MVAR. Default is ``0.01``.
    v_lims_tol (float): Tolerance for bus voltage magnitude enforcement in p.u. Default is ``1e-4``.
    flow_lims_tol (float): Tolerance for line MVA flow enforcement in MVA. Default is ``0.01``.
    plot (CpfPlotConfig): Nose curve plotting options.
        - ``level`` (CpfPlotLevel): control plotting of nose curve (default ``0``)
        - ``0``: do not plot nose curve
        - ``1``: plot when completed
        - ``2``: plot incrementally at each iteration
        - ``3``: same as ``2``, with pause at each iteration
        - ``bus`` (list[int]): index of bus whose voltage is to be plotted (default empty)
    user_callback (object | None): User callback specification. Default is ``None``.
        May be a function name string, a struct-like object with function name and
        optional priority and/or args, or a list of such entries.
    """

    parameterization: CpfParameterization = 3
    stop_at: CpfStopAt = "NOSE"
    enforce_p_lims: CpfBinaryFlag = 0
    enforce_q_lims: CpfBinaryFlag = 0
    enforce_v_lims: CpfBinaryFlag = 0
    enforce_flow_lims: CpfBinaryFlag = 0
    step: float = 0.05
    adapt_step: CpfBinaryFlag = 0
    step_min: float = 1e-4
    step_max: float = 0.2
    adapt_step_damping: float = 0.7
    adapt_step_tol: float = 1e-3
    target_lam_tol: float = 1e-5
    nose_tol: float = 1e-5
    p_lims_tol: float = 0.01
    q_lims_tol: float = 0.01
    v_lims_tol: float = 1e-4
    flow_lims_tol: float = 0.01
    plot: CpfPlotConfig = field(default_factory=CpfPlotConfig)
    user_callback: object | None = None


@dataclass
class OpfAcConfig(DataclassDictMixin):
    solver: OpfAcSolver = "DEFAULT"


@dataclass
class OpfDcConfig(DataclassDictMixin):
    solver: OpfDcSolver = "DEFAULT"


@dataclass
class OpfSoftlimsConfig(DataclassDictMixin):
    default: OpfSoftlimsDefault = 1


@dataclass
class OpfPowerModelsConfig(DataclassDictMixin):
    formulation: PowerModelsFormulation = "ACP"
    solver: PowerModelsSolver = "DEFAULT"
    extensions: tuple[object, ...] = ()
    highs_options: dict[str, object] = field(default_factory=dict)
    glpk_options: dict[str, object] = field(default_factory=dict)
    clarabel_options: dict[str, object] = field(default_factory=dict)
    scs_options: dict[str, object] = field(default_factory=dict)
    mosek_options: dict[str, object] = field(default_factory=dict)


@dataclass
class OpfConfig(DataclassDictMixin):
    """Optimal power flow options.

    Attributes:
    backend (OpfBackend): OPF implementation. Default is ``"MATPOWER"``.
        - ``"MATPOWER"``: use the MATPOWER-compatible OPF path
        - ``"POWER_MODELS"``: use the PowerModels-compatible formulation path
    power_models (OpfPowerModelsConfig): PowerModels backend configuration.
        - ``formulation``: network formulation, ``"ACP"``, ``"ACR"``, ``"ACT"``, ``"SOCWR"``,
          ``"DCP"``, ``"DCMP"``, ``"NFA"``, ``"DCPLL"``, ``"LPACC"``, ``"BFA"``,
          ``"SOCBF"``, ``"IVR"``, ``"QCRM"``, ``"QCLS"``, ``"SOCWRCONIC"``,
          ``"SOCBFCONIC"``, ``"SDPWRM"``, or ``"SPARSESDPWRM"``
        - ``solver``: ``"DEFAULT"`` selects by model capability, preferring
          HiGHS for LP/QP models, GLPK for LP fallback, and Ipopt for nonlinear
          models. Explicit ``"HIGHS"``, ``"GLPK"``, and ``"IPOPT"`` selections
          do not fall back. Default is ``"DEFAULT"``.
        - ``extensions``: callable model extensions applied after formulation
          construction and before solver selection. Default is empty.
        - ``highs_options`` and ``glpk_options``: native solver option mappings.
    ac (OpfAcConfig): AC OPF solver options.
        - ``solver`` (OpfAcSolver): AC optimal power flow solver. Default is ``"DEFAULT"``.
        - ``"DEFAULT"``: choose pwrs default AC solver, currently ``"MIPS"``
        - ``"MIPS"``: Pwrs Interior Point Solver
        - ``"FMINCON"``: MATLAB Optimization Toolbox ``fmincon``
        - ``"IPOPT"``: IPOPT interface
        - ``"KNITRO"``: Artelys Knitro interface
        - ``"MINOPF"``: MINOS-based MINOPF package
        - ``"PDIPM"``: TSPOPF primal/dual interior point solver
        - ``"SDPOPF"``: semidefinite-relaxation OPF solver
        - ``"TRALM"``: trust-region augmented Lagrangian method
    dc (OpfDcConfig): DC OPF solver options.
        - ``solver`` (OpfDcSolver): DC optimal power flow solver. Default is ``"DEFAULT"``.
        - ``"DEFAULT"``: choose based on availability, preferring commercial/open-source LP/QP solvers before ``"MIPS"``
        - ``"MIPS"``: Pwrs Interior Point Solver
        - ``"BPMPD"``: BPMPD interface
        - ``"CLP"``: COIN-OR LP solver
        - ``"CPLEX"``: IBM CPLEX
        - ``"GLPK"``: GNU Linear Programming Kit
        - ``"GUROBI"``: Gurobi optimizer
        - ``"IPOPT"``: IPOPT interface
        - ``"MOSEK"``: MOSEK interface
        - ``"OSQP"``: OSQP interface
        - ``"OT"``: MATLAB Optimization Toolbox ``quadprog``/``linprog``
    current_balance (OpfCurrentBalance): Type of nodal balance equations. Default is ``0``.
        - ``0``: use complex power balance equations
        - ``1``: use complex current balance equations
    v_cartesian (OpfVCartesian): Voltage representation. Default is ``0``.
        - ``0``: bus voltage variables represented in polar coordinates
        - ``1``: bus voltage variables represented in cartesian coordinates
    violation (float): Constraint violation tolerance. Default is ``5e-6``.
    use_vg (float): How generator voltage setpoints affect bus voltage bounds. Default is ``0``.
        - ``0``: use bus ``Vmin`` and ``Vmax`` only, ignore generator ``Vg``
        - ``1``: replace bus ``Vmin`` and ``Vmax`` with corresponding generator ``Vg``
        - values between ``0`` and ``1``: use a weighted blend of the two behaviors
    flow_lim (OpfFlowLim): Quantity constrained by branch flow limits. Default is ``"S"``.
        - ``"S"``: apparent power flow, in MVA
        - ``"P"``: active power flow, in MW
        - ``"2"``: squared active power flow, corresponding to MW limits
        - ``"I"``: current magnitude, in MVA at 1 p.u. voltage
    ignore_angle_lim (OpfIgnoreAngleLim): Whether to ignore branch angle difference limits. Default is ``0``.
        - ``0``: include angle limits when specified
        - ``1``: ignore angle limits even if present
    softlims (OpfSoftlimsConfig): Soft limit behavior.
        - ``default`` (OpfSoftlimsDefault): handling for soft limits without explicit settings. Default is ``1``.
        - ``0``: do not include unspecified soft limits
        - ``1``: include unspecified soft limits using default values
    init_from_mpc (OpfInitFromMpc): Deprecated; use ``start`` instead. Default is ``-1``.
        Controls whether the current MATPOWER case state initializes the OPF.
        - ``-1``: let MATPOWER decide based on solver/algorithm
        - ``0``: do not use current case state
        - ``1``: use current case state
    start (OpfStart): OPF initialization strategy. Default is ``0``.
        - ``0``: default; MATPOWER chooses based on solver
        - ``1``: ignore current case state and use solver-specific interior initialization
        - ``2``: use current MATPOWER case state
        - ``3``: solve a power flow first and use that solution
    return_raw_der (OpfReturnRawDer): For AC OPF, include raw constraint and derivative data in ``results.raw``. Default is ``0``.
        - ``0``: do not return raw derivative info
        - ``1``: return ``g``, ``dg``, ``df`` and ``d2f`` in ``results.raw``
    """

    backend: OpfBackend = "MATPOWER"
    power_models: OpfPowerModelsConfig = field(default_factory=OpfPowerModelsConfig)
    ac: OpfAcConfig = field(default_factory=OpfAcConfig)
    dc: OpfDcConfig = field(default_factory=OpfDcConfig)
    current_balance: OpfCurrentBalance = 0
    v_cartesian: OpfVCartesian = 0
    violation: float = 5e-6
    use_vg: float = 0
    flow_lim: OpfFlowLim = "S"
    ignore_angle_lim: OpfIgnoreAngleLim = 0
    softlims: OpfSoftlimsConfig = field(default_factory=OpfSoftlimsConfig)
    init_from_mpc: OpfInitFromMpc = -1
    start: OpfStart = 0
    return_raw_der: OpfReturnRawDer = 0


@dataclass
class OutLimConfig(DataclassDictMixin):
    all: OutLimAll = -1
    v: OutLimDetail = 1
    line: OutLimDetail = 1
    pg: OutLimDetail = 1
    qg: OutLimDetail = 1


@dataclass
class OutConfig(DataclassDictMixin):
    """Output options.

    Attributes:
    all: Controls pretty-printing of results. Default is ``-1``.
        Options:
            - ``-1``: Individual flags control what prints.
            - ``0``: Do not print anything. Overrides individual flags,
              except for output written to files specified explicitly.
            - ``1``: Print everything. Overrides individual flags.
    sys_sum: Print system summary. Default is ``1``.
        Options:
            - ``0``: Do not print.
            - ``1``: Print.
    area_sum: Print area summaries. Default is ``0``.
        Options:
            - ``0``: Do not print.
            - ``1``: Print.
    bus: Print bus detail. Default is ``1``.
        Options:
            - ``0``: Do not print.
            - ``1``: Print.
    branch: Print branch detail. Default is ``1``.
        Options:
            - ``0``: Do not print.
            - ``1``: Print.
    gen: Print generator detail. Default is ``0``.
        Options:
            - ``0``: Do not print.
            - ``1``: Print.
    lim: Constraint/limit output configuration. Default is an ``OutLimConfig``.
        Includes options for:
            - ``all``: Overall control for constraint info output.
                - ``-1``: Individual limit flags control what prints.
                - ``0``: Do not print constraint info.
                - ``1``: Print binding constraint info only.
                - ``2``: Print all constraint info.
            - ``v``: Voltage limit info.
                - ``0``: Do not print.
                - ``1``: Print binding constraints only.
                - ``2``: Print all constraints.
            - ``line``: Line flow limit info, using the same options as ``v``.
            - ``pg``: Generator active power limit info, using the same options as ``v``.
            - ``qg``: Generator reactive power limit info, using the same options as ``v``.
    force: Print results even if the success flag is ``0``. Default is ``0``.
        Options:
            - ``0``: Do not force printing.
            - ``1``: Force printing.
    suppress_detail: Suppress all output except the system summary. Default is ``-1``.
        Options:
            - ``-1``: Suppress details automatically for large systems
              (more than 500 buses).
            - ``0``: Do not suppress any output enabled by other flags.
            - ``1``: Suppress all output except the system summary.
              Overrides individual flags, but not ``all = 1``.
    """

    all: OutAll = -1
    sys_sum: Literal[0, 1] = 1
    area_sum: Literal[0, 1] = 0
    bus: Literal[0, 1] = 1
    branch: Literal[0, 1] = 1
    gen: Literal[0, 1] = 0
    lim: OutLimConfig = field(default_factory=OutLimConfig)
    force: Literal[0, 1] = 0
    suppress_detail: OutBinaryOrAuto = -1


@dataclass
class MipsScConfig(DataclassDictMixin):
    red_it: int = 20


@dataclass
class MipsConfig(DataclassDictMixin):
    """MIPS solver options.

    Attributes:
        step_control (Literal[0, 1]): Enable step-size control. Default is ``0``.
            - ``0``: disabled
            - ``1``: enabled
        linsolver (MipsLinsolver): Linear system solver. Default is ``""``.
            - ``""`` or ``"\\"``: default SciPy solve
            - ``"LU3"`` variants or ``"SUPERLU"``: SciPy SuperLU
            - ``"LU"``, ``"LU4"``/``"LU5"`` variants or ``"UMFPACK"``: SuiteSparse UMFPACK
            - ``"PARDISO"``: Intel MKL PARDISO via PyPardiso
            - ``"KLU"``: SuiteSparse KLU via nbklu
            Missing optional solvers emit a warning and fall back to SciPy SuperLU.
        feastol (float): Feasibility (equality) tolerance. Default is ``0``.
            If ``0``, it is set from ``opf.violation``.
        gradtol (float): Gradient tolerance. Default is ``1e-6``.
        comptol (float): Complementarity (inequality) tolerance. Default is ``1e-6``.
        costtol (float): Optimality tolerance. Default is ``1e-6``.
        max_it (int): Maximum number of iterations. Default is ``150``.
        sc (MipsScConfig): Step-control options.
            - ``red_it`` (int): maximum number of reductions per iteration
            when step control is enabled (default ``20``)
        xi (float | None): Constant used in alpha updates. Default is ``None``
        sigma (float | None): Centering parameter. Default is ``None``
        z0 (float | None): Initial slack variable value. Default is ``None``
        alpha_min (float | None): Numerical-failure threshold for alpha
            parameters. Default is ``None``
        rho_min (float | None): Lower bound on ``rho_t``. Default is ``None``
        rho_max (float | None): Upper bound on ``rho_t``. Default is ``None``
        mu_threshold (float | None): KT multipliers below this value for
            non-binding constraints are forced to zero. Default is ``None``
        max_stepsize (float | None): Numerical-failure threshold for the
            2-norm of the reduced Newton step. Default is ``None``
    """

    step_control: Literal[0, 1] = 0
    feastol: float = 0
    gradtol: float = 1e-6
    comptol: float = 1e-6
    costtol: float = 1e-6
    max_it: int = 150
    sc: MipsScConfig = field(default_factory=MipsScConfig)
    # below are from mips.m
    verbose: int | None = None
    linsolver: MipsLinsolver | None = None
    cost_mult: float | None = None
    xi: float | None = None
    sigma: float | None = None
    z0: float | None = None
    alpha_min: float | None = None
    rho_min: float | None = None
    rho_max: float | None = None
    mu_threshold: float | None = None
    max_stepsize: float | None = None


@dataclass
class IpoptConfig(DataclassDictMixin):
    opts: dict[str, object] = field(default_factory=dict)
    opt_fname: str = ""
    opt: int = 0


@dataclass
class FminconConfig(DataclassDictMixin):
    alg: FminconAlg = 4
    tol_x: float = 1e-4
    tol_f: float = 1e-4
    max_it: int = 0
    opts: dict[str, object] = field(default_factory=dict)


@dataclass
class KnitroConfig(DataclassDictMixin):
    tol_x: float = 1e-4
    tol_f: float = 1e-4
    maxit: int = 0
    opts: dict[str, object] = field(default_factory=dict)
    opt_fname: str = ""
    opt: int = 0


@dataclass
class SysWideZipLoadsConfig(DataclassDictMixin):
    pw: tuple[float, float, float] | None = None
    qw: tuple[float, float, float] | None = None


@dataclass
class ExpConfig(DataclassDictMixin):
    sys_wide_zip_loads: SysWideZipLoadsConfig = field(default_factory=SysWideZipLoadsConfig)


@dataclass
class MatpowerConfig(DataclassDictMixin):
    """MATPOWER options struct.

    Attributes:
        v (int): version number of MATPOWER options struct
        verbose (VerboseLevel):
        model (ModelType): AC vs. DC power flow model. Default is ``AC``.
            - ``AC``: use nonlinear AC model & corresponding algorithms/options
            - ``DC``: use linear DC model & corresponding algorithms/options
        pf (PfConfig): power flow options
        cpf (CpfConfig): continuation power flow options
        opf (OpfConfig): optimal power flow options
        out (OutConfig): output options controlling what results are returned
        mips (MipsConfig): MIPS-specific options
        exp (ExpConfig): experimental features and options
        fmincon (FminconConfig): options for MATLAB Optimization Toolbox ``fmincon``
        ipopt (IpoptConfig): options for IPOPT solver, used when 'opf.ac.solver' or 'opf.dc.solver' is set to 'IPOPT'
        knitro (KnitroConfig): options for the Artelys Knitro solver
    """

    v: int = 21
    model: ModelType = "AC"
    pf: PfConfig = field(default_factory=PfConfig)
    cpf: CpfConfig = field(default_factory=CpfConfig)
    opf: OpfConfig = field(default_factory=OpfConfig)
    verbose: VerboseLevel = 1
    out: OutConfig = field(default_factory=OutConfig)
    mips: MipsConfig = field(default_factory=MipsConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)

    fmincon: FminconConfig = field(default_factory=FminconConfig)
    ipopt: IpoptConfig | None = field(default_factory=IpoptConfig)
    knitro: KnitroConfig = field(default_factory=KnitroConfig)


def fetch_mpoption[T](opt: dict | MatpowerConfig, as_type: type[T], name: str) -> T | None:
    if isinstance(opt, MatpowerConfig):
        opt = opt.to_dict()
    layers = name.split(".")
    for layer in layers:
        if layer in opt:
            opt = opt[layer]
        else:
            return None
    if not isinstance(opt, as_type):
        warnings.warn(
            f"fetch_mpoption: expected {name} to be of type {as_type.__name__}, got {type(opt).__name__}",
            UserWarning,
            stacklevel=2,
        )
    converter = cast(Callable[[object], T], as_type)
    return converter(opt)


def get_zip_weights(mpopt: MatpowerConfig) -> tuple[np.ndarray, np.ndarray]:
    pw = mpopt.exp.sys_wide_zip_loads.pw
    qw = mpopt.exp.sys_wide_zip_loads.qw
    if pw is None:
        pw = np.array([1.0, 0.0, 0.0])
    else:
        pw = np.asarray(pw, dtype=float)
    if qw is None:
        qw = np.array(pw, copy=True)
    else:
        qw = np.asarray(qw, dtype=float)
    return pw, qw


def _merge_option_dict(target: DataclassDictMixin, values: dict[str, object]) -> None:
    """Merge a legacy nested options mapping into a typed config object."""
    for key, value in values.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(current, DataclassDictMixin) and isinstance(value, dict):
            _merge_option_dict(current, value)
        else:
            setattr(target, key, value)


def mpoption(*args: Any) -> MatpowerConfig:
    """Create or modify a MATPOWER options dict.

    This function preserves the MATPOWER ``mpoption`` interface, supporting default construction, overrides by name/value pairs, merging from an existing options dict, and conversion to or from the legacy numeric options vector form.

    However, for type safety, we recommend using the native Python interface by constructing and modifying a ``MatpowerConfig`` dataclass directly. The legacy interface is still supported for compatibility, but it is more error-prone and less efficient than using the dataclass API.

    Parameters
    ----------
    *args
        MATPOWER option inputs.
        To create a default options dict, call with no arguments: ``mpoption()``.
        To modify an existing options dict, use name/value pairs: ``mpoption(existing_opt, 'name', value, ...)``.
        The first argument may also be an existing options dict to merge with, or an old-style numeric options vector to convert from.

    Returns
    -------
    MatpowerConfig
        The resulting MATPOWER options struct.
    """
    if len(args) == 0:
        return MatpowerConfig()
    if len(args) % 2 == 1:
        source = args[0]
        if isinstance(source, MatpowerConfig):
            opt = deepcopy(source)
        elif isinstance(source, dict):
            opt = MatpowerConfig()
            _merge_option_dict(opt, source)
        else:
            raise TypeError("mpoption: expected MatpowerConfig or nested options dict")
        for i in range(1, len(args), 2):
            k, v = args[i], args[i + 1]
            parts = k.split(".")
            target = opt
            for part in parts[:-1]:
                if isinstance(target, dict):
                    target = target.setdefault(part, {})
                else:
                    target = getattr(target, part)
            if isinstance(target, dict):
                target[parts[-1]] = v
            else:
                setattr(target, parts[-1], v)
        return opt
    if len(args) % 2 == 0:
        opt = MatpowerConfig()
        return mpoption(opt, *args)
    raise ValueError("mpoption: invalid argument list")
