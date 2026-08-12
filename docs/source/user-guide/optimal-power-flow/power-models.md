# PowerModels-compatible OPF

The `POWER_MODELS` backend provides PowerModels-style network formulations
through a Python-native modeling stack. It is useful for alternative AC/DC
formulations, convex relaxations, reusable domain extensions, and large OPF
problems that benefit from external optimization solvers.

Despite the name, this backend does not run Julia or call PowerModels.jl.
Its formulation names and modeling concepts are aligned with PowerModels.jl,
while `pwrs` builds and solves the models using Pyomo or CVXPY.

## Install optional dependencies

The base `pwrs` installation is sufficient for the default MATPOWER/MIPS
backend. The PowerModels-compatible backend loads its modeling frameworks
lazily and requires packages appropriate to the selected formulation and
solver.

When working from the source repository, install the complete optional stack
with:

```bash
uv sync --group all
```

The main combinations are:

| Workload | Modeling layer | Solver packages |
| --- | --- | --- |
| Nonlinear AC formulations | Pyomo | CyIpopt and Ipopt libraries |
| LP/QP formulations | Pyomo | HiGHS (`highspy`), GLPK, or CyIpopt |
| SOCP formulations | CVXPY | Clarabel, SCS, or MOSEK |
| SDP formulations | CVXPY | SCS or MOSEK, depending on problem and installation |

CyIpopt also requires an Ipopt installation and Pyomo's PyNumero ASL
interface. MOSEK requires a separately supplied license. See
[Setup with external optimizers](../getting-started.md#setup-with-external-optimizers)
for Ipopt and optional HSL setup.

## Run the default ACP formulation

Select the backend explicitly. Its default formulation is `ACP`, and its
default solver selection resolves a nonlinear ACP model to Ipopt.

```python
import pwrs as mp

mpopt = mp.MatpowerConfig()
mpopt.opf.backend = "POWER_MODELS"
mpopt.opf.power_models.formulation = "ACP"
mpopt.opf.power_models.solver = "DEFAULT"
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.runopf(mp.case30(), mpopt)
if not result.success:
    raise RuntimeError("PowerModels-compatible ACP did not converge")

print(result.f)
print(result.bus)
print(result.gen)
```

`ACP` is the exact nonlinear AC polar formulation and is the normal starting
point when the result must be AC feasible. The backend maps its native model
solution back to the same MATPOWER bus, generator, and branch result matrices
as the default backend.

## Choose a formulation

Set `opf.power_models.formulation` to one of the following names. A relaxation
or approximation solves a different mathematical problem from ACP; a lower
objective value does not mean it found a better AC-feasible dispatch.

| Family | Formulations | Purpose |
| --- | --- | --- |
| Exact nonlinear AC | `ACP`, `ACR`, `ACT`, `IVR` | AC OPF with polar, rectangular, lifted tangent, or current-voltage representations. |
| Active-power/DC | `DCP`, `DCMP`, `NFA`, `DCPLL` | DC, modified DC, network-flow, and DC-with-loss approximations. |
| AC approximations and relaxations | `LPACC`, `BFA`, `SOCBF`, `SOCWR`, `QCRM`, `QCLS` | Linearized, branch-flow, second-order-cone, and quadratic-convex models built with Pyomo. |
| Explicit conic relaxations | `SOCWRCONIC`, `SOCBFCONIC`, `SDPWRM`, `SPARSESDPWRM` | SOCP and SDP models built with CVXPY for conic solvers. |

The active-power formulations `DCP`, `DCMP`, `NFA`, and `DCPLL` require the
DC model. `rundcopf()` supplies that setting:

```python
mpopt = mp.MatpowerConfig()
mpopt.opf.backend = "POWER_MODELS"
mpopt.opf.power_models.formulation = "DCP"
mpopt.opf.power_models.solver = "DEFAULT"
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.rundcopf(mp.case30(), mpopt)
```

The other formulations require the AC model and should be called with
`runopf()`. All PowerModels-compatible AC formulations currently enforce
apparent-power branch ratings, so `opf.flow_lim` must remain `"S"`.

When choosing among formulations:

- use `ACP` first for an AC-feasible operating point;
- use `ACR` or `IVR` for formulation studies or different numerical behavior;
- use `DCP` for a conventional active-power approximation;
- use `NFA` for a simpler network-flow relaxation;
- use the SOC, QC, or SDP formulations to compute relaxation bounds or study
  convex models, not as interchangeable replacements for an AC solution.

## Automatic solver selection

With `opf.power_models.solver = "DEFAULT"`, the backend classifies the built
model after applying extensions and chooses an installed compatible solver:

| Problem class | Default candidates |
| --- | --- |
| Linear Pyomo model | HiGHS, then GLPK, then Ipopt |
| Convex separable QP Pyomo model | HiGHS, then Ipopt |
| Nonlinear Pyomo model | Ipopt |
| CVXPY SOCP or SDP model | Clarabel, then SCS |

An explicit solver request never silently falls back. For example:

```python
mpopt.opf.power_models.solver = "HIGHS"
mpopt.opf.power_models.highs_options = {
    "primal_feasibility_tolerance": 1e-7,
}
```

Supported explicit names are `HIGHS`, `GLPK`, and `IPOPT` for Pyomo models,
and `CLARABEL`, `SCS`, and `MOSEK` for CVXPY conic models. The backend rejects
a solver that cannot handle the classified problem. For example, HiGHS cannot
solve a nonlinear ACP model.

Solver-specific native options are stored in:

- `opf.power_models.highs_options` and `glpk_options`;
- `ipopt.opts`;
- `opf.power_models.clarabel_options`, `scs_options`, and `mosek_options`.

## Configure Ipopt for large AC OPF

For medium and large ACP studies, an efficient sparse linear solver can have
a much larger effect than small Python-level tuning changes:

```python
import pwrs as mp
from pwrs.data.pglibopf import pglib_opf_case1354_pegase

mpopt = mp.MatpowerConfig()
mpopt.opf.backend = "POWER_MODELS"
mpopt.opf.power_models.formulation = "ACP"
mpopt.opf.power_models.solver = "IPOPT"
mpopt.ipopt.opts.update(
    {
        "linear_solver": "ma27",
        "hsllib": "libcoinhsl.so",  # use the installed platform-specific path
        "print_level": 0,
    }
)
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.runopf(pglib_opf_case1354_pegase(), mpopt)
if not result.success:
    raise RuntimeError("Ipopt failed to solve ACP")
```

MA27 and other HSL solvers require a separately obtained Coin-HSL library.
Do not copy the example library name blindly: use the path and filename from
your installation, and confirm Ipopt reports the requested linear solver.

For performance comparisons, measure a warm and a cold end-to-end
`runopf()` call. `result.et` includes case preparation, model construction,
solver setup, the native solve, and result reconstruction. The finer timing
breakdown in `result.raw` helps distinguish solver time from modeling overhead.

## Inspect solver metadata

The backend records how a problem was classified and solved under
`result.raw["output"]`:

```python
if result.raw is not None:
    output = result.raw["output"]
    solver = output["solver"]

    print(output["formulation"])
    print(output["implementation"])
    print(solver["requested"], solver["selected"])
    print(solver["problem_class"])
    print(solver["fallbacks"])
    print(output["max_constraint_violation"])
    print(output["timings"])
```

Important fields include:

- `formulation` and `implementation` (`PYOMO` or `CVXPY`);
- requested and selected solver names;
- base and final problem class, including extension-driven reclassification;
- unavailable-solver fallbacks made during default selection;
- termination status and maximum constraint violation;
- network-build, model-build, solver, and result-mapping timings.

These fields are backend diagnostics rather than part of the stable structured
result API. Code that must work with both OPF backends should depend on
`result.success`, `result.f`, and the solved matrices instead.

## Add reusable model extensions

Extensions run after the base formulation is built and before the model is
classified for solver selection. An extension callback receives the public
`PowerModel` protocol and returns `None`. The protocol exposes semantic
quantities and named registration methods without requiring application code
to depend on an internal context class.

Prefer the public domain helpers when they express the required operation.
They work with both Pyomo and CVXPY formulations and preserve MATPOWER units
and external component indexes.

```python
import pwrs as mp
from pwrs.power_models import PowerModel, constraint_active_generation_sum


def add_reserve_floor(problem: PowerModel) -> None:
    constraint_active_generation_sum(
        problem,
        "reserve_floor",
        generator_rows=[0, 1],
        minimum_mw=80.0,
    )


mpopt = mp.MatpowerConfig()
mpopt.opf.backend = "POWER_MODELS"
mpopt.opf.power_models.formulation = "ACP"
mpopt.opf.power_models.extensions = (add_reserve_floor,)
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.runopf(mp.case30(), mpopt)
```

Generator and branch selectors passed to the domain helpers are zero-based
rows in the original MATPOWER matrices; bus selectors use `BUS_I`. Power
quantities use MW/MVAr/MVA, angles use degrees, and voltage magnitudes use per
unit. Available helpers cover generator active/reactive bounds and setpoints,
voltage-magnitude bounds and setpoints, branch active/reactive/apparent-power
limits, voltage-angle differences, weighted interface active flow, and
aggregate active generation.

Callback names are used as extension result keys. Wrap a callback in
`PowerModelExtension(name, callback)` when it needs a stable name different
from its Python function name. Extension names and all variable, constraint,
and objective-term group names must be unique identifiers.

### Inspect capabilities before adding constraints

Formulations do not all expose the same physical quantities. For example,
active-power-only formulations have no reactive generation, and NFA has no
voltage-angle quantity. Reusable callbacks should query capabilities instead
of assuming that a backend variable exists:

```python
from pwrs.power_models import PowerModel


def add_reactive_floor(problem: PowerModel) -> None:
    if not problem.has_quantity("qg"):
        raise NotImplementedError(
            f"{problem.formulation} does not expose reactive generation"
        )

    problem.add_bounded_constraint(
        "first_generator_reactive_floor",
        problem.quantity("qg", 0),
        lower=-0.5,  # internal per-unit value
    )
```

The stable protocol surface consists of:

- `network`, `formulation`, and `backend` (`"pyomo"` or `"cvxpy"`);
- `available_quantities`, `has_quantity()`, and `quantity()`;
- the named bounded/equality constraint methods and `add_constraint()`;
- `add_variable()` and `add_objective_term()`;
- the read-only semantic `variables`, `constraints`, `expressions`, and
  `objective_terms` registries.

Domain helper arguments use public MATPOWER units. Values passed directly to
`quantity()` and backend-native expressions use the formulation's internal
units, normally per unit for power.

### Add backend-native variables, constraints, and objective terms

Use the backend-specific modeling package when a domain helper is not
sufficient. The callback can branch on `problem.backend` while keeping its
lifecycle and results common:

```python
from pwrs.power_models import PowerModel


def add_shortfall_penalty(problem: PowerModel) -> None:
    if problem.backend == "pyomo":
        import pyomo.environ as pyo

        shortfall = problem.add_variable(
            "shortfall",
            pyo.Var(domain=pyo.NonNegativeReals),
        )
        balance = pyo.Constraint(
            expr=problem.quantity("pg", 0) + shortfall >= 2.0
        )
    else:
        import cvxpy as cp

        shortfall = problem.add_variable(
            "shortfall",
            cp.Variable(nonneg=True),
        )
        balance = problem.quantity("pg", 0) + shortfall >= 2.0

    problem.add_constraint(
        "shortfall_balance",
        balance,
        senses="lower",
    )
    problem.add_objective_term(
        "shortfall_penalty",
        10_000.0 * shortfall,
    )
```

`add_constraint()` accepts one native constraint or a tuple of constraints.
Its optional `senses=` argument accepts `"lower"`, `"upper"`, `"equality"`,
`"bounded"`, or `"native"`, either once for a whole group or as one value per
constraint. Pyomo directions are inferred when unambiguous. Specify directions
for native CVXPY inequalities so that reported dual signs retain domain
meaning. CVXPY additions must remain valid under its DCP rules.

Auxiliary variables do not change the MATPOWER-compatible `result.x` layout.
Objective additions must be scalar and are included before final model
classification and solver selection. Consequently, an extension can change a
linear model into a QP or nonlinear problem and can change which default
solver is selected.

### Inspect structured extension results

Each named callback has a typed entry in `result.extensions`:

```python
extension = result.extensions["add_shortfall_penalty"]

print(extension.build_time)
print(extension.variables["shortfall"])
print(extension.objective_terms["shortfall_penalty"])

constraint = extension.constraints["shortfall_balance"]
print(constraint.senses)
print(constraint.dual)
print(constraint.native_dual)
print(constraint.violation)
print(constraint.max_violation)
```

`dual` follows the registered semantic direction, while `native_dual` is the
modeling package's value. The same data remains in
`result.raw["extensions"]` for mapping compatibility, but new code should use
the structured `result.extensions` API.

For direct model manipulation, build and solve a formulation explicitly:

```python
import pyomo.environ as pyo

from pwrs.power_models import build_acp_model, solve_acp_model

problem = build_acp_model(mp.case30())
problem.add_constraint(
    "minimum_generation",
    pyo.Constraint(expr=problem.quantity("pg", 0) >= 1.0),
    senses="lower",
)
result = solve_acp_model(problem, mpopt)
print(result.extensions["__built_model__"].constraints)
```

Direct additions made before solving are reported under the reserved
`"__built_model__"` extension name. Register every native variable and
constraint with the public methods: unregistered active constraints are
rejected to prevent a solve whose custom behavior is missing from diagnostics.
See the [API Reference](../../api/index.rst) for the domain helpers, formulation
builders, and matching solve functions.

## Current input and option limitations

The PowerModels-compatible backend deliberately rejects unsupported input
instead of silently dropping it. At present:

- input must be a MATPOWER case; native PowerModels component dictionaries,
  multinetwork data, and multiconductor data are not supported;
- zero-impedance branches and negative branch thermal ratings are rejected;
- every active generator needs an active-power `gencost` row, using a
  polynomial or piecewise-linear model;
- nonempty MATPOWER generalized constraint/cost fields
  (`A`, `l`, `u`, `N`, `H`, `Cw`, `fparm`, `z0`, `zl`, or `zu`) are rejected;
- MATPOWER `userfcn` callbacks are rejected; use
  `opf.power_models.extensions` instead;
- only options compatible with the selected formulation are accepted, and AC
  formulations currently support apparent-power branch limits only;
- some nonstandard component types are not yet supported.

DC lines are supported subject to their bounds, loss, and cost validation.
If a case depends on a feature in this list, use the MATPOWER-compatible
backend or reformulate the study explicitly.

## Troubleshooting

When a PowerModels-compatible solve fails:

1. Confirm the modeling package and requested solver are installed in the
   same environment used by `uv run`.
2. Check that the formulation uses the correct AC or DC model.
3. Leave the solver at `DEFAULT` to test capability-based selection.
4. Inspect solver fallback and problem-class metadata when a result exists.
5. For Ipopt, verify PyNumero ASL, CyIpopt, Ipopt, and any requested HSL
   library independently.
6. For a relaxation, distinguish solver infeasibility from the fact that its
   solution may not map to an AC-feasible operating point.
7. Remove user extensions and solve the base formulation before debugging an
   added constraint.
8. Compare the same case with `ACP` and the MATPOWER backend to separate an
   input problem from a formulation- or solver-specific issue.
