# Optimal power flow

Optimal power flow (OPF) chooses a feasible operating point that minimizes an
objective, normally the generation cost stored in a MATPOWER case's
`gencost` matrix. Unlike an ordinary power flow, OPF can change generator
dispatch and controlled voltages while enforcing network and equipment
limits.

In `pwrs`, `runopf()` solves an AC OPF and `rundcopf()` solves its DC
approximation. Both return an `OptimalPowerFlowResult` with the same public
result interface, regardless of the selected backend.

This guide focuses on preparing a study, choosing a backend, and consuming
the result. For the equations and numerical solution process, see
[Optimal Power Flow](../internals/optimal-power-flow/index.rst) in Power
System Internals. For MATPOWER matrix formats, see [Case data](case-data.md).

```{toctree}
:maxdepth: 1
:caption: OPF backends

optimal-power-flow/matpower
optimal-power-flow/power-models
```

## A complete OPF workflow

The following example loads a built-in case, increases its demand, runs the
default AC OPF quietly, checks the solve status, and extracts dispatch and
locational marginal prices.

```python
import pwrs as mp
from pwrs.core.idx_bus import BUS_I, LAM_P, PD, QD, VA, VM
from pwrs.core.idx_gen import GEN_BUS, PG, QG

mpc = mp.case30()
mpc.bus[:, PD] *= 1.05
mpc.bus[:, QD] *= 1.05

mpopt = mp.MatpowerConfig()
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.runopf(mpc, mpopt)
if not result.success:
    raise RuntimeError("optimal power flow did not converge")

bus_solution = result.bus[:, [BUS_I, VM, VA, LAM_P]]
generator_solution = result.gen[:, [GEN_BUS, PG, QG]]

print(f"objective: {result.f:.2f}")
print(bus_solution)
print(generator_solution)
```

The `pwrs.core.idx_*` column constants are zero-based and can be used
directly as NumPy column indices. The units of `result.f` are defined by the
coefficients in `gencost`; for the bundled economic-dispatch cases this is
normally a monetary cost per hour.

## Choosing a backend

`pwrs` has two OPF backends. The MATPOWER-compatible backend is the default,
so existing calls do not need an option change.

| Backend | Default solver | Choose it when |
| --- | --- | --- |
| `MATPOWER` | MIPS for AC OPF and the built-in QP path for DC OPF | MATPOWER behavior, user costs/constraints, and low-level MP-Opt-Model compatibility are the priority. |
| `POWER_MODELS` | Selected from the formulation and problem class | You need alternative PowerModels-style formulations, model extensions, or better performance on many medium and large problems with an appropriate external solver. |

The `POWER_MODELS` name denotes the PowerModels-compatible backend implemented
by `pwrs`. It runs entirely in Python: algebraic formulations use Pyomo and
explicit conic formulations use CVXPY. It follows formulation concepts from
PowerModels.jl.

Use the following rule of thumb:

- Start with [MATPOWER-compatible OPF](optimal-power-flow/matpower.md) when
  porting MATPOWER code or when a case contains MATPOWER user constraints or
  generalized costs.
- Start with [PowerModels-compatible OPF](optimal-power-flow/power-models.md)
  for larger AC OPF studies, alternative formulations and relaxations, or
  reusable model extensions.
- Compare both backends if runtime is important. Confirm that they solve the
  same formulation and constraints before comparing objectives or timings.

## AC and DC models

`runopf()` uses the nonlinear AC network model by default. It includes voltage
magnitudes, reactive power, losses, and apparent-power branch limits. Use it
when the returned operating point must represent the AC network.

`rundcopf()` selects the active-power-only DC approximation:

```python
import pwrs as mp

mpopt = mp.MatpowerConfig()
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.rundcopf(mp.case30(), mpopt)
if not result.success:
    raise RuntimeError("DC optimal power flow failed")
```

DC OPF is useful for screening and market-style active-power studies, but its
successful solution does not establish AC feasibility. It omits reactive
power, fixes voltage magnitudes, and neglects the nonlinear AC loss model.

## Cost data and constraints

Every active generator needs an active-power cost row in `gencost`. `pwrs`
supports the standard MATPOWER polynomial and piecewise-linear cost models.
Generator limits, bus voltage bounds, branch ratings, and branch angle limits
come from the corresponding `gen`, `bus`, and `branch` columns.

Before solving a modified or programmatically created case, verify at least:

- an active generator and reference bus exist in each energized island;
- `PMIN <= PMAX`, `QMIN <= QMAX`, and `VMIN <= VMAX`;
- each active generator has the intended `gencost` row;
- branch ratings and angle bounds use the expected units and conventions;
- the objective coefficients have compatible units and scaling.

The two backends share the standard MATPOWER input matrices, but they do not
currently accept every extension in common. Consult the backend pages before
using user-defined costs, constraints, DC lines, or nonstandard formulations.

## Understanding the result

`runopf()` and `rundcopf()` return a strongly typed
`OptimalPowerFlowResult`. New code should use attribute access.

| Field | Meaning |
| --- | --- |
| `success` | Whether the solver reported a successful optimum. |
| `f` | Final objective value. |
| `et` | End-to-end elapsed time measured by the high-level OPF call. |
| `bus` | Solved bus state and bus constraint multipliers. |
| `gen` | Solved generator dispatch and generator-limit multipliers. |
| `branch` | Solved branch flows and flow/angle-limit multipliers. |
| `x` | Backend decision vector in its native result ordering. |
| `mu` | Structured variable and constraint multipliers when available. |
| `var`, `lin`, `nle`, `nli` | Named optimization-model result groups when exposed by the backend. |
| `raw` | Backend-specific solver status, diagnostics, and metadata. |

For example, `LAM_P` and `LAM_Q` in `result.bus` are the active- and
reactive-power balance multipliers, while `MU_PMIN` and `MU_PMAX` in
`result.gen` describe binding active-generation bounds. Always use the
constants from `pwrs.core.idx_bus`, `pwrs.core.idx_gen`, and
`pwrs.core.idx_brch` rather than embedding column numbers.

Mapping-style access such as `result["bus"]` remains available for MATPOWER
compatibility, but new Python code should prefer `result.bus`. `result.raw`
is deliberately an opaque mapping because its contents depend on the backend
and solver; check for `None` before reading it.

## Failure handling

Treat `success` as a required precondition for consuming an OPF solution. On
failure, first rerun with `mpopt.verbose = 2`, inspect the solver status in
`result.raw` when a result is returned, and check the input limits and costs.
Changing tolerances should not be the first response to an infeasible model.

Some external solvers raise an exception when they are missing, incompatible
with the selected formulation, or fail before producing a usable solution.
Catch those exceptions at batch boundaries and record the backend,
formulation, solver, and case name so the calculation can be reproduced.
