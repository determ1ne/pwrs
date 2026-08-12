# Power flow

Power flow determines the steady-state bus voltages and the active and
reactive power flowing through a network for a specified operating point.
In `pwrs`, use `runpf()` for an AC power flow and `rundcpf()` for the linear
DC approximation.

This guide focuses on running a study, selecting an algorithm, and consuming
the results. For the equations and numerical methods behind the calculation,
see [Power Flow](../internals/power-flow/index.rst) in Power System Internals.
For the input matrix formats, see [Case data](case-data.md).

## A complete AC power-flow workflow

The following example loads a built-in case, increases its active and
reactive demand by 5%, solves it quietly, checks convergence, and extracts
the most commonly used results.

```python
import pwrs as mp
from pwrs.core.idx_brch import F_BUS, PF, PT, QF, QT, T_BUS
from pwrs.core.idx_bus import BUS_I, PD, QD, VA, VM

# case9() returns a MatpowerCase. Its matrix fields are NumPy arrays.
mpc = mp.case9()
mpc.bus[:, PD - 1] *= 1.05
mpc.bus[:, QD - 1] *= 1.05

# Prefer the native Python configuration interface in new code.
mpopt = mp.MatpowerConfig()
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.runpf(mpc, mpopt)
if not result.success:
    raise RuntimeError("AC power flow did not converge")

bus = result.bus
branch = result.branch

bus_results = {
    "bus": bus[:, BUS_I - 1].astype(int),
    "vm_pu": bus[:, VM - 1],
    "va_deg": bus[:, VA - 1],
}
branch_results = {
    "from_bus": branch[:, F_BUS - 1].astype(int),
    "to_bus": branch[:, T_BUS - 1].astype(int),
    "pf_mw": branch[:, PF - 1],
    "qf_mvar": branch[:, QF - 1],
    "pt_mw": branch[:, PT - 1],
    "qt_mvar": branch[:, QT - 1],
}

print(
    f"converged in {result.iterations} iterations "
    f"and {result.et:.4f} seconds"
)
print(bus_results)
print(branch_results)
```

MATPOWER column constants are numbered from 1 for compatibility with
MATLAB. Subtract one whenever a constant is used as a NumPy column index.
Using `VM - 1` is safer and more readable than embedding column number `7`
directly in application code.

`runpf()` normalizes and deep-copies the case before solving it. The input
`mpc` therefore retains the operating point supplied by the caller, while
the returned `PowerFlowResult` contains the solved state. In particular, an
input branch matrix with the standard 13 columns is expanded in the result
to include `PF`, `QF`, `PT`, and `QT`.

## Understanding the result

The normal Python call returns a `PowerFlowResult` containing the original
case fields plus calculation status and solved values. Its fields use
attribute access and are statically typed.

| Field | Meaning |
| --- | --- |
| `success` | `True` when the numerical solve converged. |
| `iterations` | Total solver iterations, including repeated solves used to enforce reactive-power limits. |
| `et` | Elapsed solver time in seconds. |
| `bus` | Bus data updated with voltage magnitude `VM` and angle `VA`. |
| `gen` | Generator data updated with active power `PG` and reactive power `QG`. |
| `branch` | Branch data extended with from- and to-end active and reactive flows. |
| `order` | Bookkeeping used to map internal indexing and out-of-service elements back to the external case. |

Always check `success` before treating the returned voltages and flows as a
valid operating point. Offline generators and branches are restored in the
external result; their calculated power columns are set to zero.

For compatibility with existing MATPOWER-style Python code, mapping access
such as `result["bus"]`, `result.get("order")`, and `result.to_dict()` remains
available. New code should prefer `result.bus` and other attributes.

The relevant constants are defined in `pwrs.core.idx_bus`,
`pwrs.core.idx_gen`, and `pwrs.core.idx_brch`. The full table definitions are
available in the [API Reference](../api/index.rst).

## Choosing an AC algorithm

The default algorithm is Newton's method using power mismatch equations and
polar voltage variables. It is the appropriate starting point for most
meshed transmission-system studies.

| `pf.alg` | Method | Typical use |
| --- | --- | --- |
| `NR`, `NR-SP` | Newton, power mismatch, polar voltage | General-purpose default. |
| `NR-SC`, `NR-SH` | Newton, power mismatch, Cartesian or hybrid voltage | Alternative voltage representations for difficult cases or comparison studies. |
| `NR-IP`, `NR-IC`, `NR-IH` | Newton, current mismatch, polar, Cartesian, or hybrid voltage | Alternative balance formulations for numerical studies. |
| `FDXB`, `FDBX` | Fast-decoupled XB or BX | Lower-cost iterations when the decoupling assumptions suit the network. |
| `GS` | Gauss--Seidel | Primarily compatibility, teaching, and small studies. |
| `PQSUM`, `ISUM`, `YSUM` | Power, current, or admittance summation | Radial networks only. |

Select an algorithm and tune its stopping criteria through
`MatpowerConfig`:

```python
import pwrs as mp

mpopt = mp.MatpowerConfig()
mpopt.pf.alg = "FDXB"
mpopt.pf.tol = 1e-8
mpopt.pf.fd.max_it = 50
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.runpf(mp.case30(), mpopt)
```

Newton methods use `pf.nr.max_it`; fast-decoupled, Gauss--Seidel, and radial
methods use `pf.fd.max_it`, `pf.gs.max_it`, and `pf.radial.max_it`,
respectively. `pf.tol` is the termination tolerance on the per-unit
mismatch.

For large Newton solves, `pf.nr.lin_solver` selects the linear-system
backend. Leaving it empty lets `pwrs` choose a default. Setting it to `"KLU"`
uses the optional `nbklu` package when available; otherwise `pwrs` warns and
falls back to the default solver.

Users migrating MATPOWER scripts can configure the same option with the
compatible name/value interface:

```python
mpopt = mp.mpoption("pf.alg", "FDXB", "verbose", 0, "out.all", 0)
```

Explicit Newton names such as `NR-IC` set the corresponding balance and
voltage representation. Non-Newton algorithms require power balance with
polar voltages; incompatible `pf.current_balance` or `pf.v_cartesian`
settings raise an error. See [Power-flow solution algorithms](
../internals/power-flow/solution-algorithms.md) for the mathematical and
implementation-level distinctions.

## Enforcing generator reactive-power limits

An ordinary AC power flow maintains the voltage setpoint at PV and reference
buses even if the resulting generator reactive power lies outside `QMIN` and
`QMAX`. Enable reactive-power limit enforcement when the study must account
for this operational constraint:

```python
mpopt = mp.MatpowerConfig()
mpopt.pf.enforce_q_lims = 1
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.runpf(mp.case30(), mpopt)
```

The available modes are:

- `0`: do not enforce reactive-power limits (default).
- `1`: fix all currently violated generators at their limits and convert
  the corresponding buses to PQ simultaneously.
- `2`: apply only the largest remaining violation before solving again.

Enforcement can require multiple power-flow solves, so `iterations` is the
total across all passes. Once a voltage-controlling generator reaches a
reactive limit, its bus voltage magnitude is no longer guaranteed to remain
at the original setpoint. The solve can also become infeasible if no PV or
reference generator remains able to regulate voltage. Enforcement of a
reactive limit on a reference bus is not supported for systems with multiple
reference buses.

See [Limits, post-processing, and results](
../internals/power-flow/limits-and-results.md) for the detailed calculation
sequence.

## DC power flow

Use `rundcpf()` when the linear DC approximation is appropriate, for example
for screening, active-power transfer analysis, or initialization of another
study:

```python
import pwrs as mp
from pwrs.core.idx_brch import F_BUS, PF, T_BUS
from pwrs.core.idx_bus import BUS_I, VA

mpopt = mp.MatpowerConfig()
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.rundcpf(mp.case30(), mpopt)
if not result.success:
    raise RuntimeError("DC power flow failed")

bus_angles = result.bus[:, [BUS_I - 1, VA - 1]]
active_flows = result.branch[:, [F_BUS - 1, T_BUS - 1, PF - 1]]
print(bus_angles)
print(active_flows)
```

The DC model fixes voltage magnitudes at 1 p.u., omits reactive power, and
returns zero reactive branch flows. Bus angles remain in degrees and active
branch flows remain in MW in the external result. Because voltage magnitude,
reactive-power balance, losses, and other AC effects are absent, a successful
DC solve is not evidence that the same operating point is AC feasible.

Lower-level functions such as `dcpf`, `makeBdc`, and `makePTDF` are available
for specialized workflows; most applications should begin with `rundcpf()`.

## Controlling printed output

`verbose` controls progress and convergence messages from the algorithms,
while the fields under `out` control the formatted result report. For a quiet
batch calculation, set both controls:

```python
mpopt = mp.MatpowerConfig()
mpopt.verbose = 0
mpopt.out.all = 0
```

For an interactive diagnostic run, retain the normal report and increase
verbosity:

```python
mpopt = mp.MatpowerConfig()
mpopt.verbose = 2
mpopt.out.bus = 1
mpopt.out.branch = 1
```

`runpf()` also accepts `fname` for appending a formatted report,
`solvedcase` for saving a solved MATPOWER case, and `nargout` for emulating
MATLAB return signatures. These are compatibility interfaces; new Python
code should normally consume the returned structured result directly.

## Troubleshooting convergence and input problems

When a calculation fails or produces an unexpected result:

1. Check `result.success` and rerun with `verbose = 2` before inspecting
   the numerical result.
2. Verify the case has `baseMVA`, `bus`, `gen`, and `branch`, with valid bus
   numbers, an in-service reference generator, and a connected path for each
   energized island.
3. Check initial `VM`, `VA`, and generator `VG` values, especially for a case
   assembled or modified programmatically.
4. Retry with the default `NR` algorithm before tuning tolerances or maximum
   iteration counts. A looser tolerance can hide modeling errors and should
   not be the first remedy.
5. If `pf.enforce_q_lims` is enabled, inspect `QG`, `QMIN`, `QMAX`, and bus
   type conversion; the constrained problem may not have a feasible voltage-
   regulated solution.
6. Use `PQSUM`, `ISUM`, and `YSUM` only for radial networks, and do not combine
   non-Newton methods with current-balance or Cartesian-voltage options.
7. Treat ZIP-load warnings explicitly. Newton current-balance,
   Cartesian/hybrid Newton, and Gauss--Seidel configurations that do not
   support the requested ZIP model fall back to constant-power loads.

For diagnostics below the high-level workflow, the API Reference documents
`bustypes`, `makeYbus`, `makeSbus`, `pfsoln`, and the individual solver
functions. Their role in the full calculation is described in Power System
Internals rather than duplicated here.
