# Continuation power flow

Continuation power flow (CPF) traces a family of AC power-flow solutions as
load and generation move from a base injection pattern toward a target
pattern. It is primarily used to estimate voltage-stability margins, locate
the nose point of a P--V curve, and identify equipment limits encountered as
system stress increases.

In `pwrs`, use `runcpf()` with a base and a target MATPOWER case. The solver
first solves the base power flow and then follows the solution curve with a
predictor--corrector algorithm.

This guide focuses on defining a continuation study, configuring the trace,
and interpreting its results. For the augmented equations and numerical
algorithm, see [Continuation Power Flow](
../internals/continuation-power-flow/index.rst) in Power System Internals.
For the input matrix formats, see [Case data](case-data.md).

## A complete fixed-target workflow

The following example defines a target with 50% more active and reactive
load, traces the feasible path exactly to that target, and finds the lowest
bus-voltage magnitude at the endpoint.

```python
import numpy as np
import pwrs as mp
from pwrs.core.idx_bus import BUS_I, PD, QD

base = mp.case9()
target = mp.case9()
target.bus[:, PD] *= 1.5
target.bus[:, QD] *= 1.5

mpopt = mp.MatpowerConfig()
mpopt.cpf.stop_at = 1.0
mpopt.cpf.adapt_step = 1
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.runcpf(base, target, mpopt)
if not result.success:
    raise RuntimeError(result.cpf.done_msg)

final_vm = np.abs(result.cpf.V[:, -1])
weakest_row = int(np.nanargmin(final_vm))
weakest_bus = int(result.bus[weakest_row, BUS_I])

print(result.cpf.done_msg)
print(f"maximum lambda: {result.cpf.max_lam:.4f}")
print(f"lowest endpoint voltage: bus {weakest_bus}, {final_vm[weakest_row]:.4f} p.u.")
```

Here `lambda = 0` represents the solved base injections and `lambda = 1`
represents the target injections. Conceptually, the specified complex-power
injection follows

```text
S(lambda) = S_base + lambda * (S_target - S_base).
```

Values between zero and one interpolate between the two cases. Values above
one extrapolate the same injection direction beyond the target. A target is
therefore a direction and scale for the continuation, not necessarily the
point where the calculation stops.

## Preparing base and target cases

`runcpf()` accepts case names, `MatpowerCase` objects, or MATPOWER-style
mappings. It solves the base case before starting the continuation. If that
ordinary AC power flow does not converge, CPF terminates immediately.

The target case should describe changes in load and scheduled generation
while preserving the base network model. In practice, keep the following
identical between the cases:

- `baseMVA`, bus numbering, and row ordering;
- bus, generator, and branch counts;
- branch parameters, topology, and status;
- bus types and generator status;
- the association between generators and buses.

The implementation explicitly rejects changed bus types and generator
statuses. The admittance model is constructed from the base case, so branch
changes in the target case do not define a second network to interpolate.
Model topology changes as separate studies instead.

The target's `PD` and `QD` values define the load-transfer direction.
Scheduled active generation at non-reference generators can define how the
additional demand is shared. Reactive generation at voltage-controlled buses
and active generation at the reference bus are solved quantities, so their
target values do not define independent continuation controls.

`pwrs` includes `case9target()` as a standard example:

```python
import pwrs as mp

base = mp.case9()
target = mp.case9target()

mpopt = mp.MatpowerConfig()
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.runcpf(base, target, mpopt)
print(result.cpf.done_msg)
print(result.cpf.max_lam)
```

With the default stopping rule, this traces past `lambda = 1` when the AC
solution remains feasible beyond the injection pattern stored in
`case9target()`.

## Choosing where to stop

`cpf.stop_at` determines which event ends the trace.

| Value | Behavior |
| --- | --- |
| `"NOSE"` | Stop at the steady-state loading limit. This is the default. |
| Positive `float` | Stop at the requested lambda, locating it rather than accepting an overshooting step. |
| `"FULL"` | Pass the nose point, follow the lower branch, and stop when lambda returns to zero. |

Use a numeric target when the question is whether a specified transfer can be
reached:

```python
mpopt = mp.MatpowerConfig()
mpopt.cpf.stop_at = 0.75
```

Use `"NOSE"` when estimating the maximum loading parameter in the selected
direction. Use `"FULL"` mainly for analysis and visualization of the entire
P--V curve; it performs more continuation steps and follows the low-voltage
branch, which is normally not an acceptable operating region.

The returned `cpf.max_lam` is the largest corrected lambda encountered. For a
full trace it differs from the final lambda, which returns to zero. For this
reason, do not assume that `result.cpf.lam[0, -1]` is always the loading
margin.

## Choosing a parameterization

The added parameterization equation keeps the corrector problem nonsingular
as the path bends. Select it with `cpf.parameterization`:

| Value | Parameterization | Guidance |
| --- | --- | --- |
| `1` | Natural | Steps directly in lambda. Convenient for a target known to be before the nose, but poorly suited to passing the nose where lambda ceases to be monotonic. |
| `2` | Arc length | Constrains the Euclidean distance from the preceding solution. Can follow a folded curve in either direction. |
| `3` | Pseudo arc length | Constrains progress along the preceding tangent. This is the robust default for general studies. |

Keep pseudo arc length unless reproducing a particular study or comparing
continuation formulations. During precise target-lambda location, the solver
can temporarily use natural parameterization to land on the requested value.

## Controlling step size and corrector convergence

`cpf.step` is the initial and fixed continuation step size; its default is
`0.05`. A smaller value produces more points and generally makes the
corrector and event location easier, at the cost of more solves.

Enable adaptive step sizing for traces with both gently varying and sharply
curved regions:

```python
mpopt = mp.MatpowerConfig()
mpopt.cpf.step = 0.05
mpopt.cpf.adapt_step = 1
mpopt.cpf.step_min = 1e-4
mpopt.cpf.step_max = 0.2
mpopt.cpf.adapt_step_tol = 1e-3
mpopt.cpf.adapt_step_damping = 0.7
```

The adaptive controller compares each predicted state with its corrected
state. It increases the regular step when prediction is accurate and reduces
it when the local curve is harder to follow. `step_min` and `step_max` bound
normal adaptive steps; event-location rollbacks can still introduce a
smaller or zero-length step.

The corrector uses `pf.tol` as its mismatch tolerance and `pf.nr.max_it` as
its Newton iteration limit:

```python
mpopt.pf.tol = 1e-8
mpopt.pf.nr.max_it = 20
```

If the corrector fails, first reduce `cpf.step` or enable adaptive sizing.
Increasing the Newton iteration limit is useful only when the residual is
still making progress. Loosening `pf.tol` can hide an inaccurate point and
should not be the first remedy.

## Enforcing operating limits

Limit enforcement is disabled by default. This means a default nose-point
study traces the mathematical power-flow curve without changing bus types at
reactive limits or stopping at voltage and branch ratings.

| Option | Implemented behavior |
| --- | --- |
| `cpf.enforce_p_lims` | Locate generator `PMAX`, fix the generator at the limit, update the reference bus when necessary, and continue. |
| `cpf.enforce_q_lims` | Locate `QMIN`/`QMAX`, fix reactive generation, convert the affected voltage-controlled bus to PQ, and continue. |
| `cpf.enforce_v_lims` | Stop when a bus reaches `VMIN` or `VMAX`. |
| `cpf.enforce_flow_lims` | Stop when the larger from- or to-end apparent power reaches branch `RATE_A`. |

For example, a practical margin study might include all standard limits:

```python
mpopt = mp.MatpowerConfig()
mpopt.cpf.stop_at = "NOSE"
mpopt.cpf.adapt_step = 1
mpopt.cpf.enforce_p_lims = 1
mpopt.cpf.enforce_q_lims = 1
mpopt.cpf.enforce_v_lims = 1
mpopt.cpf.enforce_flow_lims = 1
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.runcpf(mp.case9(), mp.case9target(), mpopt)
print(result.cpf.done_msg)
for event in result.cpf.events:
    print(event.name, event.msg)
```

Active- and reactive-power limits change the equations and continue from the
located event, so `result.cpf.steps` can contain a zero-length restart.
Voltage and branch-flow limits are terminal events. Enabling the latter two
also validates the solved base case; the run stops immediately if the base
state already violates an enabled limit.

`enforce_p_lims` and `enforce_q_lims` currently require exactly one reference
bus when their event callbacks must change generator controls. Reactive-limit
enforcement can terminate with no REF or PV bus remaining, and active-limit
enforcement can terminate when all available generators are at `PMAX`.

Before enabling branch-flow enforcement, give every in-service branch a
positive, meaningful `RATE_A`. The current CPF event detector compares each
row directly with `RATE_A`; a zero value is therefore treated as a zero-MVA
limit rather than as an unlimited branch.

Each limit has a location tolerance in its native units:

- `cpf.p_lims_tol`: MW, default `0.01`;
- `cpf.q_lims_tol`: MVAr, default `0.01`;
- `cpf.v_lims_tol`: p.u., default `1e-4`;
- `cpf.flow_lims_tol`: MVA, default `0.01`.

`cpf.nose_tol` and `cpf.target_lam_tol` similarly control nose-point and
target-lambda event location. Tightening these values can add rollback steps
without materially changing the engineering conclusion.

## Understanding the result

`runcpf()` returns a strongly typed `ContinuationPowerFlowResult`. Its
`bus`, `gen`, and `branch` fields describe the final corrected point, while
`result.cpf` contains the full continuation trace.

| Field | Meaning |
| --- | --- |
| `result.success` | Whether the numerical calculation reached a normal termination rather than failing its base solve or corrector. |
| `result.et` | End-to-end CPF elapsed time. |
| `result.bus`, `gen`, `branch` | Solved MATPOWER matrices at the final corrected point. |
| `cpf.V` | Corrected complex bus voltages in p.u.; shape is `(number of external buses, number of saved points)`. |
| `cpf.V_hat` | Predictor voltages corresponding to the saved points. |
| `cpf.lam` | Corrected lambda values, with shape `(1, number of saved points)`. |
| `cpf.lam_hat` | Predictor lambda values. |
| `cpf.steps` | Step sizes recorded along the trace. |
| `cpf.iterations` | Number of completed continuation steps, excluding the base point. |
| `cpf.max_lam` | Maximum corrected lambda reached anywhere on the trace. |
| `cpf.events` | Located and logged limit/termination events. |
| `cpf.done_msg` | Human-readable reason the run stopped. |

Always inspect both `success` and `cpf.done_msg`. A voltage or flow limit is a
successful numerical termination, but it means the requested target or nose
was not reached. Conversely, `success=False` normally accompanies a failed
base power flow or corrector.

Each typed `CpfEvent` exposes `name`, `idx`, and `msg`. The meaning of `idx`
depends on the detector and may refer to a stacked internal event vector;
`msg` is the appropriate user-facing description of the affected external
bus, generator, or branch.

Mapping compatibility such as `result["cpf"]["lam"]` remains available, but
new code should prefer `result.cpf.lam` and the other typed attributes.

## Plotting a P--V curve

The trace arrays make it straightforward to plot one or more external buses:

```python
import matplotlib.pyplot as plt
import numpy as np
from pwrs.core.idx_bus import BUS_I

bus_number = 5
bus_ids = result.bus[:, BUS_I].astype(int)
row = int(np.flatnonzero(bus_ids == bus_number)[0])

lam = result.cpf.lam.ravel()
vm = np.abs(result.cpf.V[row, :])

plt.plot(lam, vm)
plt.xlabel("lambda")
plt.ylabel("voltage magnitude (p.u.)")
plt.title(f"P--V curve at bus {bus_number}")
plt.grid(True)
plt.show()
```

`pwrs` also provides a compatible built-in plotting callback through
`cpf.plot.level`. Zero disables it and a nonzero value enables creation and
finalization of the default plot; external bus numbers are supplied through
`cpf.plot.bus`. Although the configuration accepts MATPOWER levels `1`, `2`,
and `3`, the current Python callback does not yet distinguish incremental and
paused modes. Explicit plotting from the returned arrays is therefore
preferable in notebooks, tests, and headless batch jobs.

## Adding a user callback

Advanced studies can register one or more callbacks with
`cpf.user_callback`. A callback runs at initialization (`k == 0`), after
continuation steps (`k > 0`), and during finalization (`k < 0`). It receives
the next/current/previous continuation states, event information, callback
data, custom arguments, and the result accumulator.

The following callback records the minimum voltage at every accepted point:

```python
import numpy as np
import pwrs as mp


def record_min_voltage(k, nx, cx, px, done, rollback, evnts, cb_data, cb_args, results):
    key = "min_voltage"
    if rollback and k > 0:
        return nx, cx, done, rollback, evnts, cb_data, results

    if k == 0:
        nx.setdefault("cb", {})[key] = [float(np.min(np.abs(nx["V"])))]
    elif k > 0:
        values = list(nx["cb"][key])
        values.append(float(np.min(np.abs(nx["V"]))))
        nx["cb"][key] = values
    else:
        results[key] = np.asarray(nx["cb"][key])

    return nx, cx, done, rollback, evnts, cb_data, results


mpopt = mp.MatpowerConfig()
mpopt.cpf.stop_at = 0.5
mpopt.cpf.user_callback = record_min_voltage
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.runcpf(mp.case9(), mp.case9target(), mpopt)
minimum_voltage = result.cpf["min_voltage"]
```

Custom result fields are backend-specific extras, so mapping access is
appropriate for `minimum_voltage`; standard CPF fields remain typed
attributes. A callback that modifies state must preserve the seven-value
return contract shown above and avoid recording rolled-back steps.

To control ordering and pass arguments, assign a callback specification such
as `{"fcn": callback, "priority": 10, "args": [...]}`. Higher priorities run
first. A list of specifications registers multiple callbacks. The lower-level
`cpf_register_callback()` and `cpf_register_event()` functions are available
for MATPOWER-compatible extension work; most applications only need
`cpf.user_callback`.

## Output and compatibility interfaces

`verbose` controls CPF progress and event-location messages. For a quiet
batch calculation, use:

```python
mpopt.verbose = 0
mpopt.out.all = 0
```

`runcpf()` also accepts `fname` to append a formatted final report,
`solvedcase` to save the final case, and `nargout=2` to return
`(result, success)`. These exist for MATPOWER compatibility; normal Python
code should consume the structured result directly.

## Troubleshooting

When a continuation run ends unexpectedly:

1. Read `result.cpf.done_msg` before examining the final matrices.
2. Solve the base case with `runpf()` and correct any convergence or input
   problem there first.
3. Confirm base and target use identical topology, ordering, bus types,
   statuses, and `baseMVA`; change only the intended injection pattern.
4. For corrector failure, reduce `cpf.step`, enable adaptive sizing, and keep
   pseudo arc-length parameterization.
5. Check whether an enabled voltage or branch limit intentionally stopped the
   trace before the requested lambda or nose.
6. When enforcing active or reactive generation limits, verify that the case
   has exactly one reference bus and enough remaining regulating generation.
7. Inspect `cpf.events` for zero-length restarts and limit-induced changes to
   the PV/PQ/reference-bus partition.
8. Use `"FULL"` only when the descending branch is actually needed; for an
   operating margin, `"NOSE"` or a numeric target is easier to interpret.

For diagnostics below the high-level workflow, the API Reference documents
`cpf_tangent`, `cpf_predictor`, `cpf_corrector`, event detectors, and callback
handlers. Their role in the full algorithm is described in Power System
Internals rather than duplicated here.
