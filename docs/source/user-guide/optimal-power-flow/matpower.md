# MATPOWER-compatible OPF

The MATPOWER-compatible backend is the default OPF implementation in `pwrs`.
It follows the MATPOWER case-to-model pipeline, exposes the familiar
MP-Opt-Model result structures, and is the appropriate choice when behavioral
compatibility with MATPOWER is more important than selecting among many
alternative network formulations.

## Run the default AC OPF

No backend option is required. The following explicit assignment can be
useful in applications that allow users to switch backends:

```python
import pwrs as mp

mpopt = mp.MatpowerConfig()
mpopt.opf.backend = "MATPOWER"
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.runopf(mp.case30(), mpopt)
if not result.success:
    raise RuntimeError("MATPOWER-compatible AC OPF did not converge")

print(result.f)
print(result.gen)
```

For AC OPF, `opf.ac.solver = "DEFAULT"` resolves to MIPS. MIPS is the
project's primal-dual interior-point solver and requires no external
optimization package.

## Configure MIPS

Use the `mips` section of `MatpowerConfig` to change its termination and
iteration controls:

```python
import pwrs as mp

mpopt = mp.MatpowerConfig()
mpopt.opf.ac.solver = "MIPS"
mpopt.opf.violation = 5e-6
mpopt.mips.feastol = 5e-6
mpopt.mips.gradtol = 1e-6
mpopt.mips.comptol = 1e-6
mpopt.mips.costtol = 1e-6
mpopt.mips.max_it = 200
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.runopf(mp.case118(), mpopt)
```

Leaving `mips.feastol` at zero derives it from `opf.violation`. Increase
`max_it` only after checking the model and the convergence trace. Relaxing
feasibility or optimality tolerances can make a marginal solve appear
successful without fixing the underlying modeling or scaling problem.

Set `mips.linsolver` to select the linear solver used by MIPS:

| Option | Backend | Dependency | Description |
| --- | --- | --- | --- |
| `""`, `"\\"` | SciPy/NumPy default | `scipy`, `numpy` | Uses SciPy's sparse solver for sparse matrices and NumPy's dense solver otherwise. |
| `"LU3"`, `"LU3a"`, `"LU3m"`, `"LU3am"`, `"SUPERLU"` | SuperLU | `scipy` | Explicitly uses SciPy SuperLU with minimum-degree ordering. The MATPOWER-compatible aliases have the same behavior in pwrs. |
| `"LU"`, `"LU4"`, `"LU4m"`, `"LU5"`, `"LU5m"`, `"UMFPACK"` | SuiteSparse UMFPACK | `scikit-umfpack` and a system SuiteSparse installation | Uses UMFPACK for sparse direct solves. pwrs intentionally does not distinguish the MATLAB four- and five-output LU forms. |
| `"PARDISO"` | Intel MKL PARDISO | `pypardiso` (installs the required MKL runtime) | Uses the multithreaded sparse direct solver exposed by PyPardiso. |
| `"KLU"` | SuiteSparse KLU | `nbklu` | Uses KLU with block triangular form reordering disabled, matching the MIPS integration. |

The optional Python packages are included in the `all` dependency group and
can be installed with `uv sync --group all`. If a selected optional solver or
its native library is unavailable, pwrs emits a `RuntimeWarning` and falls
back to SciPy SuperLU.

## Select the AC formulation

The default MATPOWER-compatible AC formulation uses power-balance equations
and polar voltage variables. Two options expose the other formulation choices:

| Option | Value | Formulation |
| --- | --- | --- |
| `opf.current_balance` | `0` | Complex power balance (default). |
| `opf.current_balance` | `1` | Complex current balance. |
| `opf.v_cartesian` | `0` | Polar voltage variables (default). |
| `opf.v_cartesian` | `1` | Cartesian voltage variables. |

For example:

```python
mpopt = mp.MatpowerConfig()
mpopt.opf.current_balance = 1
mpopt.opf.v_cartesian = 1
mpopt.opf.ac.solver = "MIPS"
```

Other useful controls are:

- `opf.start`: use the solver default (`0`), an interior initialization (`1`),
  the case state (`2`), or a preliminary power-flow solution (`3`).
- `opf.flow_lim`: enforce apparent power (`"S"`), active power (`"P"`),
  squared active power (`"2"`), or current magnitude (`"I"`) at rated
  branches.
- `opf.ignore_angle_lim`: ignore branch angle-difference limits when set to
  `1`.
- `opf.use_vg`: control how generator voltage setpoints affect voltage bounds.
- `opf.return_raw_der`: include raw derivative data for detailed diagnostics.

The standard power-balance/polar formulation is the best starting point.
Change one formulation option at a time when investigating numerical behavior.

## Run a DC OPF

`rundcopf()` is the clearest high-level entry point because it sets the model
to `"DC"` while preserving the remaining options:

```python
import pwrs as mp
from pwrs.core.idx_bus import BUS_I, LAM_P, VA
from pwrs.core.idx_gen import GEN_BUS, PG

mpopt = mp.MatpowerConfig()
mpopt.opf.backend = "MATPOWER"
mpopt.verbose = 0
mpopt.out.all = 0

result = mp.rundcopf(mp.case30(), mpopt)
if not result.success:
    raise RuntimeError("MATPOWER-compatible DC OPF failed")

print(result.bus[:, [BUS_I, VA, LAM_P]])
print(result.gen[:, [GEN_BUS, PG]])
```

## MATPOWER user costs and constraints

Choose this backend when the case uses MATPOWER's generalized OPF extension
fields, including user linear constraints (`A`, `l`, and `u`) or generalized
cost fields (`N`, `fparm`, `H`, `Cw`). These fields are incorporated by the
MATPOWER-compatible setup pipeline and are retained in the structured result.

For reusable MATPOWER-style callbacks, use the `userfcn` mechanism and the
corresponding low-level OPF model APIs. The PowerModels-compatible backend has
a separate extension interface and does not currently consume these MATPOWER
user constraint or generalized cost fields.

## Inspect optimization results

The solved matrices contain the most portable economic and engineering
results:

- `result.bus`: `LAM_P`, `LAM_Q`, `MU_VMIN`, and `MU_VMAX`;
- `result.gen`: `PG`, `QG`, and generator bound multipliers;
- `result.branch`: terminal flows and branch flow/angle multipliers.

The MATPOWER path also exposes model-level structures through `result.x`,
`result.mu`, `result.var`, `result.lin`, `result.nle`, and `result.nli`.
These are useful for advanced analysis, but their grouping and ordering follow
the selected formulation. Prefer solved matrix columns for application-level
interfaces that must remain stable across formulations.

Backend diagnostics remain in `result.raw`. For example:

```python
if result.raw is not None:
    print(result.raw.get("info"))
    print(result.raw.get("output"))
```

## Low-level workflow

Most applications should use `runopf()` or `rundcopf()`. For custom
MATPOWER-compatible models, the lower-level lifecycle is:

1. `opf_setup()` constructs the OPF model.
2. `opf_execute()` dispatches the selected nonlinear or quadratic solver.
3. `int2ext()` and the high-level runner restore external indexing and
   out-of-service rows.

The API Reference documents these functions and the objective, constraint,
Jacobian, and Hessian helpers. Their mathematical roles are described in
[OPF solution process](../../internals/optimal-power-flow/solution-process.md).

## Troubleshooting

When MIPS does not converge:

1. Set `verbose = 2` and keep the default power-balance/polar formulation.
2. Check generator, voltage, branch, and angle bounds for contradictions.
3. Confirm every active island has an online generator and a reference bus.
4. Inspect the initial `VM`, `VA`, `PG`, and `QG`, or try `opf.start = 3`.
5. Check cost scaling and avoid very different coefficient magnitudes where
   possible.
6. Compare an AC failure with DC OPF only as a diagnostic; DC feasibility does
   not imply AC feasibility.
7. Inspect the solver status and iteration metadata in `result.raw` before
   changing tolerances.
