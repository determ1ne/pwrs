import pwrs as mp
from pwrs.data.pglibopf import pglib_opf_case1354_pegase
from pwrs.power_models import PowerModel, constraint_active_generation_sum


def add_minimum_generation(problem: PowerModel) -> None:
    constraint_active_generation_sum(
        problem,
        "minimum_generation",
        generator_rows=[0],
        minimum_mw=100.0,
    )

mpc = pglib_opf_case1354_pegase()
mpopt = mp.mpoption()
mpopt.out.all = 0
mpopt.verbose = 0
mpopt.opf.backend = "POWER_MODELS"
mpopt.opf.power_models.formulation = "ACP"
mpopt.opf.power_models.extensions = (add_minimum_generation,)
mpopt.opf.ac.solver = "IPOPT"
mpopt.ipopt.opts.update(
    {
        "linear_solver": "ma27",
        "hsllib": "libcoinhsl.so",
    }
)

result, success = mp.runopf(mpc, mpopt, nargout=2)
assert result.extensions is not None
extension = result.extensions["add_minimum_generation"]
constraint = extension.constraints["minimum_generation"]

print(f"success = {int(success)}")
print(f"solver = {mpopt.opf.ac.solver}")
print(f"objective = {float(result.f):.6f}")
print(f"extension dual = {constraint.dual[0]:.6f}")
