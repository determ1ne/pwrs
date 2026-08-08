import pwrs as mp
from pwrs.data.pglibopf import pglib_opf_case1354_pegase

mpc = pglib_opf_case1354_pegase()
mpopt = mp.mpoption(
    "out.all",
    0,
    "verbose",
    0,
    "opf.backend",
    "POWER_MODELS",
    "opf.power_models.formulation",
    "ACP",
    "opf.ac.solver",
    "IPOPT",
    "ipopt.opts.linear_solver",
    "ma27",
    "ipopt.opts.hsllib",
    "libcoinhsl.dll",
    "ipopt.opts.print_level",
    5,
)

result = mp.runopf(mpc, mpopt)

print(f"success = {int(result['success'])}")
print(f"backend = {result['raw']['output']['backend']}")
print(f"formulation = {result['raw']['output']['formulation']}")
print(f"objective = {float(result['f']):.6f}")
print(f"max constraint violation = {result['raw']['output']['max_constraint_violation']:.3e}")
