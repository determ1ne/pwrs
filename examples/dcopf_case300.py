import pwrs as mp
from pwrs.core.idx_gen import PG


mpc = mp.case300()
mpopt = mp.mpoption()
mpopt.out.all = 0
mpopt.verbose = 0

result = mp.rundcopf(mpc, mpopt, nargout=1)

print(f"success = {int(result['success'])}")
print(f"objective = {float(result['f']):.6f}")
print(f"total Pg = {float(result['gen'][:, PG - 1].sum()):.6f}")
