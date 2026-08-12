import pwrs as mp
from pwrs.core.idx_gen import PG

mpc = mp.case300()
result = mp.rundcopf(mpc)

print(f"success = {int(result.success)}")
print(f"objective = {float(result.f):.6f}")
print(f"total Pg = {float(result.gen[:, PG].sum()):.6f}")
