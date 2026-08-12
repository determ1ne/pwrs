from copy import deepcopy

import pwrs as mp
from pwrs.core.idx_bus import PD, QD
from pwrs.core.idx_gen import PG, PMAX

base = mp.case300()
target = deepcopy(base)

# Build a simple target pattern by increasing demand and scheduled generation.
target.bus[:, PD] *= 1.10
target.bus[:, QD] *= 1.10
target.gen[:, PG] *= 1.10
target.gen[:, PMAX] *= 1.10

mpopt = mp.mpoption()
mpopt.out.all = 0
mpopt.verbose = 1

result = mp.runcpf(base, target, mpopt, nargout=1)

print(f"success = {int(result.success)}")
print(f"done = {result.cpf.done_msg}")
print(f"steps = {result.cpf.lam.size}")
print(f"max lambda = {result.cpf.max_lam}")
