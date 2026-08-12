import pwrs as mp
from pwrs.core.idx_brch import PF
from pwrs.core.idx_bus import VA

results = mp.rundcpf(mp.case300())

print(f"rundcpf success = {int(results.success)}")
print(f"first 10 bus angles (deg) = {results.bus[:10, VA]}")
print(f"first 10 branch from-end flows (MW) = {results.branch[:10, PF]}")
