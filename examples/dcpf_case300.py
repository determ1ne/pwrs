import pwrs as mp

results = mp.rundcpf(mp.case300(), nargout=1)

print(f"rundcpf success = {int(results['success'])}")
print(f"objective present = {'f' in results}")
print(f"first 10 bus angles (deg) = {results['bus'][:10, 8]}")
print(f"first 10 branch from-end flows (MW) = {results['branch'][:10, 13]}")
