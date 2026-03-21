import pwrs as mp

mpc = mp.case9()
mpopt = mp.mpoption()
mpopt.verbose = 3
result = mp.runpf(mpc, mpopt)
