import pwrs as mp

mpc = mp.case9()
mpopt = mp.mpoption()
mpopt.verbose = 3
mpopt.pf.alg = "FDXB"
result = mp.runpf(mpc, mpopt)
