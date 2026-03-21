import pwrs as mp

mpc = mp.case9241pegase()
opt = mp.mpoption()
# opt.mips.linsolver = 'LU'
result = mp.runopf(mpc, opt)
