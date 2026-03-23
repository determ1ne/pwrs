import pwrs as mp

# case = mp.case13659pegase()
case = mp.case_ACTIVSg70k()
opt = mp.mpoption()
opt.pf.nr.lin_solver = 'KLU'
results = mp.runpf(case, opt)