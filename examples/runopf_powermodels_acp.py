import pwrs as mp
from pwrs.data.pglibopf import pglib_opf_case1354_pegase


def main():
    mpc = pglib_opf_case1354_pegase()
    mpopt = mp.mpoption()
    mpopt.opf.backend = "POWER_MODELS"
    mpopt.opf.power_models.formulation = "ACP"
    mpopt.opf.ac.solver = "IPOPT"
    mpopt.ipopt.opts.update(
        {
            "linear_solver": "ma27",
            "hsllib": "libcoinhsl.dll",
            "print_level": 5,
        }
    )

    result = mp.runopf(mpc, mpopt)

    print(f"success = {int(result.success)}")
    print(f"backend = {mpopt.opf.backend}")
    print(f"formulation = {mpopt.opf.power_models.formulation}")
    print(f"objective = {float(result.f):.6f}")

if __name__ == '__main__':
    main()