# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import time

import numpy as np

from ..corex import MatpowerCase
from ..power_models import solve_acp_opf
from .ext2int import ext2int
from .idx_brch import MU_ANGMAX, MU_ANGMIN, MU_SF, MU_ST, PF, PT, QF, QT
from .idx_bus import MU_VMIN
from .idx_gen import MU_PMAX, MU_PMIN, MU_QMIN, PG, QG
from .int2ext import int2ext
from .mpoption import mpoption
from .opf_args import opf_args
from .opf_execute import opf_execute
from .opf_setup import opf_setup
from .runpf import runpf


def opf(*args, nargout=1):
    """Solve an optimal power flow.

    This port preserves the MATLAB ``opf`` calling conventions, supporting
    case dicts, case names, and the extended argument forms handled by
    :func:`opf_args`.

    Parameters
    ----------
    *args
        MATPOWER OPF inputs. Supported forms include ``opf(mpc)``,
        ``opf(mpc, mpopt)``, user linear/cost extensions, and the expanded
        ``baseMVA, bus, gen, branch, ...`` signatures.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    dict or tuple
        With one or two outputs, returns the solved results dict and
        optional success flag. With more outputs, returns
        ``(bus, gen, branch, f, success, info, et, g, jac, xr, pimul)``
        with ``g`` and ``jac`` currently left as ``None``.
    """
    t0 = time.time()
    mpc, mpopt = opf_args(*args, nargout=2)
    if str(mpopt.opf.backend).upper() == "POWER_MODELS":
        formulation = str(mpopt.opf.power_models.formulation).upper()
        if formulation != "ACP":
            raise NotImplementedError(f"unsupported PowerModels formulation: {formulation}")
        results, success, raw = solve_acp_opf(mpc, mpopt, nargout=3)
        et = time.time() - t0
        results["et"] = et
        results["raw"] = raw
        if nargout <= 2:
            return (results, success)[:nargout] if nargout > 1 else results
        outputs = (
            results["bus"],
            results["gen"],
            results["branch"],
            results["f"],
            success,
            raw["info"],
            et,
            None,
            None,
            raw["xr"],
            raw["pimul"],
        )
        return outputs[:nargout]
    if str(mpopt.opf.backend).upper() != "MATPOWER":
        raise ValueError(f"unsupported OPF backend: {mpopt.opf.backend}")
    if mpopt.opf.ac.solver.upper() == "DEFAULT":
        mpopt = mpoption(mpopt, "opf.ac.solver", "MIPS")
    if mpopt.opf.start == 3:
        mpopt_pf = mpoption(mpopt, "out.all", 0, "verbose", max(0, mpopt.verbose - 1))
        rpf = runpf(mpc, mpopt_pf, nargout=1)
        if rpf["success"]:
            mpc = rpf.to_dict() if isinstance(rpf, MatpowerCase) else dict(rpf)
            mpc = {key: value for key, value in mpc.items() if value is not None}
            mpc.pop("order", None)
    nb = mpc["bus"].shape[0]
    nl = mpc["branch"].shape[0]
    ng = mpc["gen"].shape[0]
    if mpc["bus"].shape[1] < MU_VMIN:
        mpc["bus"] = np.c_[mpc["bus"], np.zeros((nb, MU_VMIN - mpc["bus"].shape[1]))]
    if mpc["gen"].shape[1] < MU_QMIN:
        mpc["gen"] = np.c_[mpc["gen"], np.zeros((ng, MU_QMIN - mpc["gen"].shape[1]))]
    if mpc["branch"].shape[1] < MU_ANGMAX:
        mpc["branch"] = np.c_[mpc["branch"], np.zeros((nl, MU_ANGMAX - mpc["branch"].shape[1]))]
    mpc = ext2int(mpc, mpopt)
    om = opf_setup(mpc, mpopt, nargout=1)
    results, success, raw = opf_execute(om, mpopt, nargout=3)
    results["success"] = success
    results = int2ext(results, nargout=1)
    if "order" in results:
        gen_off = np.asarray(results["order"]["gen"]["status"]["off"]).astype(int).reshape(-1)
        if gen_off.size:
            results["gen"][np.ix_(gen_off - 1, [PG - 1, QG - 1, MU_PMAX - 1, MU_PMIN - 1])] = 0
        branch_off = np.asarray(results["order"]["branch"]["status"]["off"]).astype(int).reshape(-1)
        if branch_off.size:
            results["branch"][
                np.ix_(
                    branch_off - 1, [PF - 1, QF - 1, PT - 1, QT - 1, MU_SF - 1, MU_ST - 1, MU_ANGMIN - 1, MU_ANGMAX - 1]
                )
            ] = 0
    et = time.time() - t0
    if nargout <= 2:
        results["et"] = et
        results["raw"] = raw
        return (results, success)[:nargout] if nargout > 1 else results
    busout = results["bus"]
    genout = results["gen"]
    branchout = results["branch"]
    f = results["f"]
    info = raw["info"]
    xr = raw["xr"]
    pimul = raw["pimul"]
    outputs = (busout, genout, branchout, f, success, info, et, None, None, xr, pimul)
    return outputs[:nargout]
