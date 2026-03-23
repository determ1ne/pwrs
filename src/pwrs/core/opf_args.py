# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause


from scipy import sparse

from ..corex import MatpowerCase
from .loadcase import loadcase
from .mpoption import mpoption


def opf_args(
    baseMVA,
    bus=None,
    gen=None,
    branch=None,
    areas=None,
    gencost=None,
    Au=None,
    lbu=None,
    ubu=None,
    mpopt=None,
    N=None,
    fparm=None,
    H=None,
    Cw=None,
    z0=None,
    zl=None,
    zu=None,
    nargout=1,
):
    """Parse and initialize OPF input arguments.

    This function preserves the MATLAB ``opf_args`` calling conventions and
    fills in defaults for omitted arguments.

    Parameters
    ----------
    baseMVA, bus, gen, branch, areas, gencost, Au, lbu, ubu, mpopt, N, fparm, H, Cw, z0, zl, zu
        OPF inputs accepted by MATPOWER. The problem data may be supplied
        as a case name, a case dict, or as the individual matrices/vectors.
        Optional user constraint, cost, initializer, and bound data may also
        be supplied directly or through fields in the case dict.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    tuple
        With ``nargout == 2``, returns ``(mpc, mpopt)``. Otherwise returns
        the fully expanded OPF argument tuple in MATLAB order.
    """
    want_mpc = nargout == 2
    userfcn = []
    if isinstance(baseMVA, (str, dict, MatpowerCase)):
        if bus is None:
            mpopt = mpoption(nargout=1)
            Au = sparse.csc_matrix((0, 0))
            lbu = []
            ubu = []
            N = sparse.csc_matrix((0, 0))
            fparm = []
            H = sparse.csc_matrix((0, 0))
            Cw = []
            z0 = []
            zl = []
            zu = []
        elif gen is None:
            mpopt = bus
            Au = sparse.csc_matrix((0, 0))
            lbu = []
            ubu = []
            N = sparse.csc_matrix((0, 0))
            fparm = []
            H = sparse.csc_matrix((0, 0))
            Cw = []
            z0 = []
            zl = []
            zu = []
        elif branch is None:
            userfcn = bus
            mpopt = gen
            Au = sparse.csc_matrix((0, 0))
            lbu = []
            ubu = []
            N = sparse.csc_matrix((0, 0))
            fparm = []
            H = sparse.csc_matrix((0, 0))
            Cw = []
            z0 = []
            zl = []
            zu = []
        else:
            if areas is None:
                mpopt = mpoption(nargout=1)
            else:
                mpopt = areas if mpopt is None else mpopt
            ubu = branch
            lbu = gen
            Au = bus
            if Cw is None:
                Cw = []
            if H is None:
                H = sparse.csc_matrix((0, 0))
            if fparm is None:
                fparm = []
            if N is None:
                N = sparse.csc_matrix((0, 0))
            if z0 is None:
                z0 = []
            if zl is None:
                zl = []
            if zu is None:
                zu = []
        mpc = loadcase(baseMVA, nargout=1)
        baseMVA = mpc["baseMVA"]
        bus = mpc["bus"]
        gen = mpc["gen"]
        branch = mpc["branch"]
        gencost = mpc["gencost"]
        areas = mpc.get("areas", [])
        if (Au is None or Au.shape[0] == 0) and "A" in mpc:
            Au, lbu, ubu = mpc["A"], mpc["l"], mpc["u"]
        if (N is None or N.shape[0] == 0) and "N" in mpc:
            N, Cw = mpc["N"], mpc["Cw"]
        if (H is None or H.shape == (0, 0)) and "H" in mpc:
            H = mpc["H"]
        if not fparm and "fparm" in mpc:
            fparm = mpc["fparm"]
        if not z0 and "z0" in mpc:
            z0 = mpc["z0"]
        if not zl and "zl" in mpc:
            zl = mpc["zl"]
        if not zu and "zu" in mpc:
            zu = mpc["zu"]
        if not userfcn and "userfcn" in mpc:
            userfcn = mpc["userfcn"]
    else:
        if mpopt is None:
            mpopt = mpoption()
        if Au is None:
            Au = sparse.csc_matrix((0, 0))
            lbu = []
            ubu = []
        if N is None:
            N = sparse.csc_matrix((0, 0))
            fparm = []
            H = sparse.csc_matrix((0, 0))
            Cw = []
            z0 = []
            zl = []
            zu = []
    if N is None:
        N = sparse.csc_matrix((0, 0))
    if H is None:
        H = sparse.csc_matrix((0, 0))
    if Au is None:
        Au = sparse.csc_matrix((0, 0))
    if mpopt is None:
        mpopt = mpoption(nargout=1)
    if want_mpc:
        mpc = {"baseMVA": baseMVA, "bus": bus, "gen": gen, "branch": branch, "gencost": gencost}
        if areas is not None and len(areas):
            mpc["areas"] = areas
        if Au.shape[0]:
            mpc["A"], mpc["l"], mpc["u"] = Au, lbu, ubu
        if N.shape[0]:
            mpc["N"], mpc["Cw"] = N, Cw
            if fparm is not None and len(fparm):
                mpc["fparm"] = fparm
            if H.shape != (0, 0):
                mpc["H"] = H
        if z0 is not None and len(z0):
            mpc["z0"] = z0
        if zl is not None and len(zl):
            mpc["zl"] = zl
        if zu is not None and len(zu):
            mpc["zu"] = zu
        if userfcn:
            mpc["userfcn"] = userfcn
        return mpc, mpopt
    return baseMVA, bus, gen, branch, gencost, Au, lbu, ubu, mpopt, N, fparm, H, Cw, z0, zl, zu, userfcn
