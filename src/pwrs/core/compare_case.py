# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import io

import numpy as np

from .idx_brch import ANGMAX, MU_ST, QT
from .idx_bus import MU_VMIN, VMIN
from .idx_gen import APF, MU_QMIN
from .loadcase import loadcase


def compare_case(mpc1, mpc2):
    """Print a column-wise comparison of two MATPOWER cases.

    Loads two MATPOWER cases, compares the shared bus, generator, and branch
    matrices column by column, and prints the largest row-wise differences in
    the same style as MATPOWER's ``compare_case``.

    Parameters
    ----------
    mpc1 : dict or str
        First MATPOWER case struct or case name/path.
    mpc2 : dict or str
        Second MATPOWER case struct or case name/path.
    nargout : int, optional
        MATLAB compatibility flag. ``compare_case`` prints output and does not
        return values.

    Returns
    -------
    None
        This function prints formatted output and does not return a value.
    """
    _, bus1, gen1, branch1 = loadcase(mpc1, nargout=4)
    _, bus2, gen2, branch2 = loadcase(mpc2, nargout=4)

    solvedPF = 0
    solvedOPF = 0
    Nb = VMIN
    Ng = APF
    Nl = ANGMAX

    if branch1.shape[1] >= QT and branch2.shape[1] >= QT:
        solvedPF = 1
        Nl = QT
        if branch1.shape[1] >= MU_ST and branch2.shape[1] >= MU_ST:
            solvedOPF = 1
            Nb = MU_VMIN
            Ng = MU_QMIN
            Nl = MU_ST

    buscols = ["BUS_I", "BUS_TYPE", "PD", "QD", "GS", "BS", "BUS_AREA", "VM", "VA", "BASE_KV", "ZONE", "VMAX", "VMIN"]
    gencols = [
        "GEN_BUS",
        "PG",
        "QG",
        "QMAX",
        "QMIN",
        "VG",
        "MBASE",
        "GEN_STATUS",
        "PMAX",
        "PMIN",
        "PC1",
        "PC2",
        "QC1MIN",
        "QC1MAX",
        "QC2MIN",
        "QC2MAX",
        "RAMP_AGC",
        "RAMP_10",
        "RAMP_30",
        "RAMP_Q",
        "APF",
    ]
    brcols = [
        "F_BUS",
        "T_BUS",
        "BR_R",
        "BR_X",
        "BR_B",
        "RATE_A",
        "RATE_B",
        "RATE_C",
        "TAP",
        "SHIFT",
        "BR_STATUS",
        "ANGMIN",
        "ANGMAX",
    ]
    if solvedPF:
        brcols.extend(["PF", "QF", "PT", "QT"])
        if solvedOPF:
            buscols.extend(["LAM_P", "LAM_Q", "MU_VMAX", "MU_VMIN"])
            gencols.extend(["MU_PMAX", "MU_PMIN", "MU_QMAX", "MU_QMIN"])
            brcols.extend(["MU_SF", "MU_ST"])

    out = io.StringIO()
    out.write("----------------  --------------  --------------  --------------  -----\n")
    out.write(" matrix / col         case 1          case 2        difference     row \n")
    out.write("----------------  --------------  --------------  --------------  -----\n")

    def _emit(name, cols, a1, a2, ncols):
        temp = np.max(np.abs(a1[:, :ncols] - a2[:, :ncols]), axis=0)
        gmax = int(np.argmax(temp)) + 1
        out.write(name)
        nodiff = " : no differences found"
        for j in range(len(cols)):
            diff = np.abs(a1[:, j] - a2[:, j])
            i = int(np.argmax(diff))
            v = diff[i]
            if v != 0:
                nodiff = ""
                s = " *" if j + 1 == gmax else ""
                out.write(f"\n  {cols[j]:<12}")
                out.write(f"{a1[i, j]:16g}{a2[i, j]:16g}{v:16g}{i + 1:7d}{s}")
        out.write(f"{nodiff}\n")

    _emit("bus", buscols, bus1, bus2, Nb)
    out.write("\n")
    _emit("gen", gencols, gen1, gen2, Ng)
    out.write("\n")
    _emit("branch", brcols, branch1, branch2, Nl)

    print(out.getvalue(), end="")
