"""Zero-based DC-line matrix column indices."""

F_BUS = 0
T_BUS = 1
BR_STATUS = 2
PF = 3
PT = 4
QF = 5
QT = 6
VF = 7
VT = 8
PMIN = 9
PMAX = 10
QMINF = 11
QMAXF = 12
QMINT = 13
QMAXT = 14
LOSS0 = 15
LOSS1 = 16
MU_PMIN = 17
MU_PMAX = 18
MU_QMINF = 19
MU_QMAXF = 20
MU_QMINT = 21
MU_QMAXT = 22


def idx_dcline():
    return {
        "F_BUS": F_BUS,
        "T_BUS": T_BUS,
        "BR_STATUS": BR_STATUS,
        "PF": PF,
        "PT": PT,
        "QF": QF,
        "QT": QT,
        "VF": VF,
        "VT": VT,
        "PMIN": PMIN,
        "PMAX": PMAX,
        "QMINF": QMINF,
        "QMAXF": QMAXF,
        "QMINT": QMINT,
        "QMAXT": QMAXT,
        "LOSS0": LOSS0,
        "LOSS1": LOSS1,
        "MU_PMIN": MU_PMIN,
        "MU_PMAX": MU_PMAX,
        "MU_QMINF": MU_QMINF,
        "MU_QMAXF": MU_QMAXF,
        "MU_QMINT": MU_QMINT,
        "MU_QMAXT": MU_QMAXT,
    }
