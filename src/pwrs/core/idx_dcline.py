F_BUS = 1
T_BUS = 2
BR_STATUS = 3
PF = 4
PT = 5
QF = 6
QT = 7
VF = 8
VT = 9
PMIN = 10
PMAX = 11
QMINF = 12
QMAXF = 13
QMINT = 14
QMAXT = 15
LOSS0 = 16
LOSS1 = 17
MU_PMIN = 18
MU_PMAX = 19
MU_QMINF = 20
MU_QMAXF = 21
MU_QMINT = 22
MU_QMAXT = 23


def idx_dcline(*, nargout=None):
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
