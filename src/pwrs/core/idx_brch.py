"""Zero-based branch-matrix column indices."""

F_BUS = 0
T_BUS = 1
BR_R = 2
BR_X = 3
BR_B = 4
RATE_A = 5
RATE_B = 6
RATE_C = 7
TAP = 8
SHIFT = 9
BR_STATUS = 10
ANGMIN = 11
ANGMAX = 12
PF = 13
QF = 14
PT = 15
QT = 16
MU_SF = 17
MU_ST = 18
MU_ANGMIN = 19
MU_ANGMAX = 20


def idx_brch():
    return (
        F_BUS,
        T_BUS,
        BR_R,
        BR_X,
        BR_B,
        RATE_A,
        RATE_B,
        RATE_C,
        TAP,
        SHIFT,
        BR_STATUS,
        PF,
        QF,
        PT,
        QT,
        MU_SF,
        MU_ST,
        ANGMIN,
        ANGMAX,
        MU_ANGMIN,
        MU_ANGMAX,
    )
