F_BUS = 1
T_BUS = 2
BR_R = 3
BR_X = 4
BR_B = 5
RATE_A = 6
RATE_B = 7
RATE_C = 8
TAP = 9
SHIFT = 10
BR_STATUS = 11
ANGMIN = 12
ANGMAX = 13
PF = 14
QF = 15
PT = 16
QT = 17
MU_SF = 18
MU_ST = 19
MU_ANGMIN = 20
MU_ANGMAX = 21


def idx_brch(*, nargout=None):
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
