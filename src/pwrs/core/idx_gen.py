GEN_BUS = 1
PG = 2
QG = 3
QMAX = 4
QMIN = 5
VG = 6
MBASE = 7
GEN_STATUS = 8
PMAX = 9
PMIN = 10
PC1 = 11
PC2 = 12
QC1MIN = 13
QC1MAX = 14
QC2MIN = 15
QC2MAX = 16
RAMP_AGC = 17
RAMP_10 = 18
RAMP_30 = 19
RAMP_Q = 20
APF = 21
MU_PMAX = 22
MU_PMIN = 23
MU_QMAX = 24
MU_QMIN = 25


def idx_gen():
    return (
        GEN_BUS,
        PG,
        QG,
        QMAX,
        QMIN,
        VG,
        MBASE,
        GEN_STATUS,
        PMAX,
        PMIN,
        MU_PMAX,
        MU_PMIN,
        MU_QMAX,
        MU_QMIN,
        PC1,
        PC2,
        QC1MIN,
        QC1MAX,
        QC2MIN,
        QC2MAX,
        RAMP_AGC,
        RAMP_10,
        RAMP_30,
        RAMP_Q,
        APF,
    )
