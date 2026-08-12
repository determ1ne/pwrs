"""Zero-based generator-matrix column indices."""

GEN_BUS = 0
PG = 1
QG = 2
QMAX = 3
QMIN = 4
VG = 5
MBASE = 6
GEN_STATUS = 7
PMAX = 8
PMIN = 9
PC1 = 10
PC2 = 11
QC1MIN = 12
QC1MAX = 13
QC2MIN = 14
QC2MAX = 15
RAMP_AGC = 16
RAMP_10 = 17
RAMP_30 = 18
RAMP_Q = 19
APF = 20
MU_PMAX = 21
MU_PMIN = 22
MU_QMAX = 23
MU_QMIN = 24


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
