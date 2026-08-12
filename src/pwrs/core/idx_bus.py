"""Zero-based bus-matrix columns and MATPOWER bus-type codes."""

PQ = 1
PV = 2
REF = 3
NONE = 4
BUS_I = 0
BUS_TYPE = 1
PD = 2
QD = 3
GS = 4
BS = 5
BUS_AREA = 6
VM = 7
VA = 8
BASE_KV = 9
ZONE = 10
VMAX = 11
VMIN = 12
LAM_P = 13
LAM_Q = 14
MU_VMAX = 15
MU_VMIN = 16


def idx_bus():
    return (
        PQ,
        PV,
        REF,
        NONE,
        BUS_I,
        BUS_TYPE,
        PD,
        QD,
        GS,
        BS,
        BUS_AREA,
        VM,
        VA,
        BASE_KV,
        ZONE,
        VMAX,
        VMIN,
        LAM_P,
        LAM_Q,
        MU_VMAX,
        MU_VMIN,
    )
