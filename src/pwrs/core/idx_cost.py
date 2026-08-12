"""Zero-based generator-cost columns and MATPOWER cost-model codes."""

PW_LINEAR = 1
POLYNOMIAL = 2

MODEL = 0
STARTUP = 1
SHUTDOWN = 2
NCOST = 3
COST = 4


def idx_cost():
    return (PW_LINEAR, POLYNOMIAL, MODEL, STARTUP, SHUTDOWN, NCOST, COST)
