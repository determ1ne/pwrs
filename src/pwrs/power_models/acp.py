# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""PowerModels-compatible AC-polar OPF formulation.

The formulation follows PowerModels.jl 0.19.9 ``ACPPowerModel``: branch
flows are explicit variables, nodal balances are sparse, and Ohm's law is
enforced independently at both ends of every active branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    MU_ANGMAX,
    MU_ANGMIN,
    MU_SF,
    MU_ST,
    PF,
    PT,
    QF,
    QT,
    RATE_A,
    SHIFT,
    T_BUS,
    TAP,
)
from ..core.idx_bus import (
    BS,
    BUS_I,
    BUS_TYPE,
    GS,
    LAM_P,
    LAM_Q,
    MU_VMAX,
    MU_VMIN,
    NONE,
    PD,
    PQ,
    PV,
    QD,
    REF,
    VA,
    VM,
    VMAX,
    VMIN,
)
from ..core.idx_cost import COST, MODEL, NCOST, POLYNOMIAL
from ..core.idx_gen import (
    GEN_BUS,
    GEN_STATUS,
    MU_PMAX,
    MU_PMIN,
    MU_QMAX,
    MU_QMIN,
    PG,
    PMAX,
    PMIN,
    QG,
    QMAX,
    QMIN,
    VG,
)


def _case_dict(mpc: Any) -> dict[str, Any]:
    data = mpc.to_dict() if hasattr(mpc, "to_dict") else dict(mpc)
    return {key: value for key, value in data.items() if value is not None}


def _pad(matrix: np.ndarray, columns: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[1] >= columns:
        return matrix.copy()
    return np.c_[matrix, np.zeros((matrix.shape[0], columns - matrix.shape[1]))]


def _evaluate_polynomials(coefficients: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Evaluate one descending-power polynomial for each row of ``coefficients``."""
    output = np.zeros_like(values)
    for coefficient in coefficients.T:
        output *= values
        output += coefficient
    return output


@dataclass(frozen=True)
class _Layout:
    va: slice
    vm: slice
    pg: slice
    qg: slice
    pf: slice
    qf: slice
    pt: slice
    qt: slice
    n: int


@dataclass
class _Network:
    source: dict[str, Any]
    bus_external: np.ndarray
    gen_external: np.ndarray
    branch_external: np.ndarray
    base_mva: float
    bus: np.ndarray
    gen: np.ndarray
    branch: np.ndarray
    gencost: np.ndarray
    bus_rows: np.ndarray
    gen_rows: np.ndarray
    branch_rows: np.ndarray
    gen_bus: np.ndarray
    f_bus: np.ndarray
    t_bus: np.ndarray
    refs: np.ndarray
    angle_pairs: np.ndarray
    angle_min: np.ndarray
    angle_max: np.ndarray
    rate: np.ndarray
    branch_reversed: np.ndarray
    tap: np.ndarray
    shift: np.ndarray
    b_fr: np.ndarray
    b_to: np.ndarray
    flow_equations: list[tuple[int, int, int, float, float, float]]
    layout: _Layout


def _connected_references(
    nb: int, f_bus: np.ndarray, t_bus: np.ndarray, refs: np.ndarray, gen_bus: np.ndarray
) -> np.ndarray:
    adjacency: list[list[int]] = [[] for _ in range(nb)]
    for f, t in zip(f_bus, t_bus):
        adjacency[int(f)].append(int(t))
        adjacency[int(t)].append(int(f))
    selected: list[int] = []
    seen = np.zeros(nb, dtype=bool)
    ref_set = set(int(i) for i in refs)
    gen_set = set(int(i) for i in gen_bus)
    for root in range(nb):
        if seen[root]:
            continue
        stack = [root]
        component: list[int] = []
        seen[root] = True
        while stack:
            i = stack.pop()
            component.append(i)
            for j in adjacency[i]:
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)
        candidates = sorted(ref_set.intersection(component))
        if candidates:
            selected.extend(candidates)
        else:
            gen_candidates = sorted(gen_set.intersection(component))
            selected.append(gen_candidates[0] if gen_candidates else min(component))
    return np.asarray(selected, dtype=int)


def _correct_bus_types(bus: np.ndarray, gen: np.ndarray) -> None:
    active_gen_buses = set(gen[:, GEN_BUS - 1].astype(int).tolist())
    slack_found = False
    for row in bus:
        bus_id = int(row[BUS_I - 1])
        bus_type = int(row[BUS_TYPE - 1])
        has_generator = bus_id in active_gen_buses
        if bus_type == PQ and has_generator:
            row[BUS_TYPE - 1] = PV
        elif bus_type == PV and not has_generator:
            row[BUS_TYPE - 1] = PQ
        elif bus_type == REF:
            if has_generator:
                slack_found = True
            else:
                row[BUS_TYPE - 1] = PQ
        elif bus_type not in (PQ, PV, REF, NONE):
            row[BUS_TYPE - 1] = PV if has_generator else PQ

    if not slack_found:
        if not len(gen):
            raise ValueError("POWER_MODELS/ACP requires an active generator to select a reference bus")
        largest = gen[np.argmax(gen[:, PMAX - 1])]
        reference_bus = int(largest[GEN_BUS - 1])
        matches = np.flatnonzero(bus[:, BUS_I - 1].astype(int) == reference_bus)
        if not matches.size:
            raise ValueError("POWER_MODELS/ACP reference generator is not connected to an active bus")
        bus[matches[0], BUS_TYPE - 1] = REF


def _build_network(mpc: Any) -> _Network:
    data = _case_dict(mpc)
    if data.get("dcline") is not None and np.size(data["dcline"]):
        raise NotImplementedError("POWER_MODELS/ACP does not yet support DC lines")
    if any(key in data for key in ("A", "N")):
        raise NotImplementedError("POWER_MODELS/ACP does not yet support MATPOWER user constraints or costs")

    base = float(data["baseMVA"])
    bus_all = np.asarray(data["bus"], dtype=float)
    gen_all = np.asarray(data["gen"], dtype=float)
    branch_all = np.asarray(data["branch"], dtype=float)
    gencost_all = np.asarray(data["gencost"], dtype=float)

    bus_rows = np.flatnonzero(bus_all[:, BUS_TYPE - 1] != NONE)
    active_ids = set(bus_all[bus_rows, BUS_I - 1].astype(int).tolist())
    gen_rows = np.flatnonzero(
        (gen_all[:, GEN_STATUS - 1] > 0) & np.isin(gen_all[:, GEN_BUS - 1].astype(int), list(active_ids))
    )
    branch_rows = np.flatnonzero(
        (branch_all[:, BR_STATUS - 1] > 0)
        & np.isin(branch_all[:, F_BUS - 1].astype(int), list(active_ids))
        & np.isin(branch_all[:, T_BUS - 1].astype(int), list(active_ids))
    )
    bus = bus_all[bus_rows].copy()
    gen = gen_all[gen_rows].copy()
    branch_corrected = branch_all.copy()

    tap_all = branch_corrected[:, TAP - 1].copy()
    tap_all[tap_all <= 0] = 1.0
    shift_all = np.deg2rad(branch_corrected[:, SHIFT - 1].copy())
    angmin_all = np.deg2rad(branch_corrected[:, ANGMIN - 1].copy())
    angmax_all = np.deg2rad(branch_corrected[:, ANGMAX - 1].copy())
    zero_angles = (angmin_all == 0) & (angmax_all == 0)
    angmin_all[(angmin_all <= -np.pi / 2) | zero_angles] = -1.0472
    angmax_all[(angmax_all >= np.pi / 2) | zero_angles] = 1.0472

    rate_all = branch_corrected[:, RATE_A - 1].copy() / base
    if np.any(rate_all < 0):
        raise ValueError("POWER_MODELS/ACP does not support negative branch thermal ratings")
    rate_all[rate_all == 0] = np.inf

    b_fr_all = branch_corrected[:, BR_B - 1].copy() / 2
    b_to_all = b_fr_all.copy()
    branch_reversed_all = np.zeros(len(branch_corrected), dtype=bool)
    orientations: set[tuple[int, int]] = set()
    for i, row in enumerate(branch_corrected):
        f, t = int(row[F_BUS - 1]), int(row[T_BUS - 1])
        if (t, f) not in orientations:
            orientations.add((f, t))
            continue

        branch_reversed_all[i] = True
        tap_original = tap_all[i]
        tap_squared = tap_original**2
        b_fr_original, b_to_original = b_fr_all[i], b_to_all[i]
        row[F_BUS - 1], row[T_BUS - 1] = row[T_BUS - 1], row[F_BUS - 1]
        row[BR_R - 1] *= tap_squared
        row[BR_X - 1] *= tap_squared
        b_fr_all[i] = b_to_original / tap_squared
        b_to_all[i] = b_fr_original * tap_squared
        tap_all[i] = 1 / tap_original
        shift_all[i] = -shift_all[i]
        angmin_all[i], angmax_all[i] = -angmax_all[i], -angmin_all[i]

    branch = branch_corrected[branch_rows].copy()
    tap = tap_all[branch_rows]
    shift = shift_all[branch_rows]
    angmin = angmin_all[branch_rows]
    angmax = angmax_all[branch_rows]
    rate = rate_all[branch_rows]
    b_fr = b_fr_all[branch_rows]
    b_to = b_to_all[branch_rows]
    branch_reversed = branch_reversed_all[branch_rows]

    if gencost_all.shape[0] < gen_all.shape[0]:
        raise ValueError("POWER_MODELS/ACP requires an active-power cost row for every generator")
    gencost = gencost_all[gen_rows].copy()
    if np.any(gencost[:, MODEL - 1].astype(int) != POLYNOMIAL):
        raise NotImplementedError("POWER_MODELS/ACP currently supports polynomial generator costs only")

    _correct_bus_types(bus, gen)

    bus_lookup = {int(bus_all[row, BUS_I - 1]): i for i, row in enumerate(bus_rows)}
    gen_bus = np.asarray([bus_lookup[int(v)] for v in gen[:, GEN_BUS - 1]], dtype=int)
    f_bus = np.asarray([bus_lookup[int(v)] for v in branch[:, F_BUS - 1]], dtype=int)
    t_bus = np.asarray([bus_lookup[int(v)] for v in branch[:, T_BUS - 1]], dtype=int)

    pair_values: dict[tuple[int, int], tuple[float, float]] = {}
    for f, t, amin, amax in zip(f_bus, t_bus, angmin, angmax):
        pair = (int(f), int(t))
        old = pair_values.get(pair, (-np.inf, np.inf))
        pair_values[pair] = (max(old[0], float(amin)), min(old[1], float(amax)))
    pairs = sorted(pair_values)
    angle_pairs = np.asarray(pairs, dtype=int).reshape((-1, 2))
    angle_min = np.asarray([pair_values[p][0] for p in pairs])
    angle_max = np.asarray([pair_values[p][1] for p in pairs])

    nb, ng, nl = len(bus), len(gen), len(branch)
    k = 0
    blocks: list[slice] = []
    for size in (nb, nb, ng, ng, nl, nl, nl, nl):
        blocks.append(slice(k, k + size))
        k += size
    layout = _Layout(*blocks, k)

    flow_equations: list[tuple[int, int, int, float, float, float]] = []
    r = branch[:, BR_R - 1]
    x = branch[:, BR_X - 1]
    den = r * r + x * x
    if np.any(den == 0):
        raise ValueError("POWER_MODELS/ACP does not support zero-impedance branches")
    g = r / den
    b = -x / den
    tr = tap * np.cos(shift)
    ti = tap * np.sin(shift)
    tm2 = tap * tap
    for i in range(nl):
        f, t = int(f_bus[i]), int(t_bus[i])
        flow_equations.extend(
            [
                (
                    layout.pf.start + i,
                    f,
                    t,
                    (g[i]) / tm2[i],
                    (-g[i] * tr[i] + b[i] * ti[i]) / tm2[i],
                    (-b[i] * tr[i] - g[i] * ti[i]) / tm2[i],
                ),
                (
                    layout.qf.start + i,
                    f,
                    t,
                    -(b[i] + b_fr[i]) / tm2[i],
                    (b[i] * tr[i] + g[i] * ti[i]) / tm2[i],
                    (-g[i] * tr[i] + b[i] * ti[i]) / tm2[i],
                ),
                (
                    layout.pt.start + i,
                    t,
                    f,
                    g[i],
                    (-g[i] * tr[i] - b[i] * ti[i]) / tm2[i],
                    (-b[i] * tr[i] + g[i] * ti[i]) / tm2[i],
                ),
                (
                    layout.qt.start + i,
                    t,
                    f,
                    -(b[i] + b_to[i]),
                    (b[i] * tr[i] - g[i] * ti[i]) / tm2[i],
                    (-g[i] * tr[i] - b[i] * ti[i]) / tm2[i],
                ),
            ]
        )

    refs = np.flatnonzero(bus[:, BUS_TYPE - 1] == REF)
    refs = _connected_references(nb, f_bus, t_bus, refs, gen_bus)
    return _Network(
        data,
        bus_all,
        gen_all,
        branch_all,
        base,
        bus,
        gen,
        branch,
        gencost,
        bus_rows,
        gen_rows,
        branch_rows,
        gen_bus,
        f_bus,
        t_bus,
        refs,
        angle_pairs,
        angle_min,
        angle_max,
        rate,
        branch_reversed,
        tap,
        shift,
        b_fr,
        b_to,
        flow_equations,
        layout,
    )


class _ACPProblem:
    def __init__(self, net: _Network):
        self.net = net
        self.nb = len(net.bus)
        self.ng = len(net.gen)
        self.nl = len(net.branch)
        self.nr = len(net.refs)
        self.np = len(net.angle_pairs)
        self.thermal = np.flatnonzero(np.isfinite(net.rate))
        self.nt = len(self.thermal)
        self.eq_count = self.nr + 2 * self.nb + 4 * self.nl
        self.m = self.eq_count + 2 * self.np + 2 * self.nt
        if net.flow_equations:
            flow = np.asarray(net.flow_equations, dtype=float)
            self._flow_idx = flow[:, 0].astype(np.intp)
            self._flow_u = flow[:, 1].astype(np.intp)
            self._flow_w = flow[:, 2].astype(np.intp)
            self._flow_a = flow[:, 3]
            self._flow_c = flow[:, 4]
            self._flow_s = flow[:, 5]
        else:
            self._flow_idx = np.empty(0, dtype=np.intp)
            self._flow_u = np.empty(0, dtype=np.intp)
            self._flow_w = np.empty(0, dtype=np.intp)
            self._flow_a = np.empty(0)
            self._flow_c = np.empty(0)
            self._flow_s = np.empty(0)
        self._prepare_cost_coefficients()
        self._jr, self._jc = self._jacobian_triplets(self.x0(), structure=True)
        self._hr, self._hc, self._hpos = self._build_hessian_structure()
        self._prepare_hessian_positions()

    def _prepare_cost_coefficients(self) -> None:
        orders = self.net.gencost[:, NCOST - 1].astype(np.intp)
        max_order = int(np.max(orders, initial=0))
        coefficients = np.zeros((self.ng, max_order))
        for i, order in enumerate(orders):
            coefficients[i, max_order - order :] = self.net.gencost[i, COST - 1 : COST - 1 + order]
        self._cost_coefficients = coefficients
        if max_order > 1:
            degree = np.arange(max_order - 1, 0, -1, dtype=float)
            self._cost_gradient_coefficients = coefficients[:, :-1] * degree
        else:
            self._cost_gradient_coefficients = np.zeros((self.ng, 0))
        if max_order > 2:
            degree = np.arange(max_order - 2, 0, -1, dtype=float)
            self._cost_hessian_coefficients = self._cost_gradient_coefficients[:, :-1] * degree
        else:
            self._cost_hessian_coefficients = np.zeros((self.ng, 0))

    def x0(self) -> np.ndarray:
        x = np.zeros(self.net.layout.n)
        x[self.net.layout.vm] = 1.0
        return x

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        n, v = self.net, self.net.layout
        lb = np.full(v.n, -np.inf)
        ub = np.full(v.n, np.inf)
        lb[v.vm], ub[v.vm] = n.bus[:, VMIN - 1], n.bus[:, VMAX - 1]
        lb[v.pg], ub[v.pg] = n.gen[:, PMIN - 1] / n.base_mva, n.gen[:, PMAX - 1] / n.base_mva
        lb[v.qg], ub[v.qg] = n.gen[:, QMIN - 1] / n.base_mva, n.gen[:, QMAX - 1] / n.base_mva
        for block in (v.pf, v.qf, v.pt, v.qt):
            lb[block], ub[block] = -n.rate, n.rate
        return lb, ub

    def constraint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lb = np.r_[
            np.zeros(self.eq_count), -np.inf * np.ones(self.np), self.net.angle_min, -np.inf * np.ones(2 * self.nt)
        ]
        ub = np.r_[
            np.zeros(self.eq_count),
            self.net.angle_max,
            np.inf * np.ones(self.np),
            self.net.rate[self.thermal] ** 2,
            self.net.rate[self.thermal] ** 2,
        ]
        return lb, ub

    def objective(self, x: np.ndarray) -> float:
        pg = x[self.net.layout.pg] * self.net.base_mva
        return float(np.sum(_evaluate_polynomials(self._cost_coefficients, pg)))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros(self.net.layout.n)
        pg = x[self.net.layout.pg] * self.net.base_mva
        grad[self.net.layout.pg] = _evaluate_polynomials(self._cost_gradient_coefficients, pg) * self.net.base_mva
        return grad

    def constraints(self, x: np.ndarray) -> np.ndarray:
        n, v = self.net, self.net.layout
        va, vm = x[v.va], x[v.vm]
        pg, qg = x[v.pg], x[v.qg]
        pf, qf, pt, qt = x[v.pf], x[v.qf], x[v.pt], x[v.qt]
        out = np.empty(self.m)
        row = 0
        out[row : row + self.nr] = va[n.refs]
        row += self.nr

        pbal = n.bus[:, PD - 1] / n.base_mva + n.bus[:, GS - 1] / n.base_mva * vm**2
        qbal = n.bus[:, QD - 1] / n.base_mva - n.bus[:, BS - 1] / n.base_mva * vm**2
        np.add.at(pbal, n.f_bus, pf)
        np.add.at(pbal, n.t_bus, pt)
        np.add.at(qbal, n.f_bus, qf)
        np.add.at(qbal, n.t_bus, qt)
        np.add.at(pbal, n.gen_bus, -pg)
        np.add.at(qbal, n.gen_bus, -qg)
        out[row : row + self.nb] = pbal
        row += self.nb
        out[row : row + self.nb] = qbal
        row += self.nb

        u, w = self._flow_u, self._flow_w
        d = va[u] - va[w]
        trig = self._flow_c * np.cos(d) + self._flow_s * np.sin(d)
        flow_count = len(self._flow_idx)
        out[row : row + flow_count] = x[self._flow_idx] - self._flow_a * vm[u] ** 2 - vm[u] * vm[w] * trig
        row += flow_count

        angle = va[n.angle_pairs[:, 0]] - va[n.angle_pairs[:, 1]]
        out[row : row + self.np] = angle
        row += self.np
        out[row : row + self.np] = angle
        row += self.np
        out[row : row + self.nt] = pf[self.thermal] ** 2 + qf[self.thermal] ** 2
        row += self.nt
        out[row : row + self.nt] = pt[self.thermal] ** 2 + qt[self.thermal] ** 2
        return out

    def jacobianstructure(self):
        return self._jr, self._jc

    def _jacobian_triplets(self, x: np.ndarray, structure: bool = False):
        n, v = self.net, self.net.layout
        va, vm = x[v.va], x[v.vm]
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        def add(r: int, c: int, value: float):
            rows.append(r)
            cols.append(c)
            vals.append(value)

        row = 0
        for bus_i in n.refs:
            add(row, v.va.start + int(bus_i), 1.0)
            row += 1
        for i in range(self.nb):
            add(row + i, v.vm.start + i, 2 * n.bus[i, GS - 1] / n.base_mva * vm[i])
        for i, bus_i in enumerate(n.gen_bus):
            add(row + int(bus_i), v.pg.start + i, -1.0)
        for i, (f, t) in enumerate(zip(n.f_bus, n.t_bus)):
            add(row + int(f), v.pf.start + i, 1.0)
            add(row + int(t), v.pt.start + i, 1.0)
        row += self.nb
        for i in range(self.nb):
            add(row + i, v.vm.start + i, -2 * n.bus[i, BS - 1] / n.base_mva * vm[i])
        for i, bus_i in enumerate(n.gen_bus):
            add(row + int(bus_i), v.qg.start + i, -1.0)
        for i, (f, t) in enumerate(zip(n.f_bus, n.t_bus)):
            add(row + int(f), v.qf.start + i, 1.0)
            add(row + int(t), v.qt.start + i, 1.0)
        row += self.nb

        for flow_idx, u, w, a, c, s in n.flow_equations:
            d = va[u] - va[w]
            trig = c * np.cos(d) + s * np.sin(d)
            dtrig = -c * np.sin(d) + s * np.cos(d)
            add(row, flow_idx, 1.0)
            add(row, v.va.start + u, -vm[u] * vm[w] * dtrig)
            add(row, v.va.start + w, vm[u] * vm[w] * dtrig)
            add(row, v.vm.start + u, -(2 * a * vm[u] + vm[w] * trig))
            add(row, v.vm.start + w, -vm[u] * trig)
            row += 1

        for lower in (False, True):
            for f, t in n.angle_pairs:
                add(row, v.va.start + int(f), 1.0)
                add(row, v.va.start + int(t), -1.0)
                row += 1
        for i in self.thermal:
            add(row, v.pf.start + i, 2 * x[v.pf.start + i])
            add(row, v.qf.start + i, 2 * x[v.qf.start + i])
            row += 1
        for i in self.thermal:
            add(row, v.pt.start + i, 2 * x[v.pt.start + i])
            add(row, v.qt.start + i, 2 * x[v.qt.start + i])
            row += 1
        if structure:
            return np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)
        return np.asarray(vals, dtype=float)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        n, v = self.net, self.net.layout
        va, vm = x[v.va], x[v.vm]
        values = np.empty(len(self._jr))
        offset = 0

        values[offset : offset + self.nr] = 1.0
        offset += self.nr
        values[offset : offset + self.nb] = 2 * n.bus[:, GS - 1] / n.base_mva * vm
        offset += self.nb
        values[offset : offset + self.ng] = -1.0
        offset += self.ng
        values[offset : offset + 2 * self.nl] = 1.0
        offset += 2 * self.nl
        values[offset : offset + self.nb] = -2 * n.bus[:, BS - 1] / n.base_mva * vm
        offset += self.nb
        values[offset : offset + self.ng] = -1.0
        offset += self.ng
        values[offset : offset + 2 * self.nl] = 1.0
        offset += 2 * self.nl

        u, w = self._flow_u, self._flow_w
        d = va[u] - va[w]
        trig = self._flow_c * np.cos(d) + self._flow_s * np.sin(d)
        dtrig = -self._flow_c * np.sin(d) + self._flow_s * np.cos(d)
        flow_values = np.column_stack(
            (
                np.ones(len(u)),
                -vm[u] * vm[w] * dtrig,
                vm[u] * vm[w] * dtrig,
                -(2 * self._flow_a * vm[u] + vm[w] * trig),
                -vm[u] * trig,
            )
        )
        count = flow_values.size
        values[offset : offset + count] = flow_values.ravel()
        offset += count

        count = 4 * self.np
        values[offset : offset + count] = np.tile((1.0, -1.0), 2 * self.np)
        offset += count
        count = 2 * self.nt
        values[offset : offset + count] = np.column_stack(
            (2 * x[v.pf][self.thermal], 2 * x[v.qf][self.thermal])
        ).ravel()
        offset += count
        values[offset : offset + count] = np.column_stack(
            (2 * x[v.pt][self.thermal], 2 * x[v.qt][self.thermal])
        ).ravel()
        return values

    def _build_hessian_structure(self):
        v = self.net.layout
        entries: set[tuple[int, int]] = set()

        def add(i: int, j: int):
            entries.add((max(i, j), min(i, j)))

        for i in range(self.ng):
            add(v.pg.start + i, v.pg.start + i)
        for i in range(self.nb):
            add(v.vm.start + i, v.vm.start + i)
        for _, u, w, _, _, _ in self.net.flow_equations:
            au, aw = v.va.start + u, v.va.start + w
            vu, vw = v.vm.start + u, v.vm.start + w
            for i, j in (
                (au, au),
                (aw, aw),
                (au, aw),
                (vu, vu),
                (vu, vw),
                (au, vu),
                (au, vw),
                (aw, vu),
                (aw, vw),
            ):
                add(i, j)
        for block in (v.pf, v.qf, v.pt, v.qt):
            for i in range(self.nl):
                add(block.start + i, block.start + i)
        ordered = sorted(entries)
        rows = np.asarray([item[0] for item in ordered], dtype=np.int32)
        cols = np.asarray([item[1] for item in ordered], dtype=np.int32)
        return rows, cols, {item: i for i, item in enumerate(ordered)}

    def hessianstructure(self):
        return self._hr, self._hc

    def _prepare_hessian_positions(self) -> None:
        v = self.net.layout

        def position(i: int, j: int) -> int:
            return self._hpos[(max(i, j), min(i, j))]

        self._pg_hpos = np.asarray([position(v.pg.start + i, v.pg.start + i) for i in range(self.ng)])
        self._vm_hpos = np.asarray([position(v.vm.start + i, v.vm.start + i) for i in range(self.nb)])
        flow_positions = np.empty((len(self._flow_u), 9), dtype=np.intp)
        for i, (u, w) in enumerate(zip(self._flow_u, self._flow_w)):
            au, aw = v.va.start + u, v.va.start + w
            vu, vw = v.vm.start + u, v.vm.start + w
            flow_positions[i] = [
                position(au, au),
                position(aw, aw),
                position(au, aw),
                position(vu, vu),
                position(vu, vw),
                position(au, vu),
                position(au, vw),
                position(aw, vu),
                position(aw, vw),
            ]
        self._flow_hpos = flow_positions.ravel()
        self._pf_hpos = np.asarray([position(v.pf.start + i, v.pf.start + i) for i in self.thermal])
        self._qf_hpos = np.asarray([position(v.qf.start + i, v.qf.start + i) for i in self.thermal])
        self._pt_hpos = np.asarray([position(v.pt.start + i, v.pt.start + i) for i in self.thermal])
        self._qt_hpos = np.asarray([position(v.qt.start + i, v.qt.start + i) for i in self.thermal])

    def hessian(self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float) -> np.ndarray:
        n, v = self.net, self.net.layout
        va, vm = x[v.va], x[v.vm]
        values = np.zeros(len(self._hr))

        pg = x[v.pg] * n.base_mva
        values[self._pg_hpos] += obj_factor * _evaluate_polynomials(self._cost_hessian_coefficients, pg) * n.base_mva**2

        pbal_row = self.nr
        qbal_row = pbal_row + self.nb
        values[self._vm_hpos] += (
            lagrange[pbal_row : pbal_row + self.nb] * 2 * n.bus[:, GS - 1] / n.base_mva
            - lagrange[qbal_row : qbal_row + self.nb] * 2 * n.bus[:, BS - 1] / n.base_mva
        )

        row = self.nr + 2 * self.nb
        u, w = self._flow_u, self._flow_w
        mult = lagrange[row : row + len(u)]
        d = va[u] - va[w]
        trig = self._flow_c * np.cos(d) + self._flow_s * np.sin(d)
        dtrig = -self._flow_c * np.sin(d) + self._flow_s * np.cos(d)
        common = mult * vm[u] * vm[w] * trig
        contributions = np.column_stack(
            (
                common,
                common,
                -common,
                -mult * 2 * self._flow_a,
                -mult * trig,
                -mult * vm[w] * dtrig,
                -mult * vm[u] * dtrig,
                mult * vm[w] * dtrig,
                mult * vm[u] * dtrig,
            )
        )
        np.add.at(values, self._flow_hpos, contributions.ravel())

        thermal_row = self.eq_count + 2 * self.np
        mult = 2 * lagrange[thermal_row : thermal_row + self.nt]
        values[self._pf_hpos] += mult
        values[self._qf_hpos] += mult
        thermal_row += self.nt
        mult = 2 * lagrange[thermal_row : thermal_row + self.nt]
        values[self._pt_hpos] += mult
        values[self._qt_hpos] += mult
        return values


def _result(
    net: _Network, x: np.ndarray, objective: float, success: int, info: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    bus = _pad(net.bus_external, MU_VMIN)
    gen = _pad(net.gen_external, MU_QMIN)
    branch = _pad(net.branch_external, MU_ANGMAX)
    v = net.layout
    va, vm = x[v.va], x[v.vm]
    bus[net.bus_rows, VA - 1] = np.rad2deg(va)
    bus[net.bus_rows, VM - 1] = vm
    gen[net.gen_rows, PG - 1] = x[v.pg] * net.base_mva
    gen[net.gen_rows, QG - 1] = x[v.qg] * net.base_mva
    gen[net.gen_rows, VG - 1] = vm[net.gen_bus]
    pf, qf = x[v.pf].copy(), x[v.qf].copy()
    pt, qt = x[v.pt].copy(), x[v.qt].copy()
    reversed_branches = net.branch_reversed
    pf[reversed_branches], pt[reversed_branches] = pt[reversed_branches], pf[reversed_branches]
    qf[reversed_branches], qt[reversed_branches] = qt[reversed_branches], qf[reversed_branches]
    branch[net.branch_rows, PF - 1] = pf * net.base_mva
    branch[net.branch_rows, QF - 1] = qf * net.base_mva
    branch[net.branch_rows, PT - 1] = pt * net.base_mva
    branch[net.branch_rows, QT - 1] = qt * net.base_mva

    mult_g = np.asarray(info.get("mult_g", np.zeros(0)), dtype=float).reshape(-1)
    mult_lb = np.asarray(info.get("mult_x_L", np.zeros(v.n)), dtype=float).reshape(-1)
    mult_ub = np.asarray(info.get("mult_x_U", np.zeros(v.n)), dtype=float).reshape(-1)
    nr, nb, nl, npairs = len(net.refs), len(net.bus), len(net.branch), len(net.angle_pairs)
    if mult_g.size:
        pbal = nr
        qbal = pbal + nb
        bus[net.bus_rows, LAM_P - 1] = mult_g[pbal : pbal + nb] / net.base_mva
        bus[net.bus_rows, LAM_Q - 1] = mult_g[qbal : qbal + nb] / net.base_mva
        thermal = nr + 2 * nb + 4 * nl + 2 * npairs
        finite_rate = np.flatnonzero(np.isfinite(net.rate))
        mu_sf = np.zeros(nl)
        mu_st = np.zeros(nl)
        nt = len(finite_rate)
        mu_sf[finite_rate] = 2 * mult_g[thermal : thermal + nt] * net.rate[finite_rate] / net.base_mva
        mu_st[finite_rate] = 2 * mult_g[thermal + nt : thermal + 2 * nt] * net.rate[finite_rate] / net.base_mva
        mu_sf[reversed_branches], mu_st[reversed_branches] = (
            mu_st[reversed_branches],
            mu_sf[reversed_branches],
        )
        branch[net.branch_rows, MU_SF - 1] = mu_sf
        branch[net.branch_rows, MU_ST - 1] = mu_st
        angle_upper = nr + 2 * nb + 4 * nl
        for pair_i, (f, t) in enumerate(net.angle_pairs):
            matches = np.flatnonzero((net.f_bus == f) & (net.t_bus == t))
            if matches.size:
                branch_i = matches[0]
                external = net.branch_rows[branch_i]
                if net.branch_reversed[branch_i]:
                    branch[external, MU_ANGMIN - 1] = mult_g[angle_upper + pair_i] * np.pi / 180
                    branch[external, MU_ANGMAX - 1] = -mult_g[angle_upper + npairs + pair_i] * np.pi / 180
                else:
                    branch[external, MU_ANGMAX - 1] = mult_g[angle_upper + pair_i] * np.pi / 180
                    branch[external, MU_ANGMIN - 1] = -mult_g[angle_upper + npairs + pair_i] * np.pi / 180
    if mult_lb.size == v.n and mult_ub.size == v.n:
        bus[net.bus_rows, MU_VMIN - 1] = mult_lb[v.vm]
        bus[net.bus_rows, MU_VMAX - 1] = mult_ub[v.vm]
        gen[net.gen_rows, MU_PMIN - 1] = mult_lb[v.pg] / net.base_mva
        gen[net.gen_rows, MU_PMAX - 1] = mult_ub[v.pg] / net.base_mva
        gen[net.gen_rows, MU_QMIN - 1] = mult_lb[v.qg] / net.base_mva
        gen[net.gen_rows, MU_QMAX - 1] = mult_ub[v.qg] / net.base_mva
    result = net.source.copy()
    result.update({"bus": bus, "gen": gen, "branch": branch, "f": objective, "x": x, "success": success})
    status_msg = info.get("status_msg", "")
    if isinstance(status_msg, (bytes, bytearray)):
        status_msg = status_msg.decode(errors="replace")
    raw = {
        "xr": x,
        "pimul": np.array([]),
        "info": int(info.get("status", -1)),
        "output": {
            "alg": "POWER_MODELS/ACP/IPOPT",
            "backend": "POWER_MODELS",
            "formulation": "ACP",
            "status": int(info.get("status", -1)),
            "status_msg": status_msg,
            "iterations": info.get("iter_count", info.get("iter")),
        },
    }
    return result, raw


def solve_acp_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Solve a MATPOWER case with the PowerModels ACP formulation and Ipopt."""
    if str(mpopt.model).upper() != "AC":
        raise ValueError("POWER_MODELS/ACP requires model='AC'")
    if str(mpopt.opf.ac.solver).upper() not in ("DEFAULT", "IPOPT"):
        raise ValueError("POWER_MODELS/ACP currently requires opf.ac.solver='IPOPT' or 'DEFAULT'")
    if float(mpopt.opf.current_balance) or float(mpopt.opf.v_cartesian):
        raise ValueError("POWER_MODELS/ACP uses polar voltage and power-balance equations")
    if str(mpopt.opf.flow_lim).upper() != "S":
        raise ValueError("POWER_MODELS/ACP currently supports apparent-power branch limits only")

    import cyipopt

    source = _case_dict(mpc)
    net = _build_network(source)
    problem_obj = _ACPProblem(net)
    lb, ub = problem_obj.bounds()
    cl, cu = problem_obj.constraint_bounds()
    problem = cyipopt.Problem(net.layout.n, problem_obj.m, problem_obj, lb, ub, cl, cu)
    options = dict(getattr(mpopt.ipopt, "opts", {}) or {})
    options.setdefault("print_level", min(12, int(mpopt.verbose) * 2 + 1) if int(mpopt.verbose) else 0)
    for key, value in options.items():
        problem.add_option(key, value)
    x, info = problem.solve(problem_obj.x0())
    x = np.asarray(x, dtype=float)
    status = int(info.get("status", -1))
    success = int(status in (0, 1))
    objective = float(info.get("obj_val", problem_obj.objective(x)))
    result, raw = _result(net, x, objective, success, info)
    values = problem_obj.constraints(x)
    finite_lower = np.isfinite(cl)
    finite_upper = np.isfinite(cu)
    violation = np.zeros_like(values)
    violation[finite_lower] = np.maximum(violation[finite_lower], cl[finite_lower] - values[finite_lower])
    violation[finite_upper] = np.maximum(violation[finite_upper], values[finite_upper] - cu[finite_upper])
    raw["output"]["max_constraint_violation"] = float(np.max(violation, initial=0.0))
    outputs = (result, success, raw)
    return outputs[:nargout] if nargout > 1 else result
