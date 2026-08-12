# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Formulation-neutral PowerModels network preparation.

This module owns MATPOWER input cleanup, active-component indexing, topology,
and branch admittance coefficients. It must not contain formulation variables,
constraint ordering, solver state, or result-vector layouts.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    RATE_A,
    SHIFT,
    T_BUS,
    TAP,
)
from ..core.idx_bus import BUS_I, BUS_TYPE, NONE, PQ, PV, REF
from ..core.idx_cost import MODEL, POLYNOMIAL, PW_LINEAR
from ..core.idx_dcline import BR_STATUS as DC_STATUS
from ..core.idx_dcline import F_BUS as DC_F_BUS
from ..core.idx_dcline import LOSS0, LOSS1, QMAXF, QMAXT, QMINF, QMINT
from ..core.idx_dcline import PMAX as DC_PMAX
from ..core.idx_dcline import PMIN as DC_PMIN
from ..core.idx_dcline import T_BUS as DC_T_BUS
from ..core.idx_gen import GEN_BUS, GEN_STATUS, PMAX


def case_dict(mpc: Any) -> dict[str, Any]:
    """Return the populated fields of a MATPOWER case as a plain dictionary."""
    data = mpc.to_dict() if hasattr(mpc, "to_dict") else dict(mpc)
    return {key: value for key, value in data.items() if value is not None}


def pad_matrix(matrix: np.ndarray, columns: int) -> np.ndarray:
    """Copy a MATPOWER matrix and append zero result columns when required."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[1] >= columns:
        return matrix.copy()
    return np.c_[matrix, np.zeros((matrix.shape[0], columns - matrix.shape[1]))]


def _has_component_data(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return bool(value)
    try:
        return bool(np.size(value))
    except TypeError:
        return True


def _validate_single_network_opf_data(data: dict[str, Any]) -> None:
    if data.get("multinetwork") or _has_component_data(data.get("nw")):
        raise NotImplementedError("POWER_MODELS does not yet support multinetwork data")
    if "conductors" in data:
        raise NotImplementedError("POWER_MODELS does not yet support multiconductor data")

    unsupported = tuple(name for name in ("storage", "switch", "ne_branch") if _has_component_data(data.get(name)))
    if unsupported:
        raise NotImplementedError(
            f"POWER_MODELS standard OPF does not yet support components: {', '.join(unsupported)}"
        )

    if any(isinstance(data.get(name), Mapping) for name in ("bus", "gen", "branch")):
        raise NotImplementedError(
            "POWER_MODELS native component dictionaries are not yet supported; provide a MATPOWER case"
        )

    generalized = tuple(
        name
        for name in ("A", "l", "u", "N", "H", "Cw", "fparm", "z0", "zl", "zu")
        if _has_component_data(data.get(name))
    )
    if generalized:
        raise NotImplementedError(
            "POWER_MODELS does not yet translate MATPOWER generalized constraints/costs: "
            + ", ".join(generalized)
        )
    if _has_component_data(data.get("userfcn")):
        raise NotImplementedError(
            "POWER_MODELS does not execute MATPOWER userfcn callbacks; use opf.power_models.extensions"
        )


@dataclass(frozen=True)
class BranchAdmittance:
    """Real and imaginary parts of the four branch admittance blocks."""

    g_ff: np.ndarray
    b_ff: np.ndarray
    g_ft: np.ndarray
    b_ft: np.ndarray
    g_tf: np.ndarray
    b_tf: np.ndarray
    g_tt: np.ndarray
    b_tt: np.ndarray


@dataclass
class PowerNetwork:
    """Prepared network data shared by all PowerModels formulations."""

    source: dict[str, Any]
    bus_external: np.ndarray
    gen_external: np.ndarray
    branch_external: np.ndarray
    dcline_external: np.ndarray
    base_mva: float
    bus: np.ndarray
    gen: np.ndarray
    branch: np.ndarray
    gencost: np.ndarray
    dcline: np.ndarray
    dclinecost: np.ndarray
    bus_rows: np.ndarray
    gen_rows: np.ndarray
    branch_rows: np.ndarray
    dcline_rows: np.ndarray
    gen_bus: np.ndarray
    f_bus: np.ndarray
    t_bus: np.ndarray
    dc_f_bus: np.ndarray
    dc_t_bus: np.ndarray
    dc_pmin_from: np.ndarray
    dc_pmax_from: np.ndarray
    dc_pmin_to: np.ndarray
    dc_pmax_to: np.ndarray
    dc_qmin_from: np.ndarray
    dc_qmax_from: np.ndarray
    dc_qmin_to: np.ndarray
    dc_qmax_to: np.ndarray
    dc_loss0: np.ndarray
    dc_loss1: np.ndarray
    refs: np.ndarray
    angle_pairs: np.ndarray
    angle_min: np.ndarray
    angle_max: np.ndarray
    branch_angle_min: np.ndarray
    branch_angle_max: np.ndarray
    rate: np.ndarray
    branch_reversed: np.ndarray
    tap: np.ndarray
    shift: np.ndarray
    b_fr: np.ndarray
    b_to: np.ndarray
    admittance: BranchAdmittance
    generators_at_bus: tuple[tuple[int, ...], ...]
    from_branches_at_bus: tuple[tuple[int, ...], ...]
    to_branches_at_bus: tuple[tuple[int, ...], ...]
    from_dclines_at_bus: tuple[tuple[int, ...], ...]
    to_dclines_at_bus: tuple[tuple[int, ...], ...]

    def bus_index(self, bus_id: int) -> int:
        """Return the internal index for an active MATPOWER bus number."""
        matches = np.flatnonzero(self.bus[:, BUS_I].astype(int) == int(bus_id))
        if not matches.size:
            raise KeyError(f"bus {bus_id} is not active in the PowerModels network")
        return int(matches[0])

    def generator_indices(self, rows: Any | None = None) -> tuple[int, ...]:
        """Map zero-based source generator rows to active internal indexes."""
        return self._component_indices("generator", self.gen_rows, rows)

    def branch_indices(self, rows: Any | None = None) -> tuple[int, ...]:
        """Map zero-based source branch rows to active internal indexes."""
        return self._component_indices("branch", self.branch_rows, rows)

    def dcline_indices(self, rows: Any | None = None) -> tuple[int, ...]:
        """Map zero-based source DC-line rows to active internal indexes."""
        return self._component_indices("DC line", self.dcline_rows, rows)

    @staticmethod
    def _component_indices(kind: str, active_rows: np.ndarray, rows: Any | None) -> tuple[int, ...]:
        if rows is None:
            return tuple(range(len(active_rows)))
        requested = np.asarray(rows, dtype=int).reshape(-1)
        if len(set(requested.tolist())) != len(requested):
            raise ValueError(f"duplicate {kind} source rows are not allowed")
        lookup = {int(row): i for i, row in enumerate(active_rows)}
        missing = [int(row) for row in requested if int(row) not in lookup]
        if missing:
            joined = ", ".join(map(str, missing))
            raise KeyError(f"{kind} source rows are inactive or absent: {joined}")
        return tuple(lookup[int(row)] for row in requested)


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
    active_gen_buses = set(gen[:, GEN_BUS].astype(int).tolist())
    slack_found = False
    for row in bus:
        bus_id = int(row[BUS_I])
        bus_type = int(row[BUS_TYPE])
        has_generator = bus_id in active_gen_buses
        if bus_type == PQ and has_generator:
            row[BUS_TYPE] = PV
        elif bus_type == PV and not has_generator:
            row[BUS_TYPE] = PQ
        elif bus_type == REF:
            if has_generator:
                slack_found = True
            else:
                row[BUS_TYPE] = PQ
        elif bus_type not in (PQ, PV, REF, NONE):
            row[BUS_TYPE] = PV if has_generator else PQ

    if not slack_found:
        if not len(gen):
            raise ValueError("POWER_MODELS requires an active generator to select a reference bus")
        largest = gen[np.argmax(gen[:, PMAX])]
        reference_bus = int(largest[GEN_BUS])
        matches = np.flatnonzero(bus[:, BUS_I].astype(int) == reference_bus)
        if not matches.size:
            raise ValueError("POWER_MODELS reference generator is not connected to an active bus")
        bus[matches[0], BUS_TYPE] = REF


def _branch_admittance(
    branch: np.ndarray,
    tap: np.ndarray,
    shift: np.ndarray,
    b_fr: np.ndarray,
    b_to: np.ndarray,
) -> BranchAdmittance:
    r = branch[:, BR_R]
    x = branch[:, BR_X]
    denominator = r * r + x * x
    if np.any(denominator == 0):
        raise ValueError("POWER_MODELS does not support zero-impedance branches")
    g = r / denominator
    b = -x / denominator
    tr = tap * np.cos(shift)
    ti = tap * np.sin(shift)
    tm2 = tap * tap
    return BranchAdmittance(
        g_ff=g / tm2,
        b_ff=(b + b_fr) / tm2,
        g_ft=(-g * tr + b * ti) / tm2,
        b_ft=(-b * tr - g * ti) / tm2,
        g_tf=(-g * tr - b * ti) / tm2,
        b_tf=(-b * tr + g * ti) / tm2,
        g_tt=g,
        b_tt=b + b_to,
    )


def _incidence_lists(
    nb: int, gen_bus: np.ndarray, f_bus: np.ndarray, t_bus: np.ndarray
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    generators: list[list[int]] = [[] for _ in range(nb)]
    from_branches: list[list[int]] = [[] for _ in range(nb)]
    to_branches: list[list[int]] = [[] for _ in range(nb)]
    for i, bus_i in enumerate(gen_bus):
        generators[int(bus_i)].append(i)
    for i, (f, t) in enumerate(zip(f_bus, t_bus)):
        from_branches[int(f)].append(i)
        to_branches[int(t)].append(i)
    return tuple(map(tuple, generators)), tuple(map(tuple, from_branches)), tuple(map(tuple, to_branches))


def _edge_incidence_lists(
    nb: int,
    f_bus: np.ndarray,
    t_bus: np.ndarray,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    from_edges: list[list[int]] = [[] for _ in range(nb)]
    to_edges: list[list[int]] = [[] for _ in range(nb)]
    for i, (f, t) in enumerate(zip(f_bus, t_bus)):
        from_edges[int(f)].append(i)
        to_edges[int(t)].append(i)
    return tuple(map(tuple, from_edges)), tuple(map(tuple, to_edges))


def _dcline_active_bounds(dcline: np.ndarray, base_mva: float) -> tuple[np.ndarray, ...]:
    pmin = dcline[:, DC_PMIN]
    pmax = dcline[:, DC_PMAX]
    loss0 = dcline[:, LOSS0] / base_mva
    loss1 = dcline[:, LOSS1]
    if np.any(loss1 < 0) or np.any(loss1 >= 1):
        raise ValueError("POWER_MODELS requires DC-line LOSS1 in [0, 1)")
    pmin = pmin / base_mva
    pmax = pmax / base_mva
    pminf = np.empty(len(dcline))
    pmaxf = np.empty(len(dcline))
    pmint = np.empty(len(dcline))
    pmaxt = np.empty(len(dcline))
    for i, (minimum, maximum) in enumerate(zip(pmin, pmax)):
        if minimum >= 0 and maximum >= 0:
            pminf[i], pmaxf[i] = minimum, maximum
            pmint[i] = loss0[i] - pmaxf[i] * (1 - loss1[i])
            pmaxt[i] = loss0[i] - pminf[i] * (1 - loss1[i])
        elif minimum >= 0 and maximum < 0:
            pminf[i], pmint[i] = minimum, maximum
            pmaxf[i] = (-pmint[i] + loss0[i]) / (1 - loss1[i])
            pmaxt[i] = loss0[i] - pminf[i] * (1 - loss1[i])
        elif minimum < 0 and maximum >= 0:
            pmaxt[i], pmaxf[i] = -minimum, maximum
            pminf[i] = (-pmaxt[i] + loss0[i]) / (1 - loss1[i])
            pmint[i] = loss0[i] - pmaxf[i] * (1 - loss1[i])
        else:
            pmaxt[i], pmint[i] = -minimum, maximum
            pmaxf[i] = (-pmint[i] + loss0[i]) / (1 - loss1[i])
            pminf[i] = (-pmaxt[i] + loss0[i]) / (1 - loss1[i])
    return pminf, pmaxf, pmint, pmaxt, loss0, loss1


def prepare_network(mpc: Any) -> PowerNetwork:
    """Clean and index a MATPOWER case without choosing a formulation."""
    data = case_dict(mpc)
    _validate_single_network_opf_data(data)
    if any(key in data for key in ("A", "N")):
        raise NotImplementedError("POWER_MODELS does not yet support MATPOWER user constraints or costs")

    base = float(data["baseMVA"])
    bus_all = np.asarray(data["bus"], dtype=float)
    gen_all = np.asarray(data["gen"], dtype=float)
    branch_all = np.asarray(data["branch"], dtype=float)
    gencost_all = np.asarray(data["gencost"], dtype=float)
    dcline_value = data.get("dcline")
    dcline_all = (
        np.empty((0, LOSS1 + 1))
        if dcline_value is None or not np.size(dcline_value)
        else np.atleast_2d(np.asarray(dcline_value, dtype=float))
    )
    if dcline_all.shape[1] <= LOSS1:
        raise ValueError(f"POWER_MODELS requires DC-line rows with at least {LOSS1 + 1} columns")

    bus_rows = np.flatnonzero(bus_all[:, BUS_TYPE] != NONE)
    active_ids = set(bus_all[bus_rows, BUS_I].astype(int).tolist())
    gen_rows = np.flatnonzero(
        (gen_all[:, GEN_STATUS] > 0) & np.isin(gen_all[:, GEN_BUS].astype(int), list(active_ids))
    )
    branch_rows = np.flatnonzero(
        (branch_all[:, BR_STATUS] > 0)
        & np.isin(branch_all[:, F_BUS].astype(int), list(active_ids))
        & np.isin(branch_all[:, T_BUS].astype(int), list(active_ids))
    )
    dcline_rows = np.flatnonzero(
        (dcline_all[:, DC_STATUS] > 0)
        & np.isin(dcline_all[:, DC_F_BUS].astype(int), list(active_ids))
        & np.isin(dcline_all[:, DC_T_BUS].astype(int), list(active_ids))
    )
    bus = bus_all[bus_rows].copy()
    gen = gen_all[gen_rows].copy()
    branch_corrected = branch_all.copy()

    tap_all = branch_corrected[:, TAP].copy()
    tap_all[tap_all <= 0] = 1.0
    shift_all = np.deg2rad(branch_corrected[:, SHIFT].copy())
    angmin_all = np.deg2rad(branch_corrected[:, ANGMIN].copy())
    angmax_all = np.deg2rad(branch_corrected[:, ANGMAX].copy())
    zero_angles = (angmin_all == 0) & (angmax_all == 0)
    angmin_all[(angmin_all <= -np.pi / 2) | zero_angles] = -1.0472
    angmax_all[(angmax_all >= np.pi / 2) | zero_angles] = 1.0472

    rate_all = branch_corrected[:, RATE_A].copy() / base
    if np.any(rate_all < 0):
        raise ValueError("POWER_MODELS does not support negative branch thermal ratings")
    rate_all[rate_all == 0] = np.inf

    b_fr_all = branch_corrected[:, BR_B].copy() / 2
    b_to_all = b_fr_all.copy()
    branch_reversed_all = np.zeros(len(branch_corrected), dtype=bool)
    orientations: set[tuple[int, int]] = set()
    for i, row in enumerate(branch_corrected):
        f, t = int(row[F_BUS]), int(row[T_BUS])
        if (t, f) not in orientations:
            orientations.add((f, t))
            continue

        branch_reversed_all[i] = True
        tap_original = tap_all[i]
        tap_squared = tap_original**2
        b_fr_original, b_to_original = b_fr_all[i], b_to_all[i]
        row[F_BUS], row[T_BUS] = row[T_BUS], row[F_BUS]
        row[BR_R] *= tap_squared
        row[BR_X] *= tap_squared
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
        raise ValueError("POWER_MODELS requires an active-power cost row for every generator")
    gencost = gencost_all[gen_rows].copy()
    cost_models = gencost[:, MODEL].astype(int)
    if np.any(~np.isin(cost_models, (POLYNOMIAL, PW_LINEAR))):
        raise NotImplementedError("POWER_MODELS supports polynomial and piecewise-linear generator costs")
    dcline = dcline_all[dcline_rows].copy()
    dclinecost_value = data.get("dclinecost")
    if dclinecost_value is None or not np.size(dclinecost_value):
        dclinecost_all = np.tile(np.asarray([POLYNOMIAL, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]), (len(dcline_all), 1))
    else:
        dclinecost_all = np.atleast_2d(np.asarray(dclinecost_value, dtype=float))
        if len(dclinecost_all) != len(dcline_all):
            raise ValueError("POWER_MODELS requires one dclinecost row per DC line")
    dclinecost = dclinecost_all[dcline_rows].copy()

    _correct_bus_types(bus, gen)

    bus_lookup = {int(bus_all[row, BUS_I]): i for i, row in enumerate(bus_rows)}
    gen_bus = np.asarray([bus_lookup[int(v)] for v in gen[:, GEN_BUS]], dtype=int)
    f_bus = np.asarray([bus_lookup[int(v)] for v in branch[:, F_BUS]], dtype=int)
    t_bus = np.asarray([bus_lookup[int(v)] for v in branch[:, T_BUS]], dtype=int)
    dc_f_bus = np.asarray([bus_lookup[int(v)] for v in dcline[:, DC_F_BUS]], dtype=int)
    dc_t_bus = np.asarray([bus_lookup[int(v)] for v in dcline[:, DC_T_BUS]], dtype=int)
    dc_bounds = _dcline_active_bounds(dcline, base)
    dc_qmin_from = dcline[:, QMINF] / base
    dc_qmax_from = dcline[:, QMAXF] / base
    dc_qmin_to = dcline[:, QMINT] / base
    dc_qmax_to = dcline[:, QMAXT] / base
    if np.any(dc_qmin_from > dc_qmax_from) or np.any(dc_qmin_to > dc_qmax_to):
        raise ValueError("POWER_MODELS DC-line reactive lower bounds cannot exceed upper bounds")

    pair_values: dict[tuple[int, int], tuple[float, float]] = {}
    for f, t, amin, amax in zip(f_bus, t_bus, angmin, angmax):
        pair = (int(f), int(t))
        old = pair_values.get(pair, (-np.inf, np.inf))
        pair_values[pair] = (max(old[0], float(amin)), min(old[1], float(amax)))
    pairs = sorted(pair_values)
    angle_pairs = np.asarray(pairs, dtype=int).reshape((-1, 2))
    angle_min = np.asarray([pair_values[p][0] for p in pairs])
    angle_max = np.asarray([pair_values[p][1] for p in pairs])

    refs = np.flatnonzero(bus[:, BUS_TYPE] == REF)
    refs = _connected_references(len(bus), f_bus, t_bus, refs, gen_bus)
    incidence = _incidence_lists(len(bus), gen_bus, f_bus, t_bus)
    dc_incidence = _edge_incidence_lists(len(bus), dc_f_bus, dc_t_bus)
    return PowerNetwork(
        source=data,
        bus_external=bus_all,
        gen_external=gen_all,
        branch_external=branch_all,
        dcline_external=dcline_all,
        base_mva=base,
        bus=bus,
        gen=gen,
        branch=branch,
        gencost=gencost,
        dcline=dcline,
        dclinecost=dclinecost,
        bus_rows=bus_rows,
        gen_rows=gen_rows,
        branch_rows=branch_rows,
        dcline_rows=dcline_rows,
        gen_bus=gen_bus,
        f_bus=f_bus,
        t_bus=t_bus,
        dc_f_bus=dc_f_bus,
        dc_t_bus=dc_t_bus,
        dc_pmin_from=dc_bounds[0],
        dc_pmax_from=dc_bounds[1],
        dc_pmin_to=dc_bounds[2],
        dc_pmax_to=dc_bounds[3],
        dc_qmin_from=dc_qmin_from,
        dc_qmax_from=dc_qmax_from,
        dc_qmin_to=dc_qmin_to,
        dc_qmax_to=dc_qmax_to,
        dc_loss0=dc_bounds[4],
        dc_loss1=dc_bounds[5],
        refs=refs,
        angle_pairs=angle_pairs,
        angle_min=angle_min,
        angle_max=angle_max,
        branch_angle_min=angmin,
        branch_angle_max=angmax,
        rate=rate,
        branch_reversed=branch_reversed,
        tap=tap,
        shift=shift,
        b_fr=b_fr,
        b_to=b_to,
        admittance=_branch_admittance(branch, tap, shift, b_fr, b_to),
        generators_at_bus=incidence[0],
        from_branches_at_bus=incidence[1],
        to_branches_at_bus=incidence[2],
        from_dclines_at_bus=dc_incidence[0],
        to_dclines_at_bus=dc_incidence[1],
    )


__all__ = ["BranchAdmittance", "PowerNetwork", "case_dict", "pad_matrix", "prepare_network"]
