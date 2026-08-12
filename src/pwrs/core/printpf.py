# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import sys
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..corex import MatpowerCase
from .get_losses import get_losses_full
from .idx_brch import (
    BR_B,
    BR_R,
    BR_STATUS,
    F_BUS,
    MU_SF,
    MU_ST,
    PF,
    PT,
    QF,
    QT,
    RATE_A,
    T_BUS,
    TAP,
)
from .idx_bus import (
    BS,
    BUS_AREA,
    BUS_I,
    BUS_TYPE,
    GS,
    LAM_P,
    LAM_Q,
    MU_VMAX,
    MU_VMIN,
    NONE,
    PD,
    QD,
    REF,
    VA,
    VM,
    VMAX,
    VMIN,
)
from .idx_gen import (
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
)
from .isload import isload
from .mpoption import mpoption
from .run_userfcn import run_userfcn
from .total_load import total_load_pq


def _format_optional_mu(value: float, condition: bool) -> str:
    return f"{value:10.3f}" if condition else "      -   "


def _build_e2i(bus: np.ndarray) -> dict[int, int]:
    i2e = bus[:, BUS_I].astype(int)
    return {ext: idx for idx, ext in enumerate(i2e)}


def _map_e2i(e2i: dict[int, int], bus_numbers: np.ndarray) -> np.ndarray:
    bus_numbers = np.asarray(bus_numbers, dtype=int).reshape(-1)
    return np.fromiter((e2i[int(bus)] for bus in bus_numbers), dtype=int, count=bus_numbers.size)


def _parse_param(args, index):
    if len(args) > index:
        return args[index]
    else:
        return None


def _parse_inputs(args: tuple[Any, ...]):
    if not args:
        raise TypeError("printpf: missing required inputs")

    if isinstance(args[0], (Mapping, MatpowerCase)):
        # first argument is a results struct
        results = args[0]
        fd = _parse_param(args, 1)
        if fd is None:
            fd = sys.stdout
        mpopt = _parse_param(args, 2)
        if mpopt is None:
            mpopt = mpoption()
        else:
            mpopt = mpoption(mpopt)
        f = results["f"] if "f" in results else np.array([])
        if f is None:
            f = np.array([])
        return (
            True,
            results,
            results["baseMVA"],
            results["bus"],
            results["gen"],
            results["branch"],
            f,
            bool(results["success"]),
            results["et"] if "et" in results else 0.0,
            fd,
            mpopt,
        )

    padded = list(args) + [None] * max(0, 9 - len(args))
    baseMVA, bus, gen, branch, f, success, et, fd, mpopt_arg = padded[:9]
    if fd is None:
        fd = sys.stdout
    if mpopt_arg is None:
        mpopt = mpoption()
    else:
        mpopt = mpoption(mpopt_arg)
    return (
        False,
        None,
        baseMVA,
        bus,
        gen,
        branch,
        np.array([]) if f is None else f,
        bool(success),
        et,
        fd,
        mpopt,
    )


def _append(lines: list[str], text: str = "") -> None:
    lines.append(text)


def printpf(*args: Any, nargout: int | None = None):
    """Pretty-print MATPOWER power flow or OPF results.

    It accepts either a results struct and a fd,
    or the expanded ``baseMVA, bus, gen, branch, ...`` argument form,
    then formats the standard system summary, bus, generator, branch, and
    limit information for stdout output.

    Parameters
    ----------
    *args
        MATPOWER-style ``printpf`` inputs, either as a results struct or as
        the expanded numeric argument list.
    nargout : int, optional
        MATLAB compatibility flag. ``printpf`` does not return values.

    Examples
    --------

    The following calls are supported::

        import io
        buf = io.StringIO()
        printpf(results)
        printpf(results, buf)
        printpf(results, buf, mpopt)
        printpf(baseMVA, bus, gen, branch, f, success, et)
        printpf(baseMVA, bus, gen, branch, f, success, et, buf)
        printpf(baseMVA, bus, gen, branch, f, success, et, buf, mpopt)
        result = buf.getvalue()

    Returns
    -------
    None
        This function prints formatted output and does not return a value.
    """
    if nargout not in (None, 0):
        raise ValueError("printpf: no outputs expected")

    have_results_struct, results, baseMVA, bus, gen, branch, f, success, et, fd, mpopt = _parse_inputs(args)

    is_opf = np.asarray(f).size > 0
    is_dc = mpopt.model.upper() == "DC"

    suppress = mpopt.out.suppress_detail
    if suppress == -1:
        suppress = 1.0 if bus.shape[0] > 500 else 0.0

    out_all = mpopt.out.all
    out_force = mpopt.out.force
    out_any = out_all == 1
    out_sys_sum = out_all == 1 or (out_all == -1 and bool(mpopt.out.sys_sum))
    out_area_sum = out_all == 1 or (out_all == -1 and not suppress and bool(mpopt.out.area_sum))
    out_bus = out_all == 1 or (out_all == -1 and not suppress and bool(mpopt.out.bus))
    out_branch = out_all == 1 or (out_all == -1 and not suppress and bool(mpopt.out.branch))
    out_gen = out_all == 1 or (out_all == -1 and not suppress and bool(mpopt.out.gen))
    out_any = out_any or (out_all == -1 and (out_sys_sum or out_area_sum or out_bus or out_branch or out_gen))

    if out_all == -1:
        out_all_lim = (0 if suppress else 1) * mpopt.out.lim.all
    elif out_all == 1:
        out_all_lim = 2
    else:
        out_all_lim = 0
    out_any = out_any or out_all_lim >= 1
    if out_all_lim == -1:
        mult = 0 if suppress else 1
        out_v_lim = mult * mpopt.out.lim.v
        out_line_lim = mult * mpopt.out.lim.line
        out_pg_lim = mult * mpopt.out.lim.pg
        out_qg_lim = mult * mpopt.out.lim.qg
    else:
        out_v_lim = out_all_lim
        out_line_lim = out_all_lim
        out_pg_lim = out_all_lim
        out_qg_lim = out_all_lim
    out_any = out_any or (out_all_lim == -1 and (out_v_lim or out_line_lim or out_pg_lim or out_qg_lim))

    ptol = 1e-4
    is_sdp = False
    mineigratio = np.array([])
    zero_eval = np.array([])
    if out_any and is_opf and (not is_dc) and mpopt.opf.ac.solver.upper() == "SDPOPF":
        is_sdp = True
        ptol = 0.1
        if have_results_struct and isinstance(results, Mapping):
            mineigratio = results.get("mineigratio", np.array([]))
            zero_eval = results.get("zero_eval", np.array([]))

    e2i = _build_e2i(bus)
    nb = bus.shape[0]
    nl = branch.shape[0]

    if is_dc:
        bus = bus.copy()
        gen = gen.copy()
        branch = branch.copy()
        bus[:, [QD, BS]] = 0
        gen[:, [QG, QMAX, QMIN]] = 0
        branch[:, [BR_R, BR_B]] = 0

    branch_f_idx = _map_e2i(e2i, branch[:, F_BUS])
    branch_t_idx = _map_e2i(e2i, branch[:, T_BUS])
    gen_bus_idx = _map_e2i(e2i, gen[:, GEN_BUS])
    ties = np.flatnonzero(bus[branch_f_idx, BUS_AREA] != bus[branch_t_idx, BUS_AREA])
    xfmr = np.flatnonzero(branch[:, TAP] != 0)
    nzld = np.flatnonzero(((bus[:, PD] != 0) | (bus[:, QD] != 0)) & (bus[:, BUS_TYPE] != NONE))
    sorted_areas = np.sort(bus[:, BUS_AREA].astype(int))
    s_areas = (
        sorted_areas[np.r_[0, np.flatnonzero(np.diff(sorted_areas)) + 1]]
        if sorted_areas.size
        else np.array([], dtype=int)
    )
    nzsh = np.flatnonzero(((bus[:, GS] != 0) | (bus[:, BS] != 0)) & (bus[:, BUS_TYPE] != NONE))
    isload_mask = isload(gen)
    allg = np.flatnonzero(~isload_mask)
    alld = np.flatnonzero(isload_mask)
    ong = np.flatnonzero((gen[:, GEN_STATUS] > 0) & (~isload_mask))
    onld = np.flatnonzero((gen[:, GEN_STATUS] > 0) & isload_mask)
    V = bus[:, VM] * np.exp(1j * np.pi / 180 * bus[:, VA])
    Pdf, Qdf = total_load_pq(bus, gen, "bus", {"type": "FIXED"}, mpopt)
    Pdd, Qdd = total_load_pq(bus, gen, "bus", {"type": "DISPATCHABLE"}, mpopt)
    Pdf = Pdf.reshape(-1)
    Qdf = Qdf.reshape(-1)
    Pdd = Pdd.reshape(-1)
    Qdd = Qdd.reshape(-1)
    if is_dc:
        loss = np.zeros(nl, dtype=complex)
        fchg = np.zeros(nl)
        tchg = np.zeros(nl)
    else:
        loss, fchg, tchg, _, _ = get_losses_full(baseMVA, bus, branch)

    lines: list[str] = []
    if out_any:
        if success:
            if is_sdp:
                _append(
                    lines,
                    f"\nSolution satisfies rank and consistency conditions, {et:.2f} seconds.\nmineigratio = {mineigratio:0.5g}, zero_eval = {zero_eval:0.5g}",
                )
            else:
                _append(lines, f"\nConverged in {et:.2f} seconds")
        else:
            if is_sdp:
                _append(
                    lines,
                    f"\n>>>>>  Solution does NOT satisfy rank and/or consistency conditions ({et:.2f} seconds).  <<<<<\nmineigratio = {mineigratio:0.5g}, zero_eval = {zero_eval:0.5g}\n",
                )
            else:
                _append(lines, f"\n>>>>>  Did NOT converge ({et:.2f} seconds)  <<<<<")
        if is_opf and (success or out_force):
            _append(lines, f"Objective Function Value = {f:.2f} $/hr")

    if out_sys_sum and (success or out_force):
        _append(lines, "================================================================================")
        _append(lines, "|     System Summary                                                           |")
        _append(lines, "================================================================================")
        _append(lines, "")
        _append(lines, "How many?                How much?              P (MW)            Q (MVAr)")
        _append(lines, "---------------------    -------------------  -------------  -----------------")
        _append(
            lines,
            "Buses         %6d     Total Gen Capacity   %7.1f       %7.1f to %.1f"
            % (nb, np.sum(gen[allg, PMAX]), np.sum(gen[allg, QMIN]), np.sum(gen[allg, QMAX])),
        )
        _append(
            lines,
            "Generators     %5d     On-line Capacity     %7.1f       %7.1f to %.1f"
            % (len(allg), np.sum(gen[ong, PMAX]), np.sum(gen[ong, QMIN]), np.sum(gen[ong, QMAX])),
        )
        _append(
            lines,
            "Committed Gens %5d     Generation (actual)  %7.1f           %7.1f"
            % (len(ong), np.sum(gen[ong, PG]), np.sum(gen[ong, QG])),
        )
        _append(
            lines,
            "Loads          %5d     Load                 %7.1f           %7.1f"
            % (
                len(nzld) + len(onld),
                np.sum(Pdf[nzld]) - np.sum(gen[onld, PG]),
                np.sum(Qdf[nzld]) - np.sum(gen[onld, QG]),
            ),
        )
        _append(
            lines,
            "  Fixed        %5d       Fixed              %7.1f           %7.1f"
            % (len(nzld), np.sum(Pdf[nzld]), np.sum(Qdf[nzld])),
        )
        _append(
            lines,
            "  Dispatchable %5d       Dispatchable       %7.1f of %-7.1f%7.1f"
            % (len(onld), -np.sum(gen[onld, PG]), -np.sum(gen[onld, PMIN]), -np.sum(gen[onld, QG])),
        )
        _append(
            lines,
            "Shunts         %5d     Shunt (inj)          %7.1f           %7.1f"
            % (
                len(nzsh),
                -np.sum(bus[nzsh, VM] ** 2 * bus[nzsh, GS]),
                np.sum(bus[nzsh, VM] ** 2 * bus[nzsh, BS]),
            ),
        )
        _append(
            lines,
            "Branches       %5d     Losses (I^2 * Z)     %8.2f          %8.2f"
            % (nl, np.sum(np.real(loss)), np.sum(np.imag(loss))),
        )
        _append(
            lines,
            "Transformers   %5d     Branch Charging (inj)     -            %7.1f"
            % (len(xfmr), np.sum(fchg) + np.sum(tchg)),
        )
        _append(
            lines,
            "Inter-ties     %5d     Total Inter-tie Flow %7.1f           %7.1f"
            % (
                len(ties),
                np.sum(np.abs(branch[ties, PF] - branch[ties, PT])) / 2,
                np.sum(np.abs(branch[ties, QF] - branch[ties, QT])) / 2,
            ),
        )
        _append(lines, "Areas          %5d" % len(s_areas))
        _append(lines, "")
        _append(lines, "                          Minimum                      Maximum")
        _append(lines, "                 -------------------------  --------------------------------")
        min_vm_i = int(np.argmin(bus[:, VM]))
        max_vm_i = int(np.argmax(bus[:, VM]))
        _append(
            lines,
            "Voltage Magnitude %7.3f p.u. @ bus %-4d     %7.3f p.u. @ bus %-4d"
            % (
                bus[min_vm_i, VM],
                int(bus[min_vm_i, BUS_I]),
                bus[max_vm_i, VM],
                int(bus[max_vm_i, BUS_I]),
            ),
        )
        min_va_i = int(np.argmin(bus[:, VA]))
        max_va_i = int(np.argmax(bus[:, VA]))
        _append(
            lines,
            "Voltage Angle   %8.2f deg   @ bus %-4d   %8.2f deg   @ bus %-4d"
            % (
                bus[min_va_i, VA],
                int(bus[min_va_i, BUS_I]),
                bus[max_va_i, VA],
                int(bus[max_va_i, BUS_I]),
            ),
        )
        if not is_dc:
            max_pl_i = int(np.argmax(np.real(loss)))
            max_ql_i = int(np.argmax(np.imag(loss)))
            _append(
                lines,
                "P Losses (I^2*R)             -              %8.2f MW    @ line %d-%d"
                % (np.real(loss[max_pl_i]), int(branch[max_pl_i, F_BUS]), int(branch[max_pl_i, T_BUS])),
            )
            _append(
                lines,
                "Q Losses (I^2*X)             -              %8.2f MVAr  @ line %d-%d"
                % (np.imag(loss[max_ql_i]), int(branch[max_ql_i, F_BUS]), int(branch[max_ql_i, T_BUS])),
            )
        if is_opf:
            min_lp_i = int(np.argmin(bus[:, LAM_P]))
            max_lp_i = int(np.argmax(bus[:, LAM_P]))
            _append(
                lines,
                "Lambda P        %8.2f $/MWh @ bus %-4d   %8.2f $/MWh @ bus %-4d"
                % (
                    bus[min_lp_i, LAM_P],
                    int(bus[min_lp_i, BUS_I]),
                    bus[max_lp_i, LAM_P],
                    int(bus[max_lp_i, BUS_I]),
                ),
            )
            min_lq_i = int(np.argmin(bus[:, LAM_Q]))
            max_lq_i = int(np.argmax(bus[:, LAM_Q]))
            _append(
                lines,
                "Lambda Q        %8.2f $/MWh @ bus %-4d   %8.2f $/MWh @ bus %-4d"
                % (
                    bus[min_lq_i, LAM_Q],
                    int(bus[min_lq_i, BUS_I]),
                    bus[max_lq_i, LAM_Q],
                    int(bus[max_lq_i, BUS_I]),
                ),
            )
        _append(lines, "")

    if out_area_sum and (success or out_force):
        _append(lines, "================================================================================")
        _append(lines, "|     Area Summary                                                             |")
        _append(lines, "================================================================================")
        _append(lines, "Area  # of      # of Gens        # of Loads         # of    # of   # of   # of")
        _append(lines, " Num  Buses   Total  Online   Total  Fixed  Disp    Shunt   Brchs  Xfmrs   Ties")
        _append(lines, "----  -----   -----  ------   -----  -----  -----   -----   -----  -----  -----")
        for a in s_areas:
            ib = np.flatnonzero(bus[:, BUS_AREA] == a)
            gen_areas = bus[gen_bus_idx, BUS_AREA]
            ig = np.flatnonzero((gen_areas == a) & (~isload_mask))
            igon = np.flatnonzero((gen_areas == a) & (gen[:, GEN_STATUS] > 0) & (~isload_mask))
            ildon = np.flatnonzero((gen_areas == a) & (gen[:, GEN_STATUS] > 0) & isload_mask)
            inzld = np.flatnonzero(
                (bus[:, BUS_AREA] == a) & ((Pdf != 0) | (Qdf != 0)) & (bus[:, BUS_TYPE] != NONE)
            )
            inzsh = np.flatnonzero(
                (bus[:, BUS_AREA] == a)
                & ((bus[:, GS] != 0) | (bus[:, BS] != 0))
                & (bus[:, BUS_TYPE] != NONE)
            )
            from_area = bus[branch_f_idx, BUS_AREA]
            to_area = bus[branch_t_idx, BUS_AREA]
            ibrch = np.flatnonzero((from_area == a) & (to_area == a))
            in_tie = np.flatnonzero((from_area == a) & (to_area != a))
            out_tie = np.flatnonzero((from_area != a) & (to_area == a))
            nxfmr = 0 if xfmr.size == 0 else len(np.flatnonzero((from_area[xfmr] == a) & (to_area[xfmr] == a)))
            _append(
                lines,
                "%3d  %6d   %5d  %5d   %5d  %5d  %5d   %5d   %5d  %5d  %5d"
                % (
                    a,
                    len(ib),
                    len(ig),
                    len(igon),
                    len(inzld) + len(ildon),
                    len(inzld),
                    len(ildon),
                    len(inzsh),
                    len(ibrch),
                    nxfmr,
                    len(in_tie) + len(out_tie),
                ),
            )
        _append(lines, "----  -----   -----  ------   -----  -----  -----   -----   -----  -----  -----")
        _append(
            lines,
            "Tot: %6d   %5d  %5d   %5d  %5d  %5d   %5d   %5d  %5d  %5d"
            % (
                nb,
                len(allg),
                len(ong),
                len(nzld) + len(onld),
                len(nzld),
                len(onld),
                len(nzsh),
                nl,
                len(xfmr),
                len(ties),
            ),
        )
        _append(lines, "")
        _append(lines, "Area      Total Gen Capacity           On-line Gen Capacity         Generation")
        _append(lines, " Num     MW           MVAr            MW           MVAr             MW    MVAr")
        _append(lines, "----   ------  ------------------   ------  ------------------    ------  ------")
        for a in s_areas:
            gen_areas = bus[gen_bus_idx, BUS_AREA]
            ig = np.flatnonzero((gen_areas == a) & (~isload_mask))
            igon = np.flatnonzero((gen_areas == a) & (gen[:, GEN_STATUS] > 0) & (~isload_mask))
            _append(
                lines,
                "%3d   %7.1f  %7.1f to %-7.1f  %7.1f  %7.1f to %-7.1f   %7.1f %7.1f"
                % (
                    a,
                    np.sum(gen[ig, PMAX]),
                    np.sum(gen[ig, QMIN]),
                    np.sum(gen[ig, QMAX]),
                    np.sum(gen[igon, PMAX]),
                    np.sum(gen[igon, QMIN]),
                    np.sum(gen[igon, QMAX]),
                    np.sum(gen[igon, PG]),
                    np.sum(gen[igon, QG]),
                ),
            )
        _append(lines, "----   ------  ------------------   ------  ------------------    ------  ------")
        _append(
            lines,
            "Tot:  %7.1f  %7.1f to %-7.1f  %7.1f  %7.1f to %-7.1f   %7.1f %7.1f"
            % (
                np.sum(gen[allg, PMAX]),
                np.sum(gen[allg, QMIN]),
                np.sum(gen[allg, QMAX]),
                np.sum(gen[ong, PMAX]),
                np.sum(gen[ong, QMIN]),
                np.sum(gen[ong, QMAX]),
                np.sum(gen[ong, PG]),
                np.sum(gen[ong, QG]),
            ),
        )
        _append(lines, "")
        _append(lines, "Area    Disp Load Cap       Disp Load         Fixed Load        Total Load")
        _append(lines, " Num      MW     MVAr       MW     MVAr       MW     MVAr       MW     MVAr")
        _append(lines, "----    ------  ------    ------  ------    ------  ------    ------  ------")
        qlim = (gen[:, QMIN] == 0) * gen[:, QMAX] + (gen[:, QMAX] == 0) * gen[:, QMIN]
        gen_areas = bus[gen_bus_idx, BUS_AREA]
        for a in s_areas:
            ildon = np.flatnonzero((gen_areas == a) & (gen[:, GEN_STATUS] > 0) & isload_mask)
            inzld = np.flatnonzero((bus[:, BUS_AREA] == a) & ((Pdf != 0) | (Qdf != 0)))
            _append(
                lines,
                "%3d    %7.1f %7.1f   %7.1f %7.1f   %7.1f %7.1f   %7.1f %7.1f"
                % (
                    a,
                    -np.sum(gen[ildon, PMIN]),
                    -np.sum(qlim[ildon]),
                    -np.sum(gen[ildon, PG]),
                    -np.sum(gen[ildon, QG]),
                    np.sum(Pdf[inzld]),
                    np.sum(Qdf[inzld]),
                    -np.sum(gen[ildon, PG]) + np.sum(Pdf[inzld]),
                    -np.sum(gen[ildon, QG]) + np.sum(Qdf[inzld]),
                ),
            )
        _append(lines, "----    ------  ------    ------  ------    ------  ------    ------  ------")
        _append(
            lines,
            "Tot:   %7.1f %7.1f   %7.1f %7.1f   %7.1f %7.1f   %7.1f %7.1f"
            % (
                -np.sum(gen[onld, PMIN]),
                -np.sum(qlim[onld]),
                -np.sum(gen[onld, PG]),
                -np.sum(gen[onld, QG]),
                np.sum(Pdf[nzld]),
                np.sum(Qdf[nzld]),
                -np.sum(gen[onld, PG]) + np.sum(Pdf[nzld]),
                -np.sum(gen[onld, QG]) + np.sum(Qdf[nzld]),
            ),
        )
        _append(lines, "")
        _append(lines, "Area      Shunt Inj        Branch      Series Losses      Net Export")
        _append(lines, " Num      MW     MVAr     Charging      MW     MVAr       MW     MVAr")
        _append(lines, "----    ------  ------    --------    ------  ------    ------  ------")
        from_area = bus[branch_f_idx, BUS_AREA]
        to_area = bus[branch_t_idx, BUS_AREA]
        for a in s_areas:
            inzsh = np.flatnonzero((bus[:, BUS_AREA] == a) & ((bus[:, GS] != 0) | (bus[:, BS] != 0)))
            ibrch = np.flatnonzero((from_area == a) & (to_area == a) & (branch[:, BR_STATUS] != 0))
            in_tie = np.flatnonzero((from_area != a) & (to_area == a) & (branch[:, BR_STATUS] != 0))
            out_tie = np.flatnonzero((from_area == a) & (to_area != a) & (branch[:, BR_STATUS] != 0))
            both_ties = np.r_[in_tie, out_tie]
            _append(
                lines,
                "%3d    %7.1f %7.1f    %7.1f    %7.2f %7.2f   %7.1f %7.1f"
                % (
                    a,
                    -np.sum(bus[inzsh, VM] ** 2 * bus[inzsh, GS]),
                    np.sum(bus[inzsh, VM] ** 2 * bus[inzsh, BS]),
                    np.sum(fchg[ibrch]) + np.sum(tchg[ibrch]) + np.sum(fchg[out_tie]) + np.sum(tchg[in_tie]),
                    np.sum(np.real(loss[ibrch])) + np.sum(np.real(loss[both_ties])) / 2,
                    np.sum(np.imag(loss[ibrch])) + np.sum(np.imag(loss[both_ties])) / 2,
                    np.sum(branch[in_tie, PT])
                    + np.sum(branch[out_tie, PF])
                    - np.sum(np.real(loss[both_ties])) / 2,
                    np.sum(branch[in_tie, QT])
                    + np.sum(branch[out_tie, QF])
                    - np.sum(np.imag(loss[both_ties])) / 2,
                ),
            )
        _append(lines, "----    ------  ------    --------    ------  ------    ------  ------")
        _append(
            lines,
            "Tot:   %7.1f %7.1f    %7.1f    %7.2f %7.2f       -       -"
            % (
                -np.sum(bus[nzsh, VM] ** 2 * bus[nzsh, GS]),
                np.sum(bus[nzsh, VM] ** 2 * bus[nzsh, BS]),
                np.sum(fchg) + np.sum(tchg),
                np.sum(np.real(loss)),
                np.sum(np.imag(loss)),
            ),
        )
        _append(lines, "")

    if out_gen and (success or out_force):
        genlamP = np.zeros(gen.shape[0])
        genlamQ = np.zeros(gen.shape[0])
        if is_opf:
            genlamP = bus[gen_bus_idx, LAM_P]
            genlamQ = bus[gen_bus_idx, LAM_Q]
        _append(lines, "================================================================================")
        _append(lines, "|     Generator Data                                                           |")
        _append(lines, "================================================================================")
        hdr = " Gen   Bus   Status     Pg        Qg   "
        if is_opf:
            hdr += "   Lambda ($/MVA-hr)"
        _append(lines, hdr)
        hdr = "  #     #              (MW)     (MVAr) "
        if is_opf:
            hdr += "     P         Q    "
        _append(lines, hdr)
        hdr = "----  -----  ------  --------  --------"
        if is_opf:
            hdr += "  --------  --------"
        _append(lines, hdr)
        for i in allg:
            row = "%3d %6d     %2d " % (i + 1, int(gen[i, GEN_BUS]), int(gen[i, GEN_STATUS]))
            if gen[i, GEN_STATUS] > 0 and (gen[i, PG] != 0 or gen[i, QG] != 0):
                row += "%10.2f%10.2f" % (gen[i, PG], gen[i, QG])
            else:
                row += "       -         -  "
            if is_opf:
                row += "%10.2f%10.2f" % (genlamP[i], genlamQ[i])
            _append(lines, row)
        _append(lines, "                     --------  --------")
        _append(lines, "            Total: %9.2f%10.2f" % (np.sum(gen[ong, PG]), np.sum(gen[ong, QG])))
        _append(lines, "")
        if alld.size:
            _append(lines, "================================================================================")
            _append(lines, "|     Dispatchable Load Data                                                   |")
            _append(lines, "================================================================================")
            hdr = " Gen   Bus   Status     Pd        Qd   "
            if is_opf:
                hdr += "   Lambda ($/MVA-hr)"
            _append(lines, hdr)
            hdr = "  #     #              (MW)     (MVAr) "
            if is_opf:
                hdr += "     P         Q    "
            _append(lines, hdr)
            hdr = "----  -----  ------  --------  --------"
            if is_opf:
                hdr += "  --------  --------"
            _append(lines, hdr)
            for i in alld:
                row = "%3d %6d     %2d " % (i + 1, int(gen[i, GEN_BUS]), int(gen[i, GEN_STATUS]))
                if gen[i, GEN_STATUS] > 0 and (gen[i, PG] != 0 or gen[i, QG] != 0):
                    row += "%10.2f%10.2f" % (-gen[i, PG], -gen[i, QG])
                else:
                    row += "       -         -  "
                if is_opf:
                    row += "%10.2f%10.2f" % (genlamP[i], genlamQ[i])
                _append(lines, row)
            _append(lines, "                     --------  --------")
            _append(lines, "            Total: %9.2f%10.2f" % (-np.sum(gen[onld, PG]), -np.sum(gen[onld, QG])))
            _append(lines, "")

    if out_bus and (success or out_force):
        _append(lines, "================================================================================")
        _append(lines, "|     Bus Data                                                                 |")
        _append(lines, "================================================================================")
        hdr = " Bus      Voltage          Generation             Load        "
        if is_opf:
            hdr += "  Lambda($/MVA-hr)"
        _append(lines, hdr)
        hdr = "  #   Mag(pu) Ang(deg)   P (MW)   Q (MVAr)   P (MW)   Q (MVAr)"
        if is_opf:
            hdr += "     P        Q   "
        _append(lines, hdr)
        hdr = "----- ------- --------  --------  --------  --------  --------"
        if is_opf:
            hdr += "  -------  -------"
        _append(lines, hdr)
        for i in range(nb):
            row = "%5d%7.3f%9.3f" % (int(bus[i, BUS_I]), bus[i, VM], bus[i, VA])
            if bus[i, BUS_TYPE] == REF:
                row += "*"
            elif bus[i, BUS_TYPE] == NONE:
                row += "x"
            else:
                row += " "
            g = np.flatnonzero(
                (gen[:, GEN_STATUS] > 0) & (gen[:, GEN_BUS] == bus[i, BUS_I]) & (~isload_mask)
            )
            ld = np.flatnonzero((gen[:, GEN_STATUS] > 0) & (gen[:, GEN_BUS] == bus[i, BUS_I]) & isload_mask)
            if g.size:
                row += "%9.2f%10.2f" % (np.sum(gen[g, PG]), np.sum(gen[g, QG]))
            else:
                row += "      -         -  "
            if Pdf[i] != 0 or Qdf[i] != 0 or ld.size:
                if ld.size:
                    row += "%10.2f*%9.2f*" % (Pdf[i] - np.sum(gen[ld, PG]), Qdf[i] - np.sum(gen[ld, QG]))
                else:
                    row += "%10.2f%10.2f " % (Pdf[i], Qdf[i])
            else:
                row += "       -         -   "
            if is_opf:
                row += "%9.3f" % bus[i, LAM_P]
                if abs(bus[i, LAM_Q]) > ptol:
                    row += "%8.3f" % bus[i, LAM_Q]
                else:
                    row += "     -"
            _append(lines, row)
        _append(lines, "                        --------  --------  --------  --------")
        _append(
            lines,
            "               Total: %9.2f %9.2f %9.2f %9.2f"
            % (
                np.sum(gen[ong, PG]),
                np.sum(gen[ong, QG]),
                np.sum(Pdf[nzld]) - np.sum(gen[onld, PG]),
                np.sum(Qdf[nzld]) - np.sum(gen[onld, QG]),
            ),
        )
        _append(lines, "")

    if out_branch and (success or out_force):
        _append(lines, "================================================================================")
        _append(lines, "|     Branch Data                                                              |")
        _append(lines, "================================================================================")
        _append(lines, "Brnch   From   To    From Bus Injection   To Bus Injection     Loss (I^2 * Z)  ")
        _append(lines, "  #     Bus    Bus    P (MW)   Q (MVAr)   P (MW)   Q (MVAr)   P (MW)   Q (MVAr)")
        _append(lines, "-----  -----  -----  --------  --------  --------  --------  --------  --------")
        for i in range(nl):
            _append(
                lines,
                "%4d%7d%7d%10.2f%10.2f%10.2f%10.2f%10.3f%10.2f"
                % (
                    i + 1,
                    int(branch[i, F_BUS]),
                    int(branch[i, T_BUS]),
                    branch[i, PF],
                    branch[i, QF],
                    branch[i, PT],
                    branch[i, QT],
                    float(np.real(loss[i])),
                    float(np.imag(loss[i])),
                ),
            )
        _append(lines, "                                                             --------  --------")
        _append(
            lines,
            "                                                    Total:%10.3f%10.2f"
            % (np.sum(np.real(loss)), np.sum(np.imag(loss))),
        )
        _append(lines, "")

    if out_any and is_opf and (success or out_force):
        ctol = mpopt.opf.violation
        if (not is_dc) and (
            out_v_lim == 2
            or (
                out_v_lim == 1
                and (
                    np.any(bus[:, VM] < bus[:, VMIN] + ctol)
                    or np.any(bus[:, VM] > bus[:, VMAX] - ctol)
                    or np.any(bus[:, MU_VMIN] > ptol)
                    or np.any(bus[:, MU_VMAX] > ptol)
                )
            )
        ):
            _append(lines, "================================================================================")
            _append(lines, "|     Voltage Constraints                                                      |")
            _append(lines, "================================================================================")
            _append(lines, "Bus #  Vmin mu    Vmin    |V|   Vmax    Vmax mu")
            _append(lines, "-----  --------   -----  -----  -----   --------")
            for i in range(nb):
                cond = out_v_lim == 2 or (
                    out_v_lim == 1
                    and (
                        bus[i, VM] < bus[i, VMIN] + ctol
                        or bus[i, VM] > bus[i, VMAX] - ctol
                        or bus[i, MU_VMIN] > ptol
                        or bus[i, MU_VMAX] > ptol
                    )
                )
                if cond:
                    row = f"{int(bus[i, BUS_I]):5d}"
                    row += _format_optional_mu(
                        bus[i, MU_VMIN], bus[i, VM] < bus[i, VMIN] + ctol or bus[i, MU_VMIN] > ptol
                    )
                    row += f"{bus[i, VMIN]:8.3f}{bus[i, VM]:7.3f}{bus[i, VMAX]:7.3f}"
                    row += (
                        f"{bus[i, MU_VMAX]:10.3f}"
                        if (bus[i, VM] > bus[i, VMAX] - ctol or bus[i, MU_VMAX] > ptol)
                        else "      -    "
                    )
                    _append(lines, row)
            _append(lines, "")

        gen_p_lim_hit = out_pg_lim == 2 or (
            out_pg_lim == 1
            and (
                np.any(gen[ong, PG] < gen[ong, PMIN] + ctol)
                or np.any(gen[ong, PG] > gen[ong, PMAX] - ctol)
                or np.any(gen[ong, MU_PMIN] > ptol)
                or np.any(gen[ong, MU_PMAX] > ptol)
            )
        )
        gen_q_lim_hit = (not is_dc) and (
            out_qg_lim == 2
            or (
                out_qg_lim == 1
                and (
                    np.any(gen[ong, QG] < gen[ong, QMIN] + ctol)
                    or np.any(gen[ong, QG] > gen[ong, QMAX] - ctol)
                    or np.any(gen[ong, MU_QMIN] > ptol)
                    or np.any(gen[ong, MU_QMAX] > ptol)
                )
            )
        )
        if gen_p_lim_hit or gen_q_lim_hit:
            _append(lines, "================================================================================")
            _append(lines, "|     Generation Constraints                                                   |")
            _append(lines, "================================================================================")
        if gen_p_lim_hit:
            _append(lines, " Gen   Bus                  Active Power Limits")
            _append(lines, "  #     #     Pmin mu     Pmin       Pg       Pmax    Pmax mu")
            _append(lines, "----  -----   -------   --------  --------  --------  -------")
            for i in ong:
                cond = out_pg_lim == 2 or (
                    out_pg_lim == 1
                    and (
                        gen[i, PG] < gen[i, PMIN] + ctol
                        or gen[i, PG] > gen[i, PMAX] - ctol
                        or gen[i, MU_PMIN] > ptol
                        or gen[i, MU_PMAX] > ptol
                    )
                )
                if cond:
                    row = "%4d%6d " % (i + 1, int(gen[i, GEN_BUS]))
                    row += _format_optional_mu(
                        gen[i, MU_PMIN], gen[i, PG] < gen[i, PMIN] + ctol or gen[i, MU_PMIN] > ptol
                    )
                    row += (
                        ("%10.2f%10.2f%10.2f" % (gen[i, PMIN], gen[i, PG], gen[i, PMAX]))
                        if gen[i, PG] != 0
                        else ("%10.2f       -  %10.2f" % (gen[i, PMIN], gen[i, PMAX]))
                    )
                    row += (
                        f"{gen[i, MU_PMAX]:10.3f}"
                        if (gen[i, PG] > gen[i, PMAX] - ctol or gen[i, MU_PMAX] > ptol)
                        else "      -   "
                    )
                    _append(lines, row)
            _append(lines, "")
        if gen_q_lim_hit:
            _append(lines, " Gen   Bus                 Reactive Power Limits")
            _append(lines, "  #     #     Qmin mu     Qmin       Qg       Qmax    Qmax mu")
            _append(lines, "----  -----   -------   --------  --------  --------  -------")
            for i in ong:
                cond = out_qg_lim == 2 or (
                    out_qg_lim == 1
                    and (
                        gen[i, QG] < gen[i, QMIN] + ctol
                        or gen[i, QG] > gen[i, QMAX] - ctol
                        or gen[i, MU_QMIN] > ptol
                        or gen[i, MU_QMAX] > ptol
                    )
                )
                if cond:
                    row = "%4d%6d " % (i + 1, int(gen[i, GEN_BUS]))
                    row += _format_optional_mu(
                        gen[i, MU_QMIN], gen[i, QG] < gen[i, QMIN] + ctol or gen[i, MU_QMIN] > ptol
                    )
                    row += (
                        ("%10.2f%10.2f%10.2f" % (gen[i, QMIN], gen[i, QG], gen[i, QMAX]))
                        if gen[i, QG] != 0
                        else ("%10.2f       -  %10.2f" % (gen[i, QMIN], gen[i, QMAX]))
                    )
                    row += (
                        f"{gen[i, MU_QMAX]:10.3f}"
                        if (gen[i, QG] > gen[i, QMAX] - ctol or gen[i, MU_QMAX] > ptol)
                        else "      -   "
                    )
                    _append(lines, row)
            _append(lines, "")

        load_p_lim_hit = onld.size and (
            out_pg_lim == 2
            or (
                out_pg_lim == 1
                and (
                    np.any(gen[onld, PG] < gen[onld, PMIN] + ctol)
                    or np.any(gen[onld, PG] > gen[onld, PMAX] - ctol)
                    or np.any(gen[onld, MU_PMIN] > ptol)
                    or np.any(gen[onld, MU_PMAX] > ptol)
                )
            )
        )
        load_q_lim_hit = (
            (not is_dc)
            and onld.size
            and (
                out_qg_lim == 2
                or (
                    out_qg_lim == 1
                    and (
                        np.any(gen[onld, QG] < gen[onld, QMIN] + ctol)
                        or np.any(gen[onld, QG] > gen[onld, QMAX] - ctol)
                        or np.any(gen[onld, MU_QMIN] > ptol)
                        or np.any(gen[onld, MU_QMAX] > ptol)
                    )
                )
            )
        )
        if load_p_lim_hit or load_q_lim_hit:
            _append(lines, "================================================================================")
            _append(lines, "|     Dispatchable Load Constraints                                            |")
            _append(lines, "================================================================================")
        if load_p_lim_hit:
            _append(lines, " Gen   Bus                  Active Power Limits")
            _append(lines, "  #     #     Pmin mu     Pmin       Pg       Pmax    Pmax mu")
            _append(lines, "----  -----   -------   --------  --------  --------  -------")
            for i in onld:
                cond = out_pg_lim == 2 or (
                    out_pg_lim == 1
                    and (
                        gen[i, PG] < gen[i, PMIN] + ctol
                        or gen[i, PG] > gen[i, PMAX] - ctol
                        or gen[i, MU_PMIN] > ptol
                        or gen[i, MU_PMAX] > ptol
                    )
                )
                if cond:
                    row = "%4d%6d " % (i + 1, int(gen[i, GEN_BUS]))
                    row += _format_optional_mu(
                        gen[i, MU_PMIN], gen[i, PG] < gen[i, PMIN] + ctol or gen[i, MU_PMIN] > ptol
                    )
                    row += (
                        ("%10.2f%10.2f%10.2f" % (gen[i, PMIN], gen[i, PG], gen[i, PMAX]))
                        if gen[i, PG] != 0
                        else ("%10.2f       -  %10.2f" % (gen[i, PMIN], gen[i, PMAX]))
                    )
                    row += (
                        f"{gen[i, MU_PMAX]:10.3f}"
                        if (gen[i, PG] > gen[i, PMAX] - ctol or gen[i, MU_PMAX] > ptol)
                        else "      -   "
                    )
                    _append(lines, row)
            _append(lines, "")
        if load_q_lim_hit:
            _append(lines, " Gen   Bus                 Reactive Power Limits")
            _append(lines, "  #     #     Qmin mu     Qmin       Qg       Qmax    Qmax mu")
            _append(lines, "----  -----   -------   --------  --------  --------  -------")
            for i in onld:
                cond = out_qg_lim == 2 or (
                    out_qg_lim == 1
                    and (
                        gen[i, QG] < gen[i, QMIN] + ctol
                        or gen[i, QG] > gen[i, QMAX] - ctol
                        or gen[i, MU_QMIN] > ptol
                        or gen[i, MU_QMAX] > ptol
                    )
                )
                if cond:
                    row = "%4d%6d " % (i + 1, int(gen[i, GEN_BUS]))
                    row += _format_optional_mu(
                        gen[i, MU_QMIN], gen[i, QG] < gen[i, QMIN] + ctol or gen[i, MU_QMIN] > ptol
                    )
                    row += (
                        ("%10.2f%10.2f%10.2f" % (gen[i, QMIN], gen[i, QG], gen[i, QMAX]))
                        if gen[i, QG] != 0
                        else ("%10.2f       -  %10.2f" % (gen[i, QMIN], gen[i, QMAX]))
                    )
                    row += (
                        f"{gen[i, MU_QMAX]:10.3f}"
                        if (gen[i, QG] > gen[i, QMAX] - ctol or gen[i, MU_QMAX] > ptol)
                        else "      -   "
                    )
                    _append(lines, row)
            _append(lines, "")

        lim_type = mpopt.opf.flow_lim.upper()[0]
        if is_dc or lim_type in {"P", "2"}:
            Ff = branch[:, PF]
            Ft = branch[:, PT]
            unit_str = "P in MW)       "
            hdr = "  #     Bus    Pf  mu     Pf      |Pmax|      Pt      Pt  mu   Bus"
        elif lim_type == "I":
            Ff = np.abs((branch[:, PF] + 1j * branch[:, QF]) / V[branch_f_idx])
            Ft = np.abs((branch[:, PT] + 1j * branch[:, QT]) / V[branch_t_idx])
            hdr = "  #     Bus   |If| mu    |If|     |Imax|     |It|    |It| mu   Bus"
            unit_str = "I in kA*basekV)"
        else:
            Ff = np.abs(branch[:, PF] + 1j * branch[:, QF])
            Ft = np.abs(branch[:, PT] + 1j * branch[:, QT])
            hdr = "  #     Bus   |Sf| mu    |Sf|     |Smax|     |St|    |St| mu   Bus"
            unit_str = "S in MVA)      "
        if np.any(branch[:, RATE_A] != 0) and (
            out_line_lim == 2
            or (
                out_line_lim == 1
                and (
                    np.any(np.abs(Ff) > branch[:, RATE_A] - ctol)
                    or np.any(np.abs(Ft) > branch[:, RATE_A] - ctol)
                    or np.any(branch[:, MU_SF] > ptol)
                    or np.any(branch[:, MU_ST] > ptol)
                )
            )
        ):
            _append(lines, "================================================================================")
            _append(lines, f"|     Branch Flow Constraints            ({unit_str}                      |")
            _append(lines, "================================================================================")
            _append(lines, 'Brnch   From     "From" End        Limit       "To" End        To')
            _append(lines, hdr)
            _append(lines, "-----  -----  -------  --------  --------  --------  -------  -----")
            for i in range(nl):
                cond = branch[i, RATE_A] != 0 and (
                    out_line_lim == 2
                    or (
                        out_line_lim == 1
                        and (
                            abs(Ff[i]) > branch[i, RATE_A] - ctol
                            or abs(Ft[i]) > branch[i, RATE_A] - ctol
                            or branch[i, MU_SF] > ptol
                            or branch[i, MU_ST] > ptol
                        )
                    )
                )
                if cond:
                    row = "%4d%7d" % (i + 1, int(branch[i, F_BUS]))
                    row += _format_optional_mu(
                        branch[i, MU_SF], Ff[i] > branch[i, RATE_A] - ctol or branch[i, MU_SF] > ptol
                    )
                    row += "%9.2f%10.2f%10.2f" % (Ff[i], branch[i, RATE_A], Ft[i])
                    row += (
                        f"{branch[i, MU_ST]:10.3f}"
                        if (Ft[i] > branch[i, RATE_A] - ctol or branch[i, MU_ST] > ptol)
                        else "      -   "
                    )
                    row += "%6d" % int(branch[i, T_BUS])
                    _append(lines, row)
            _append(lines, "")

    if out_any and (not success):
        if out_force:
            if is_sdp:
                _append(
                    lines,
                    f"\n>>>>>  Solution does NOT satisfy rank and/or consistency conditions ({et:.2f} seconds).  <<<<<\nmineigratio = {mineigratio:0.5g}, zero_eval = {zero_eval:0.5g}\n",
                )
            else:
                _append(lines, f"\n>>>>>  Did NOT converge ({et:.2f} seconds)  <<<<<\n")
        _append(lines, "")

    text = "\n".join(lines)
    if text:
        print(text, end="")
    if have_results_struct and isinstance(results, Mapping) and "userfcn" in results and (success or out_force):
        cb_mpopt = mpopt if is_opf else mpoption(mpopt, "out.lim.all", 0)
        run_userfcn(results["userfcn"], "printpf", results, fd, cb_mpopt)
    return None
