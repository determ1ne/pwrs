# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import math
import time
from typing import Any, TextIO, cast

import numpy as np
from scipy import sparse

from .connected_components import connected_components_full
from .idx_brch import BR_STATUS, F_BUS, PF, PT, QF, QT, T_BUS
from .idx_bus import BS, BUS_I, BUS_TYPE, GS, PD, QD, REF, VM
from .idx_dcline import idx_dcline
from .idx_gen import GEN_BUS, GEN_STATUS, PG, PMAX, PMIN, QG, QMAX, QMIN
from .isload import isload
from .loadcase import loadcase_struct


def _unknown_buses(e2i, nbase, bus_list):
    bus_list = np.asarray(bus_list, dtype=int).reshape(-1).copy()
    nonpos = np.flatnonzero(bus_list <= 0)
    bus_list[nonpos] = -bus_list[nonpos] + nbase
    return np.flatnonzero(e2i[bus_list] == 0) + 1


def _print_row(fd, p, template, name, field):
    templatez = f"%{len(template % 0)}s"
    print(f"{name:<20s}", file=fd)
    if p["page"] == 1:
        if p["total"][field] == 0:
            print(templatez % "-   ", file=fd)
        else:
            print(template % p["total"][field], file=fd)
    for k in p["islands"]:
        if p["d"][k - 1][field] == 0:
            print(templatez % "-   ", file=fd)
        else:
            print(template % p["d"][k - 1][field], file=fd)
    print("\n", file=fd)


def case_info(mpc, fd: TextIO, *, nargout=None):
    """Print structural information about a MATPOWER case.

    Performs the same high-level case inspection as MATPOWER's
    ``case_info``, including connectivity analysis, island detection, and
    summary reporting for buses, branches, generators, loads, and DC lines.

    Parameters
    ----------
    mpc : dict or str
        MATPOWER case struct or case name/path accepted by ``loadcase``.
    fd : file-like or int, optional
        Output destination. ``1`` selects stdout.
    nargout : int, optional
        MATLAB compatibility flag controlling returned outputs.

    Returns
    -------
    tuple or None
        Returns island groups and isolated buses when outputs are requested;
        otherwise prints the report and returns ``None``.
    """
    c = idx_dcline()
    t0 = time.perf_counter()
    mpc = cast(dict[str, Any], loadcase_struct(mpc))

    nb = mpc["bus"].shape[0]
    nl = mpc["branch"].shape[0]
    ndc = mpc["dcline"].shape[0] if "dcline" in mpc else 0
    ng = mpc["gen"].shape[0]

    bus_vals = [
        np.asarray(mpc["bus"][:, BUS_I]).reshape(-1),
        np.asarray(mpc["gen"][:, GEN_BUS]).reshape(-1),
        np.asarray(mpc["branch"][:, F_BUS]).reshape(-1),
        np.asarray(mpc["branch"][:, T_BUS]).reshape(-1),
    ]
    mb = int(np.max(np.abs(np.concatenate(bus_vals))))
    nbase = 10 ** (int(math.floor(math.log10(mb))) + 1)

    bus_i = np.asarray(mpc["bus"][:, BUS_I], dtype=int).reshape(-1).copy()
    nonpos_bus = np.flatnonzero(bus_i <= 0) + 1
    bus_i[nonpos_bus - 1] = -bus_i[nonpos_bus - 1] + nbase
    e2i = np.zeros(nbase + mb + 1, dtype=int)
    e2i[bus_i] = np.arange(1, nb + 1, dtype=int)

    unknown_gbus = _unknown_buses(e2i, nbase, mpc["gen"][:, GEN_BUS])
    unknown_fbus = _unknown_buses(e2i, nbase, mpc["branch"][:, F_BUS])
    unknown_tbus = _unknown_buses(e2i, nbase, mpc["branch"][:, T_BUS])
    if ndc:
        unknown_fbusdc = _unknown_buses(e2i, nbase, mpc["dcline"][:, c["F_BUS"]])
        unknown_tbusdc = _unknown_buses(e2i, nbase, mpc["dcline"][:, c["T_BUS"]])
    else:
        unknown_fbusdc = np.array([], dtype=int)
        unknown_tbusdc = np.array([], dtype=int)

    if len(nonpos_bus):
        print(f"Bad bus numbers:              {len(nonpos_bus)}\n", file=fd)
        for idx in nonpos_bus:
            print(f"{('bus(%d, BUS_I)' % idx):>24s} = {int(mpc['bus'][idx - 1, BUS_I])}\n", file=fd)
    if len(unknown_gbus):
        print(f"Unknown generator buses:      {len(unknown_gbus)}\n", file=fd)
        for idx in unknown_gbus:
            print(f"{('gen(%d, GEN_BUS)' % idx):>24s} = {int(mpc['gen'][idx - 1, GEN_BUS])}\n", file=fd)
        mpc["gen"] = np.delete(mpc["gen"], unknown_gbus - 1, axis=0)
        ng = mpc["gen"].shape[0]
    if len(unknown_fbus):
        print(f'Unknown branch "from" buses:  {len(unknown_fbus)}\n', file=fd)
        for idx in unknown_fbus:
            print(f"{('branch(%d, F_BUS)' % idx):>24s} = {int(mpc['branch'][idx - 1, F_BUS])}\n", file=fd)
    if len(unknown_tbus):
        print(f'Unknown branch "to" buses:    {len(unknown_tbus)}\n', file=fd)
        for idx in unknown_tbus:
            print(f"{('branch(%d, T_BUS)' % idx):>24s} = {int(mpc['branch'][idx - 1, T_BUS])}\n", file=fd)
    if len(unknown_fbus) or len(unknown_tbus):
        tmp = np.unique(np.r_[unknown_fbus, unknown_tbus])
        mpc["branch"] = np.delete(mpc["branch"], tmp - 1, axis=0)
        nl = mpc["branch"].shape[0]
    if ndc:
        if len(unknown_fbusdc):
            print(f'Unknown DC line "from" buses: {len(unknown_fbusdc)}\n', file=fd)
            for idx in unknown_fbusdc:
                print(
                    f"{('dcline(%d, c.F_BUS)' % idx):>24s} = {int(mpc['dcline'][idx - 1, c['F_BUS'] - 1])}\n", file=fd
                )
        if len(unknown_tbusdc):
            print(f'Unknown DC line "to" buses:   {len(unknown_tbusdc)}\n', file=fd)
            for idx in unknown_tbusdc:
                print(
                    f"{('dcline(%d, c.T_BUS)' % idx):>24s} = {int(mpc['dcline'][idx - 1, c['T_BUS'] - 1])}\n", file=fd
                )
        if len(unknown_fbusdc) or len(unknown_tbusdc):
            tmp = np.unique(np.r_[unknown_fbusdc, unknown_tbusdc])
            mpc["dcline"] = np.delete(mpc["dcline"], tmp - 1, axis=0)
            ndc = mpc["dcline"].shape[0]

    groups = []
    isolated = np.array([], dtype=int).reshape(-1, 1)
    if len(nonpos_bus) == 0:
        C_on = sparse.csc_matrix(
            (
                -np.asarray(mpc["branch"][:, BR_STATUS]).reshape(-1),
                (np.arange(nl), e2i[np.asarray(mpc["branch"][:, F_BUS], dtype=int)] - 1),
            ),
            shape=(nl, nb),
        ) + sparse.csc_matrix(
            (
                np.asarray(mpc["branch"][:, BR_STATUS]).reshape(-1),
                (np.arange(nl), e2i[np.asarray(mpc["branch"][:, T_BUS], dtype=int)] - 1),
            ),
            shape=(nl, nb),
        )
        C = sparse.csc_matrix(
            (-np.ones(nl), (np.arange(nl), e2i[np.asarray(mpc["branch"][:, F_BUS], dtype=int)] - 1)),
            shape=(nl, nb),
        ) + sparse.csc_matrix(
            (np.ones(nl), (np.arange(nl), e2i[np.asarray(mpc["branch"][:, T_BUS], dtype=int)] - 1)),
            shape=(nl, nb),
        )
        if ndc:
            _Cdc_on = sparse.csc_matrix(
                (
                    -np.asarray(mpc["dcline"][:, c["BR_STATUS"]]).reshape(-1),
                    (np.arange(ndc), e2i[np.asarray(mpc["dcline"][:, c["F_BUS"]], dtype=int)] - 1),
                ),
                shape=(ndc, nb),
            ) + sparse.csc_matrix(
                (
                    np.asarray(mpc["dcline"][:, c["BR_STATUS"]]).reshape(-1),
                    (np.arange(ndc), e2i[np.asarray(mpc["dcline"][:, c["T_BUS"]], dtype=int)] - 1),
                ),
                shape=(ndc, nb),
            )
            Cdc = sparse.csc_matrix(
                (-np.ones(ndc), (np.arange(ndc), e2i[np.asarray(mpc["dcline"][:, c["F_BUS"]], dtype=int)] - 1)),
                shape=(ndc, nb),
            ) + sparse.csc_matrix(
                (np.ones(ndc), (np.arange(ndc), e2i[np.asarray(mpc["dcline"][:, c["T_BUS"]], dtype=int)] - 1)),
                shape=(ndc, nb),
            )
        else:
            _Cdc_on = sparse.csc_matrix((0, nb))
            Cdc = sparse.csc_matrix((0, nb))
        _Cg_on = sparse.csc_matrix(
            (
                np.asarray(mpc["gen"][:, GEN_STATUS]).reshape(-1),
                (np.arange(ng), e2i[np.asarray(mpc["gen"][:, GEN_BUS], dtype=int)] - 1),
            ),
            shape=(ng, nb),
        )
        Cg = sparse.csc_matrix(
            (np.ones(ng), (np.arange(ng), e2i[np.asarray(mpc["gen"][:, GEN_BUS], dtype=int)] - 1)),
            shape=(ng, nb),
        )

        print("Checking connectivity ... ", file=fd)
        groups, isolated = connected_components_full(C_on)
        ngr = len(groups)
        nis = len(isolated)
        have_isolated = nis > 0
        if ngr == 1:
            if have_isolated:
                s = "" if nis == 1 else "es"
                print(f"single connected network, plus {nis} isolated bus{s}\n", file=fd)
            else:
                print("single fully connected network\n", file=fd)
        else:
            s = "" if nis == 1 else "es"
            print(f"{ngr} connected groups, {nis} isolated bus{s}\n", file=fd)

        bron = np.asarray(mpc["branch"][:, BR_STATUS]).reshape(-1) > 0
        broff = np.asarray(mpc["branch"][:, BR_STATUS]).reshape(-1) <= 0
        if ndc:
            dcon = np.asarray(mpc["dcline"][:, c["BR_STATUS"]]).reshape(-1) > 0
            dcoff = np.asarray(mpc["dcline"][:, c["BR_STATUS"]]).reshape(-1) <= 0
        gon = np.asarray(mpc["gen"][:, GEN_STATUS]).reshape(-1) > 0
        goff = np.asarray(mpc["gen"][:, GEN_STATUS]).reshape(-1) <= 0

        keys = [
            "nb",
            "nl",
            "nl_on",
            "nl_off",
            "nlt",
            "nlt_off",
            "ndc",
            "ndc_on",
            "ndc_off",
            "ndct",
            "ndct_on",
            "ndct_off",
            "ndc_all",
            "ng",
            "ng_on",
            "ng_off",
            "nsh",
            "nld",
            "nld_on",
            "nld_off",
            "nfld",
            "ndld",
            "ndld_on",
            "ndld_off",
            "Pmax",
            "Qmax",
            "Pmax_on",
            "Qmax_on",
            "Pmax_off",
            "Qmax_off",
            "Pmin",
            "Qmin",
            "Pmin_on",
            "Qmin_on",
            "Pmin_off",
            "Qmin_off",
            "Pg",
            "Qg",
            "Ps",
            "Qs",
            "Ploss",
            "Qloss",
            "Pd",
            "Qd",
            "Pd_fixed",
            "Qd_fixed",
            "Pd_disp_cap",
            "Qd_disp_cap",
            "Pd_disp_cap_on",
            "Qd_disp_cap_on",
            "Pd_disp_cap_off",
            "Qd_disp_cap_off",
            "Pd_disp",
            "Qd_disp",
            "Pd_curtailed",
            "Qd_curtailed",
            "Pd_cap",
            "Qd_cap",
            "Pd_cap_on",
            "Qd_cap_on",
            "Pd_cap_off",
            "Qd_cap_off",
            "Pdc",
            "Pmaxdc",
            "Pmaxdc_on",
            "Pmaxdc_off",
            "Pmindc",
            "Pmindc_on",
            "Pmindc_off",
        ]
        d0: dict[str, Any] = {k: 0 for k in keys}
        d: list[dict[str, Any]] = []
        total: dict[str, Any] = {k: 0 for k in keys}
        allrefs = np.flatnonzero(np.asarray(mpc["bus"][:, BUS_TYPE]).reshape(-1) == REF) + 1
        refs = []
        nrefs = 0
        ibr_tie_all = np.array([], dtype=int)
        idc_tie_all = np.array([], dtype=int)
        idc_tie_all_on = np.array([], dtype=int)
        idc_tie_all_off = np.array([], dtype=int)
        dcon = np.array([], dtype=bool)
        dcoff = np.array([], dtype=bool)
        idc_tie_on = np.array([], dtype=int)
        idc_tie_off = np.array([], dtype=int)

        fields = list(d0.keys())
        for k in range(1, ngr + have_isolated + 1):
            dk: dict[str, Any] = {kk: 0 for kk in fields}
            if k > ngr:
                b = np.asarray(isolated).reshape(-1)
                ibr = np.array([], dtype=int)
                idc = np.array([], dtype=int)
            else:
                b = np.asarray(groups[k - 1]).reshape(-1)
                ibr = (
                    np.flatnonzero(
                        (np.asarray(np.abs(C[:, b - 1]).sum(axis=1)).reshape(-1) != 0)
                        & (np.asarray(C[:, b - 1].sum(axis=1)).reshape(-1) == 0)
                    )
                    + 1
                )
                idc = (
                    np.flatnonzero(
                        (np.asarray(np.abs(Cdc[:, b - 1]).sum(axis=1)).reshape(-1) != 0)
                        & (np.asarray(Cdc[:, b - 1].sum(axis=1)).reshape(-1) == 0)
                    )
                    + 1
                    if ndc
                    else np.array([], dtype=int)
                )
            ibr_tie = np.flatnonzero(np.asarray(C[:, b - 1].sum(axis=1)).reshape(-1) != 0) + 1
            idc_tie = (
                np.flatnonzero(np.asarray(Cdc[:, b - 1].sum(axis=1)).reshape(-1) != 0) + 1
                if ndc
                else np.array([], dtype=int)
            )
            refk = b[np.flatnonzero(np.asarray(mpc["bus"][b - 1, BUS_TYPE]).reshape(-1) == REF)]
            refs.append(refk)
            nrefs += len(refk)

            ibr_on = ibr[bron[ibr - 1]]
            ibr_off = ibr[broff[ibr - 1]]
            ibr_tie_off = ibr_tie[broff[ibr_tie - 1]]
            ibr_tie_all = np.unique(np.r_[ibr_tie_all, ibr_tie_off])
            if ndc:
                idc_on = idc[dcon[idc - 1]]
                idc_off = idc[dcoff[idc - 1]]
                idc_tie_on = idc_tie[dcon[idc_tie - 1]]
                idc_tie_off = idc_tie[dcoff[idc_tie - 1]]
                idc_tie_all = np.unique(np.r_[idc_tie_all, idc_tie])
                idc_tie_all_on = np.unique(np.r_[idc_tie_all_on, idc_tie_on])
                idc_tie_all_off = np.unique(np.r_[idc_tie_all_off, idc_tie_off])
            else:
                idc_on = np.array([], dtype=int)
                idc_off = np.array([], dtype=int)
            ig = np.flatnonzero(np.asarray(np.abs(Cg[:, b - 1]).sum(axis=1)).reshape(-1) != 0) + 1
            if len(ig) == 0:
                ig_on = np.array([], dtype=int)
                ig_off = np.array([], dtype=int)
                idld_on = np.array([], dtype=int)
                idld_off = np.array([], dtype=int)
            else:
                ild = np.asarray(isload(mpc["gen"][ig - 1, :])).reshape(-1) != 0
                ig_on = ig[gon[ig - 1] & ~ild]
                ig_off = ig[goff[ig - 1] & ~ild]
                idld_on = ig[gon[ig - 1] & ild]
                idld_off = ig[goff[ig - 1] & ild]

            dk["nb"] = len(b)
            dk["nl"] = len(ibr)
            dk["nl_on"] = len(ibr_on)
            dk["nl_off"] = len(ibr_off)
            dk["nlt"] = len(ibr_tie)
            dk["nlt_off"] = len(ibr_tie_off)
            if ndc:
                dk["ndc"] = len(idc)
                dk["ndc_on"] = len(idc_on)
                dk["ndc_off"] = len(idc_off)
                dk["ndct"] = len(idc_tie)
                dk["ndct_on"] = len(idc_tie_on)
                dk["ndct_off"] = len(idc_tie_off)
                dk["ndc_all"] = dk["ndc"] + dk["ndct"]
            dk["ng"] = len(ig_on) + len(ig_off)
            dk["ng_on"] = len(ig_on)
            dk["ng_off"] = len(ig_off)
            dk["nsh"] = int(np.sum((mpc["bus"][b - 1, GS] != 0) | (mpc["bus"][b - 1, BS] != 0)))
            dk["nfld"] = int(np.sum((mpc["bus"][b - 1, PD] != 0) | (mpc["bus"][b - 1, QD] != 0)))
            dk["ndld"] = len(idld_on) + len(idld_off)
            dk["ndld_on"] = len(idld_on)
            dk["ndld_off"] = len(idld_off)
            dk["nld"] = dk["nfld"] + dk["ndld"]
            dk["nld_on"] = dk["nfld"] + dk["ndld_on"]
            dk["nld_off"] = dk["ndld_off"]

            dk["Pmax_on"] = float(np.sum(mpc["gen"][ig_on - 1, PMAX]))
            dk["Pmax_off"] = float(np.sum(mpc["gen"][ig_off - 1, PMAX]))
            dk["Pmax"] = dk["Pmax_on"] + dk["Pmax_off"]
            dk["Pmin_on"] = float(np.sum(mpc["gen"][ig_on - 1, PMIN]))
            dk["Pmin_off"] = float(np.sum(mpc["gen"][ig_off - 1, PMIN]))
            dk["Pmin"] = dk["Pmin_on"] + dk["Pmin_off"]
            dk["Pg"] = float(np.sum(mpc["gen"][ig_on - 1, PG]))
            dk["Qmax_on"] = float(np.sum(mpc["gen"][ig_on - 1, QMAX]))
            dk["Qmax_off"] = float(np.sum(mpc["gen"][ig_off - 1, QMAX]))
            dk["Qmax"] = dk["Qmax_on"] + dk["Qmax_off"]
            dk["Qmin_on"] = float(np.sum(mpc["gen"][ig_on - 1, QMIN]))
            dk["Qmin_off"] = float(np.sum(mpc["gen"][ig_off - 1, QMIN]))
            dk["Qmin"] = dk["Qmin_on"] + dk["Qmin_off"]
            dk["Qg"] = float(np.sum(mpc["gen"][ig_on - 1, QG]))
            dk["Ps"] = float(np.sum(-(mpc["bus"][b - 1, VM] ** 2) * mpc["bus"][b - 1, GS]))
            dk["Qs"] = float(np.sum(mpc["bus"][b - 1, VM] ** 2 * mpc["bus"][b - 1, BS]))
            if mpc["branch"].shape[1] > PF:
                dk["Ploss"] = float(np.sum(mpc["branch"][ibr_on - 1, PF] + mpc["branch"][ibr_on - 1, PT]))
                dk["Qloss"] = float(np.sum(mpc["branch"][ibr_on - 1, QF] + mpc["branch"][ibr_on - 1, QT]))
            dk["Pd_fixed"] = float(np.sum(mpc["bus"][b - 1, PD]))
            dk["Qd_fixed"] = float(np.sum(mpc["bus"][b - 1, QD]))
            dk["Pd_disp_cap_on"] = float(np.sum(-mpc["gen"][idld_on - 1, PMIN]))
            dk["Qd_disp_cap_on"] = float(np.sum(-mpc["gen"][idld_on - 1, QMIN]))
            dk["Pd_disp_cap_off"] = float(np.sum(-mpc["gen"][idld_off - 1, PMIN]))
            dk["Qd_disp_cap_off"] = float(np.sum(-mpc["gen"][idld_off - 1, QMIN]))
            dk["Pd_disp_cap"] = dk["Pd_disp_cap_on"] + dk["Pd_disp_cap_off"]
            dk["Qd_disp_cap"] = dk["Qd_disp_cap_on"] + dk["Qd_disp_cap_off"]
            dk["Pd_disp"] = float(np.sum(-mpc["gen"][idld_on - 1, PG]))
            dk["Qd_disp"] = float(np.sum(-mpc["gen"][idld_on - 1, QG]))
            dk["Pd_curtailed"] = dk["Pd_disp_cap_on"] - dk["Pd_disp"]
            dk["Qd_curtailed"] = dk["Qd_disp_cap_on"] - dk["Qd_disp"]
            dk["Pd"] = dk["Pd_fixed"] + dk["Pd_disp"]
            dk["Qd"] = dk["Qd_fixed"] + dk["Qd_disp"]
            dk["Pd_cap"] = dk["Pd_fixed"] + dk["Pd_disp_cap"]
            dk["Qd_cap"] = dk["Qd_fixed"] + dk["Qd_disp_cap"]
            dk["Pd_cap_on"] = dk["Pd_fixed"] + dk["Pd_disp_cap_on"]
            dk["Qd_cap_on"] = dk["Qd_fixed"] + dk["Qd_disp_cap_on"]
            dk["Pd_cap_off"] = dk["Pd_disp_cap_off"]
            dk["Qd_cap_off"] = dk["Qd_disp_cap_off"]
            if ndc:
                fs = np.flatnonzero(np.asarray(Cdc[:, b - 1].sum(axis=1)).reshape(-1) < 0) + 1
                ts = np.flatnonzero(np.asarray(Cdc[:, b - 1].sum(axis=1)).reshape(-1) > 0) + 1
                dk["Pdc"] = float(
                    np.sum(mpc["dcline"][fs - 1, c["PF"]]) - np.sum(mpc["dcline"][ts - 1, c["PT"]])
                )
                dk["Pmaxdc"] = float(
                    np.sum(mpc["dcline"][fs - 1, c["PMAX"]]) - np.sum(mpc["dcline"][ts - 1, c["PMAX"]])
                )
                dk["Pmindc"] = float(
                    np.sum(mpc["dcline"][fs - 1, c["PMIN"]]) - np.sum(mpc["dcline"][ts - 1, c["PMIN"]])
                )
                fs = np.flatnonzero((np.asarray(Cdc[:, b - 1].sum(axis=1)).reshape(-1) < 0) & dcon) + 1
                ts = np.flatnonzero((np.asarray(Cdc[:, b - 1].sum(axis=1)).reshape(-1) > 0) & dcon) + 1
                dk["Pmaxdc_on"] = float(
                    np.sum(mpc["dcline"][fs - 1, c["PMAX"]]) - np.sum(mpc["dcline"][ts - 1, c["PMAX"]])
                )
                dk["Pmindc_on"] = float(
                    np.sum(mpc["dcline"][fs - 1, c["PMIN"]]) - np.sum(mpc["dcline"][ts - 1, c["PMIN"]])
                )
                fs = np.flatnonzero((np.asarray(Cdc[:, b - 1].sum(axis=1)).reshape(-1) < 0) & dcoff) + 1
                ts = np.flatnonzero((np.asarray(Cdc[:, b - 1].sum(axis=1)).reshape(-1) > 0) & dcoff) + 1
                dk["Pmaxdc_off"] = float(
                    np.sum(mpc["dcline"][fs - 1, c["PMAX"]]) - np.sum(mpc["dcline"][ts - 1, c["PMAX"]])
                )
                dk["Pmindc_off"] = float(
                    np.sum(mpc["dcline"][fs - 1, c["PMIN"]]) - np.sum(mpc["dcline"][ts - 1, c["PMIN"]])
                )

            for ff in fields:
                total[ff] += dk[ff]
            total["nl"] = nl
            total["nl_on"] = int(np.sum(bron))
            total["nl_off"] = int(np.sum(broff))
            total["ndc"] = ndc
            total["ndc_on"] = int(np.sum(dcon)) if ndc else 0
            total["ndc_off"] = int(np.sum(dcoff)) if ndc else 0
            d.append(dk)

        total["nlt"] = len(ibr_tie_all)
        total["nlt_off"] = len(ibr_tie_all)
        if ndc:
            total["ndct"] = len(idc_tie_all)
            total["ndct_on"] = len(idc_tie_all_on)
            total["ndct_off"] = len(idc_tie_all_off)

        print(f"Elapsed time is {time.perf_counter() - t0:f} seconds.\n", file=fd)
        print("================================================================================\n", file=fd)
        pages = int(math.ceil((ngr + have_isolated + 1) / 5.0))
        for page in range(1, pages + 1):
            if page == 1:
                islands = [] if ngr == 1 and not have_isolated else list(range(1, min(4, ngr + have_isolated) + 1))
            else:
                print("--------------------------------------------------------------------------------\n", file=fd)
                islands = list(range(5 * (page - 1), min(5 * page - 1, ngr + have_isolated) + 1))

            print(f"{'':<20s}", file=fd)
            if page == 1:
                print("    Full    ", file=fd)
            for k in islands:
                print("  Isolated  " if k > ngr else "   Island   ", file=fd)
            print("\n", file=fd)

            print(f"{'':<20s}", file=fd)
            if page == 1:
                print("   System   ", file=fd)
            for k in islands:
                print("    Buses   " if k > ngr else f"  {k:5d}     ", file=fd)
            print("\n", file=fd)

            print(f"{'Number of:':<20s}", file=fd)
            if page == 1:
                print(" ---------- ", file=fd)
            for _ in islands:
                print(" ---------- ", file=fd)
            print("\n", file=fd)

            p = {"page": page, "islands": islands, "total": total, "d": d}
            _print_row(fd, p, " %8d   ", "  buses", "nb")
            _print_row(fd, p, " %8d   ", "  loads", "nld")
            _print_row(fd, p, " %8d   ", "    on", "nld_on")
            _print_row(fd, p, " %8d   ", "    off", "nld_off")
            _print_row(fd, p, " %8d   ", "    fixed", "nfld")
            _print_row(fd, p, " %8d   ", "    dispatchable", "ndld")
            _print_row(fd, p, " %8d   ", "      on", "ndld_on")
            _print_row(fd, p, " %8d   ", "      off", "ndld_off")
            _print_row(fd, p, " %8d   ", "  generators", "ng")
            _print_row(fd, p, " %8d   ", "    on", "ng_on")
            _print_row(fd, p, " %8d   ", "    off", "ng_off")
            _print_row(fd, p, " %8d   ", "  shunt elements", "nsh")
            _print_row(fd, p, " %8d   ", "  branches", "nl")
            _print_row(fd, p, " %8d   ", "    on", "nl_on")
            _print_row(fd, p, " %8d   ", "    off", "nl_off")
            _print_row(fd, p, " %8d   ", "    ties (off)", "nlt_off")
            if ndc:
                _print_row(fd, p, " %8d   ", "  DC lines", "ndc_all")
                _print_row(fd, p, " %8d   ", "    within", "ndc")
                _print_row(fd, p, " %8d   ", "      on", "ndc_on")
                _print_row(fd, p, " %8d   ", "      off", "ndc_off")
                _print_row(fd, p, " %8d   ", "    ties", "ndct")
                _print_row(fd, p, " %8d   ", "      on", "ndct_on")
                _print_row(fd, p, " %8d   ", "      off", "ndct_off")

            print(f"\n{'Load':<20s}\n", file=fd)
            print(f"{'  active (MW)':<20s}\n", file=fd)
            for name, field in [
                ("    dispatched", "Pd"),
                ("      fixed", "Pd_fixed"),
                ("      dispatchable", "Pd_disp"),
                ("    curtailed", "Pd_curtailed"),
                ("    nominal", "Pd_cap"),
                ("      on", "Pd_cap_on"),
                ("      off", "Pd_cap_off"),
                ("      fixed", "Pd_fixed"),
                ("      dispatchable", "Pd_disp_cap"),
                ("        on", "Pd_disp_cap_on"),
                ("        off", "Pd_disp_cap_off"),
            ]:
                _print_row(fd, p, " %11.1f ", name, field)
            print(f"{'  reactive (MVAr)':<20s}\n", file=fd)
            for name, field in [
                ("    dispatched", "Qd"),
                ("      fixed", "Qd_fixed"),
                ("      dispatchable", "Qd_disp"),
                ("    curtailed", "Qd_curtailed"),
                ("    nominal", "Qd_cap"),
                ("      on", "Qd_cap_on"),
                ("      off", "Qd_cap_off"),
                ("      fixed", "Qd_fixed"),
                ("      dispatchable", "Qd_disp_cap"),
                ("        on", "Qd_disp_cap_on"),
                ("        off", "Qd_disp_cap_off"),
            ]:
                _print_row(fd, p, " %11.1f ", name, field)

            print(f"\n{'Generation':<20s}\n", file=fd)
            print(f"{'  active (MW)':<20s}\n", file=fd)
            for name, field in [
                ("    dispatched", "Pg"),
                ("    max capacity", "Pmax"),
                ("      on", "Pmax_on"),
                ("      off", "Pmax_off"),
                ("    min capacity", "Pmin"),
                ("      on", "Pmin_on"),
                ("      off", "Pmin_off"),
            ]:
                _print_row(fd, p, " %11.1f ", name, field)
            print(f"{'  reactive (MVAr)':<20s}\n", file=fd)
            for name, field in [
                ("    dispatched", "Qg"),
                ("    max capacity", "Qmax"),
                ("      on", "Qmax_on"),
                ("      off", "Qmax_off"),
                ("    min capacity", "Qmin"),
                ("      on", "Qmin_on"),
                ("      off", "Qmin_off"),
            ]:
                _print_row(fd, p, " %11.1f ", name, field)

            print(f"\n{'Shunt Injections':<20s}\n", file=fd)
            _print_row(fd, p, " %11.1f ", "    active (MW)", "Ps")
            _print_row(fd, p, " %11.1f ", "    reactive (MVAr)", "Qs")

            print(f"\n{'Branch Losses':<20s}\n", file=fd)
            _print_row(fd, p, " %11.1f ", "    active (MW)", "Ploss")
            _print_row(fd, p, " %11.1f ", "    reactive (MVAr)", "Qloss")

            print(f"\n{'DC line':<20s}\n", file=fd)
            print(f"{'  export (MW)':<20s}\n", file=fd)
            for name, field in [
                ("    dispatch", "Pdc"),
                ("    max capacity", "Pmaxdc"),
                ("      on", "Pmaxdc_on"),
                ("      off", "Pmaxdc_off"),
                ("    min capacity", "Pmindc"),
                ("      on", "Pmindc_on"),
                ("      off", "Pmindc_off"),
            ]:
                _print_row(fd, p, " %11.1f ", name, field)

            print(f"\n{'Reference Buses':<20s}\n", file=fd)
            print(f"{'  num of ref buses':<20s}", file=fd)
            if page == 1:
                print(f" {nrefs:8d}   ", file=fd)
            for k in islands:
                print(f" {len(refs[k - 1]):8d}   ", file=fd)
            print("\n", file=fd)

            for j in range(1, nrefs + 1):
                print(f"{'  ref bus numbers' if j == 1 else '':<20s}", file=fd)
                if page == 1:
                    print(f" {int(mpc['bus'][allrefs[j - 1] - 1, BUS_I]):8d}   ", file=fd)
                for k in islands:
                    if j <= len(refs[k - 1]):
                        print(f" {int(mpc['bus'][refs[k - 1][j - 1] - 1, BUS_I]):8d}   ", file=fd)
                    else:
                        print(f" {'':8s}   ", file=fd)
                print("\n", file=fd)

            if page != pages:
                print("\n\n", file=fd)

    if nargout == 0:
        return None
    if nargout == 1:
        return groups
    return groups, isolated
