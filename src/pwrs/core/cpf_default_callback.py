# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy

import numpy as np

from ..corex import MatpowerConfig
from ..utils import map_e2i
from .mpoption import mpoption


def _get_cb_mpopt(cb_data):
    mpopt = cb_data["mpopt"]
    if isinstance(mpopt, MatpowerConfig):
        return mpopt
    return mpoption(mpopt)


def cpf_default_callback(k, nx, cx, px, done, rollback, evnts, cb_data, cb_args, results=None):
    """Default CPF callback for result accumulation and plotting.

    Mirrors MATPOWER's default CPF callback behavior by storing predictor and
    corrector trajectories across continuation steps, finalizing the CPF
    result arrays, and optionally updating the default CPF plot.

    Parameters
    ----------
    k : int
        Current continuation step index. Negative values indicate the
        callback finalization pass.
    nx : dict
        Next-step CPF state struct.
    cx : dict
        Current-step CPF state struct.
    px : dict
        Previous-step CPF state struct.
    done : dict
        CPF completion flag/message struct.
    rollback : bool
        Whether the current step is being rolled back.
    evnts : dict or list of dict
        Detected CPF events for the current step.
    cb_data : dict
        CPF callback data struct.
    cb_args : dict
        Callback arguments struct.
    results : dict, optional
        CPF results accumulator.

    Returns
    -------
    tuple
        Updated ``(nx, cx, done, rollback, evnts, cb_data, results)``.
    """
    if rollback and int(np.asarray(k).reshape(-1)[0]) > 0:
        return nx, cx, done, rollback, evnts, cb_data, results

    k = int(np.asarray(k).reshape(-1)[0])
    step = nx["step"]
    V = np.asarray(nx["V"]).reshape(-1, 1)
    lam = float(np.asarray(nx["lam"]).reshape(-1)[0])
    V_hat = np.asarray(nx["V_hat"]).reshape(-1, 1)
    lam_hat = float(np.asarray(nx["lam_hat"]).reshape(-1)[0])

    if k == 0:
        cxx = {
            "V_hat": V_hat,
            "lam_hat": np.array([[lam_hat]], dtype=float),
            "V": V,
            "lam": np.array([[lam]], dtype=float),
            "steps": np.array([[float(np.asarray(step).reshape(-1)[0])]], dtype=float),
            "iterations": 0,
        }
        nxx = copy.deepcopy(cxx)
        cx.setdefault("cb", {})["default"] = cxx
        nx.setdefault("cb", {})["default"] = nxx
        plot_data = cxx
    else:
        nxx = copy.deepcopy(nx["cb"]["default"])
        plot_data = nxx
        if k > 0:
            nxx["V_hat"] = np.concatenate([np.asarray(nxx["V_hat"]), V_hat], axis=1)
            nxx["lam_hat"] = np.concatenate([np.asarray(nxx["lam_hat"]).reshape(1, -1), [[lam_hat]]], axis=1)
            nxx["V"] = np.concatenate([np.asarray(nxx["V"]), V], axis=1)
            nxx["lam"] = np.concatenate([np.asarray(nxx["lam"]).reshape(1, -1), [[lam]]], axis=1)
            nxx["steps"] = np.concatenate(
                [np.asarray(nxx["steps"]).reshape(1, -1), [[float(np.asarray(step).reshape(-1)[0])]]], axis=1
            )
            nxx["iterations"] = k
            nx.setdefault("cb", {})["default"] = nxx
        else:
            if results is None:
                results = {}
            results["V_hat"] = np.asarray(nxx["V_hat"])
            results["lam_hat"] = np.asarray(nxx["lam_hat"]).reshape(1, -1)
            results["V"] = np.asarray(nxx["V"])
            results["lam"] = np.asarray(nxx["lam"]).reshape(1, -1)
            results["steps"] = np.asarray(nxx["steps"]).reshape(1, -1)
            results["iterations"] = -k
            results["max_lam"] = float(np.max(np.asarray(nxx["lam"]).reshape(-1)))

    plot_level = float(_get_cb_mpopt(cb_data).cpf.plot.level)
    if plot_level:
        _plot_default_callback(k, plot_data, cx, cb_data)

    return nx, cx, done, rollback, evnts, cb_data, results


def _plot_default_callback(k, nxx, cx, cb_data):
    import matplotlib.pyplot as plt

    mpopt = _get_cb_mpopt(cb_data)
    plot_bus = np.asarray(mpopt.cpf.plot.bus).reshape(-1)
    plot_bus_default = 0
    if _isempty(plot_bus) and "plot_bus_default" not in nxx:
        sxfr = cb_data["Sbust"](np.abs(cx["V"]))[0] - cb_data["Sbusb"](np.abs(cx["V"]))[0]
        pq = np.asarray(cb_data["pq"], dtype=int).reshape(-1) - 1
        if pq.size == 0:
            idx = 0
        else:
            idx = int(pq[np.argmax(np.asarray(sxfr).reshape(-1)[pq])])
        idx_e = int(np.asarray(cb_data["mpc_target"]["order"]["bus"]["i2e"]).reshape(-1)[idx])
        plot_bus_default = idx_e
    else:
        if _isempty(plot_bus):
            idx_e = nxx["plot_bus_default"]
        else:
            idx_e = plot_bus
        idx_e = np.asarray(idx_e, dtype=int).reshape(-1)
        idx = map_e2i(cb_data["mpc_target"]["order"]["bus"]["e2i"], idx_e) - 1

    if plot_bus_default:
        idx_e = np.asarray([plot_bus_default], dtype=int)
        idx = map_e2i(cb_data["mpc_target"]["order"]["bus"]["e2i"], idx_e) - 1
        cx["cb"]["default"]["plot_bus_default"] = plot_bus_default
    else:
        idx = np.asarray(idx, dtype=int).reshape(-1)
        idx_e = np.asarray(idx_e, dtype=int).reshape(-1)

    lam_hat = np.asarray(nxx["lam_hat"]).reshape(-1)
    lam = np.asarray(nxx["lam"]).reshape(-1)
    V_hat = np.asarray(nxx["V_hat"])
    V = np.asarray(nxx["V"])

    xmin = 0
    xmax = max(float(np.max(lam_hat)), float(np.max(lam)))
    ymin = min(float(np.min(np.abs(V_hat[idx, :]))), float(np.min(np.abs(V[idx, :]))))
    ymax = max(float(np.max(np.abs(V_hat[idx, :]))), float(np.max(np.abs(V[idx, :]))))
    step0 = float(mpopt.cpf.step)
    if xmax < xmin + step0 / 100:
        xmax = xmin + step0 / 100
    if ymax - ymin < 2e-5:
        ymax = ymax + 1e-5
        ymin = ymin - 1e-5
    xmax = xmax * 1.05
    ymax = ymax + 0.05 * (ymax - ymin)
    ymin = ymin - 0.05 * (ymax - ymin)

    ax = plt.gca()
    ax.axis((xmin, xmax, ymin, ymax))
    if k == 0:
        ax.plot([lam_hat[0]], np.abs(V_hat[idx, 0]), "-", color=[0.25, 0.25, 1])
        ax.set_title("Voltage at Multiple Buses" if len(idx_e) > 1 else f"Voltage at Bus {int(idx_e[0])}")
        ax.set_xlabel("\\lambda")
        ax.set_ylabel("Voltage Magnitude")
    elif k < 0:
        ax.plot(lam, np.abs(V[idx, :]).T, "-")


def _isempty(value):
    if value is None:
        return True
    arr = np.asarray(value)
    return arr.size == 0
