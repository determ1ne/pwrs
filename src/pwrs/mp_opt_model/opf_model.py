# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, overload

import numpy as np
from scipy import sparse

from .mp_idx_manager import _NamedSet
from .opt_model import OptModel


class OPFModel(OptModel):
    """OPF-specific optimization model object.

    Extends :class:`OptModel` with MATPOWER case storage and legacy cost
    support used by the OPF formulation.

    Parameters
    ----------
    mpc : dict or OPFModel, optional
        MATPOWER case dict or existing OPF model used to initialize the
        object.
    """

    def __init__(self, mpc: Any | None = None):
        self.cost = _NamedSet()
        self.mpc: dict[str, Any] = {}
        super().__init__(mpc if isinstance(mpc, (dict, OPFModel)) else None)
        if not self.cost.data and self.__class__ is OPFModel:
            self.init_set_types()
        if isinstance(mpc, dict):
            self.mpc = mpc

    def def_set_types(self) -> None:
        self.set_types = {
            "var": "variable",
            "lin": "linear constraint",
            "nle": "nonlinear equality constraint",
            "nli": "nonlinear inequality constraint",
            "qdc": "quadratic cost",
            "nlc": "general nonlinear cost",
            "cost": "legacy cost",
        }

    def init_set_types(self) -> None:
        super().init_set_types()
        self.cost.data = {"N": {}, "H": {}, "Cw": {}, "dd": {}, "rh": {}, "kk": {}, "mm": {}, "vs": {}}
        self.cost.params = None

    def get_mpc(self):
        return self.mpc

    def add_vars(self, *args, **kwargs):
        return self.add_var(*args, **kwargs)

    def add_constraints(self, name: str, *args: Any):
        items = list(args)
        if items and isinstance(items[0], list) and len(items) < 3:
            ff = "lin" if len(items) == 1 else str(items[1]).lower()
            self.init_indexed_name(ff, name, items[0])
            return self
        idx: list[int] = []
        if items and isinstance(items[0], list):
            idx = items.pop(0)
        if len(items) < 3 or callable(items[2]):
            if len(items) < 4:
                raise ValueError("@opf_model/add_constraints: insufficient arguments for nonlinear constraints")
            N, iseq, fcn, hess = items[:4]
            varsets = items[4] if len(items) > 4 else []
            return self.add_nln_constraint(name, idx, N, iseq, fcn, hess, varsets)
        A, l, u = items[:3]
        varsets = items[3] if len(items) > 3 else []
        return self.add_lin_constraint(name, idx, A, l, u, varsets)

    def add_legacy_cost(self, name: str, idx: Any, cp: dict[str, Any] | None = None, varsets: Any = None):
        if not isinstance(idx, list):
            varsets = cp if varsets is None else varsets
            cp = idx
            idx = []
        assert cp is not None
        vs = self.varsets_cell2struct(varsets if varsets is not None else [])
        nv = self.varsets_len(vs)
        if "N" in cp:
            N = sparse.csc_matrix(cp["N"])
            matrix_shape = np.asarray(N.shape, dtype=int)
            nw, nx = int(matrix_shape[0]), int(matrix_shape[1])
        else:
            nw = len(np.asarray(cp["Cw"]).reshape(-1))
            nx = nw
            N = sparse.eye(nw, nx, format="csc")
            cp = dict(cp)
            cp["N"] = N
        if nx != nv and nw != 0:
            raise ValueError(
                f"@opt_model/add_legacy_cost: number of columns in N ({nw} x {nx}) does not match\nnumber of variables ({nv})\n"
            )
        Cw = np.asarray(cp["Cw"], dtype=float).reshape(-1)
        if Cw.shape[0] != nw:
            raise ValueError(
                f"@opt_model/add_legacy_cost: number of rows of Cw ({Cw.shape[0]} x 1) and N ({nw} x {nx}) must match\n"
            )
        self.add_named_set("cost", name, idx, nw, cp, vs)
        store = self.cost.data
        normalized = {
            "N": N,
            "Cw": Cw,
            "H": sparse.csc_matrix(cp["H"]) if "H" in cp and np.size(cp["H"]) else None,
            "dd": np.asarray(cp["dd"], dtype=float).reshape(-1) if "dd" in cp and np.size(cp["dd"]) else None,
            "rh": np.asarray(cp["rh"], dtype=float).reshape(-1) if "rh" in cp and np.size(cp["rh"]) else None,
            "kk": np.asarray(cp["kk"], dtype=float).reshape(-1) if "kk" in cp and np.size(cp["kk"]) else None,
            "mm": np.asarray(cp["mm"], dtype=float).reshape(-1) if "mm" in cp and np.size(cp["mm"]) else None,
            "vs": vs,
        }
        if not idx:
            for field, value in normalized.items():
                store[field][name] = value
        else:
            key = tuple(idx)
            for field, value in normalized.items():
                store[field].setdefault(name, {})[key] = value
        self.cost.params = None
        return self

    @overload
    def params_legacy_cost(
        self, name: str, idx: list[int] | None = None
    ) -> tuple[dict[str, Any], list[Any], int, int]: ...

    @overload
    def params_legacy_cost(self, name: None = None, idx: list[int] | None = None) -> tuple[dict[str, Any], list[Any]]: ...

    def params_legacy_cost(self, name: str | None = None, idx: list[int] | None = None):
        if name is not None:
            if not idx:
                if np.size(self.cost.idx.i1[name]) != 1:
                    raise ValueError(
                        f"@opt_model/params_legacy_cost: legacy cost set '{name}' requires an IDX_LIST arg"
                    )
                N = self.cost.data["N"][name]
                Cw = self.cost.data["Cw"][name]
                vs = self.cost.data["vs"][name]
                i1 = int(self.cost.idx.i1[name])
                iN = int(self.cost.idx.iN[name])
                raw = {field: self.cost.data[field].get(name) for field in ("H", "dd", "rh", "kk", "mm")}
            else:
                key = tuple(idx)
                ref = tuple(i - 1 for i in idx)
                N = self.cost.data["N"][name][key]
                Cw = self.cost.data["Cw"][name][key]
                vs = self.cost.data["vs"][name][key]
                i1 = int(self.cost.idx.i1[name][ref])
                iN = int(self.cost.idx.iN[name][ref])
                raw = {field: self.cost.data[field].get(name, {}).get(key) for field in ("H", "dd", "rh", "kk", "mm")}
            nw, _ = N.shape
            ones = np.ones(nw)
            zeros = np.zeros(nw)
            cp = {"N": N, "Cw": Cw, "H": sparse.csc_matrix((nw, nw)), "dd": ones, "rh": zeros, "kk": zeros, "mm": ones}
            for field, value in raw.items():
                if value is not None:
                    cp[field] = value
            return cp, vs, i1, iN
        cache = self.cost.params
        if cache is None:
            nx = self.var.N
            nw = self.cost.N
            N = sparse.lil_matrix((nw, nx))
            Cw = np.zeros(nw)
            H = sparse.lil_matrix((nw, nw))
            dd = np.ones(nw)
            rh = np.zeros(nw)
            kk = np.zeros(nw)
            mm = np.ones(nw)
            for entry in self.cost.order:
                cp, vs, i1, iN = self.params_legacy_cost(entry["name"], entry["idx"])
                rows = slice(i1 - 1, iN)
                if vs:
                    jj = self.varsets_idx(vs) - 1
                    N[np.arange(i1 - 1, iN)[:, None], jj] = cp["N"].toarray()
                else:
                    N[rows, : cp["N"].shape[1]] = cp["N"]
                Cw[rows] = cp["Cw"]
                H[rows, rows] = cp["H"]
                dd[rows] = cp["dd"]
                rh[rows] = cp["rh"]
                kk[rows] = cp["kk"]
                mm[rows] = cp["mm"]
            self.cost.params = {"N": N.tocsr(), "Cw": Cw, "H": H.tocsr(), "dd": dd, "rh": rh, "kk": kk, "mm": mm}
            cache = self.cost.params
        return cache, []

    def get_cost_params(self, name: str | None = None, idx: list[int] | None = None):
        cp, *_ = self.params_legacy_cost()
        if name is None:
            return cp
        if self.getN("cost", name, idx):
            _, _, i1, iN = self.params_legacy_cost(name, idx)
            rows = slice(i1 - 1, iN)
            return {
                "N": cp["N"][rows, :],
                "Cw": cp["Cw"][rows],
                "H": cp["H"][rows, rows],
                "dd": cp["dd"][rows],
                "rh": cp["rh"][rows],
                "kk": cp["kk"][rows],
                "mm": cp["mm"][rows],
            }
        return cp

    def eval_legacy_cost(self, x: np.ndarray, name: str | None = None, idx: list[int] | None = None):
        from ..core.opf_legacy_user_cost_fcn import opf_legacy_user_cost_fcn

        if self.cost.N == 0:
            return 0.0
        if name is None:
            cp, vs = self.params_legacy_cost()[:2]
        else:
            cp, vs = self.params_legacy_cost(name, idx)[:2]
        xx = self.varsets_x(x, vs, "vector")
        return opf_legacy_user_cost_fcn(xx, cp, nargout=1)
