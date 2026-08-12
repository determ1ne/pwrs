# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from .mp_idx_manager import MPIdxManager, _NamedSet
from .nlps_master import nlps_master_full
from .qps_master import qps_master_full


@dataclass
class _VarSet:
    name: str
    idx: list[int]


class OptModel(MPIdxManager):
    """Optimization model object for named variables, constraints, and costs.

    This class encapsulates an optimization problem formulation, providing
    named block access to variables, linear constraints, nonlinear
    constraints, quadratic costs, and general nonlinear costs while
    maintaining ordering and indexing information as blocks are added.

    Parameters
    ----------
    s : OptModel or dict, optional
        Existing optimization model or compatible struct-like dict used to
        initialize the object.
    """

    def __init__(self, s: Any | None = None):
        self.var = _NamedSet()
        self.lin = _NamedSet()
        self.nle = _NamedSet()
        self.nli = _NamedSet()
        self.qdc = _NamedSet()
        self.nlc = _NamedSet()
        self.prob_type = ""
        self.soln: dict[str, Any] = {}
        self._varsets_idx_cache: dict[tuple[tuple[str, tuple[int, ...]], ...], np.ndarray] = {}
        super().__init__(s)
        if not self.var.data and self.__class__ is OptModel:
            self.init_set_types()

    def def_set_types(self) -> None:
        self.set_types = {
            "var": "variable",
            "lin": "linear constraint",
            "nle": "nonlinear equality constraint",
            "nli": "nonlinear inequality constraint",
            "qdc": "quadratic cost",
            "nlc": "general nonlinear cost",
        }

    def init_set_types(self) -> None:
        super().init_set_types()
        self.var.data = {"v0": {}, "vl": {}, "vu": {}, "vt": {}}
        self.lin.data = {"A": {}, "l": {}, "u": {}, "vs": {}}
        self.qdc.data = {"Q": {}, "c": {}, "k": {}, "vs": {}}
        self.nle.data = {"fcn": {}, "hess": {}, "computed_by": {}, "vs": {}}
        self.nli.data = {"fcn": {}, "hess": {}, "computed_by": {}, "vs": {}}
        self.nlc.data = {"N": {}, "fcn": {}, "vs": {}}

    def varsets_cell2struct(self, varsets: list[Any] | tuple[Any, ...] | None) -> list[_VarSet]:
        if not varsets:
            return []
        out: list[_VarSet] = []
        for entry in varsets:
            if isinstance(entry, dict):
                out.append(_VarSet(entry["name"], list(entry.get("idx", []))))
            else:
                out.append(_VarSet(str(entry), []))
        return out

    def varsets_len(self, varsets: list[_VarSet]) -> int:
        if not varsets:
            return self.var.N
        total = 0
        for vs in varsets:
            total += self.getN("var", vs.name, vs.idx)
        return total

    def _varsets_key(self, varsets: list[_VarSet]) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple((vs.name, tuple(vs.idx)) for vs in varsets)

    def varsets_idx(self, varsets: list[_VarSet]) -> np.ndarray:
        if not varsets:
            return np.arange(1, self.var.N + 1, dtype=int)
        key = self._varsets_key(varsets)
        cached = self._varsets_idx_cache.get(key)
        if cached is not None:
            return cached
        jj: list[int] = []
        for vs in varsets:
            idx = self.var.idx
            if not vs.idx:
                i1 = int(idx.i1[vs.name])
                iN = int(idx.iN[vs.name])
            else:
                ref = tuple(i - 1 for i in vs.idx)
                i1 = int(idx.i1[vs.name][ref])
                iN = int(idx.iN[vs.name][ref])
            jj.extend(range(i1, iN + 1))
        out = np.asarray(jj, dtype=int)
        self._varsets_idx_cache[key] = out
        return out

    def varsets_x(self, x: np.ndarray, varsets: list[_VarSet], return_type: str = "vector") -> Any:
        if not varsets:
            return x
        x = np.asarray(x)
        chunks = []
        for vs in varsets:
            jj = self.varsets_idx([vs]) - 1
            chunks.append(x[jj])
        if return_type == "cell":
            return chunks
        return np.concatenate(chunks) if chunks else np.array([])

    def add_var(self, name: str, idx: Any, *args: Any):
        if isinstance(idx, list) and not args:
            self.init_indexed_name("var", name, idx)
            return self
        if isinstance(idx, list):
            N = int(args[0])
            extra = list(args[1:])
            idx_list = idx
        else:
            N = int(idx)
            extra = list(args)
            idx_list = []
        v0 = np.zeros(N)
        vl = -np.inf * np.ones(N)
        vu = np.inf * np.ones(N)
        vt: Any = "C" if N > 0 else ""
        if len(extra) >= 1 and extra[0] is not None and len(np.atleast_1d(extra[0])) > 0:
            v0 = np.asarray(extra[0], dtype=float).reshape(-1)
            if N > 1 and v0.size == 1:
                v0 = np.full(N, float(v0[0]))
        if len(extra) >= 2 and extra[1] is not None and len(np.atleast_1d(extra[1])) > 0:
            vl = np.asarray(extra[1], dtype=float).reshape(-1)
            if N > 1 and vl.size == 1:
                vl = np.full(N, float(vl[0]))
        if len(extra) >= 3 and extra[2] is not None and len(np.atleast_1d(extra[2])) > 0:
            vu = np.asarray(extra[2], dtype=float).reshape(-1)
            if N > 1 and vu.size == 1:
                vu = np.full(N, float(vu[0]))
        if len(extra) >= 4 and extra[3] is not None:
            vt = extra[3]
        self.add_named_set("var", name, idx_list, N, v0, vl, vu, vt)
        self._varsets_idx_cache.clear()
        container = self.var.data
        if not idx_list:
            container["v0"][name] = v0
            container["vl"][name] = vl
            container["vu"][name] = vu
            container["vt"][name] = vt
        else:
            key = tuple(idx_list)
            for field, value in {"v0": v0, "vl": vl, "vu": vu, "vt": vt}.items():
                container[field].setdefault(name, {})[key] = value
        self.var.params = None
        self.prob_type = ""
        return self

    def add_lin_constraint(self, name: str, idx: Any, A: Any = None, l: Any = None, u: Any = None, varsets: Any = None):
        if not isinstance(idx, list):
            varsets = u if varsets is None else varsets
            u = l
            l = A
            A = idx
            idx = []
        A = sparse.csc_matrix(A)
        N, M = A.shape
        l = -np.inf * np.ones(N) if l is None or len(np.atleast_1d(l)) == 0 else np.asarray(l, dtype=float).reshape(-1)
        u = np.inf * np.ones(N) if u is None or len(np.atleast_1d(u)) == 0 else np.asarray(u, dtype=float).reshape(-1)
        if N > 1 and l.size == 1:
            l = np.full(N, float(l[0]))
        if N > 1 and u.size == 1:
            u = np.full(N, float(u[0]))
        if l.shape[0] != N or u.shape[0] != N:
            raise ValueError("@opt_model/add_lin_constraint: sizes of A, l and u must match")
        vs = self.varsets_cell2struct(varsets if varsets is not None else [])
        nv = self.varsets_len(vs)
        if M != nv:
            raise ValueError(
                f"@opt_model/add_lin_constraint: number of columns of A does not match\nnumber of variables, A is {N} x {M}, nv = {nv}\n"
            )
        self.add_named_set("lin", name, idx, N, A, l, u, vs)
        store = self.lin.data
        if not idx:
            store["A"][name] = A
            store["l"][name] = l
            store["u"][name] = u
            store["vs"][name] = vs
        else:
            key = tuple(idx)
            for field, value in {"A": A, "l": l, "u": u, "vs": vs}.items():
                store[field].setdefault(name, {})[key] = value
        self.lin.params = None
        self.prob_type = ""
        return self

    def add_quad_cost(self, name: str, idx: Any, Q: Any = None, c: Any = None, k: Any = None, varsets: Any = None):
        if not isinstance(idx, list):
            Q_in = idx
            c_in = Q
            k_in = c
            varsets_in = k if varsets is None else varsets
            Q = Q_in
            c = c_in
            k = 0 if k_in is None else k_in
            varsets = varsets_in
            idx = []
        Q = sparse.csc_matrix(Q) if Q is not None and np.size(Q) else sparse.csc_matrix((0, 0))
        c = np.asarray(c, dtype=float).reshape(-1) if c is not None and np.size(c) else np.array([])
        k = 0.0 if k is None else k
        vs = self.varsets_cell2struct(varsets if varsets is not None else [])
        nv = self.varsets_len(vs)
        MQ, NQ = Q.shape
        Mc = int(c.shape[0])
        if MQ and NQ not in (MQ, 1):
            raise ValueError(f"@opt_model/add_quad_cost: Q ({MQ} x {NQ}) must be square or a column vector (or empty)")
        if MQ:
            if Mc and Mc != MQ:
                raise ValueError(
                    f"@opt_model/add_quad_cost: dimensions of Q ({MQ} x {NQ}) and c ({Mc} x 1) are not compatible"
                )
            nx = MQ
        else:
            if not Mc:
                raise ValueError("@opt_model/add_quad_cost: Q and c cannot both be empty")
            nx = Mc
        if nx != nv:
            raise ValueError(
                f"@opt_model/add_quad_cost: dimensions of Q ({MQ} x {NQ}) and c ({Mc} x 1) do not match\nnumber of variables ({nv})\n"
            )
        N = nx if (NQ == 1 or Q.shape == (0, 0)) else 1
        self.add_named_set("qdc", name, idx, N, Q, c, k, vs)
        store = self.qdc.data
        if not idx:
            store["Q"][name] = Q
            store["c"][name] = c
            store["k"][name] = k
            store["vs"][name] = vs
        else:
            key = tuple(idx)
            for field, value in {"Q": Q, "c": c, "k": k, "vs": vs}.items():
                store[field].setdefault(name, {})[key] = value
        self.qdc.params = None
        self.prob_type = ""
        return self

    def add_nln_constraint(
        self,
        name: Any,
        idx: Any,
        N: Any = None,
        iseq: Any = None,
        fcn: Any = None,
        hess: Any = None,
        varsets: Any = None,
    ):
        if isinstance(idx, list):
            if varsets is None:
                varsets = []
            idx_list = idx
        else:
            if varsets is None:
                varsets = hess if hess is not None else []
            if N is not None:
                hess = fcn
                fcn = iseq
            iseq = N
            N = idx
            idx_list = []
        ff = "nle" if iseq else "nli"
        vs = self.varsets_cell2struct(varsets if varsets is not None else [])
        if isinstance(name, (list, tuple)):
            if len(name) != len(np.asarray(N).reshape(-1)):
                raise ValueError("@opt_model/add_nln_constraint: dimensions of NAME and N must match")
            Nvec = np.asarray(N, dtype=int).reshape(-1)
            self.add_named_set(ff, str(name[0]), idx_list, int(Nvec[0]), fcn, hess, "", vs)
            store = getattr(self, ff).data
            include = {"name": [str(item) for item in name[1:]], "N": Nvec[1:].astype(int)}
            if not idx_list:
                store["fcn"][str(name[0])] = fcn
                store["hess"][str(name[0])] = hess
                store["computed_by"][str(name[0])] = ""
                store["vs"][str(name[0])] = vs
                store.setdefault("include", {})[str(name[0])] = include
            else:
                key = tuple(idx_list)
                store["fcn"].setdefault(str(name[0]), {})[key] = fcn
                store["hess"].setdefault(str(name[0]), {})[key] = hess
                store["computed_by"].setdefault(str(name[0]), {})[key] = ""
                store["vs"].setdefault(str(name[0]), {})[key] = vs
            for k in range(1, len(name)):
                nk = int(Nvec[k])
                self.add_named_set(ff, str(name[k]), idx_list, nk, [], [], str(name[0]), vs)
                if not idx_list:
                    store["computed_by"][str(name[k])] = str(name[0])
                    store["vs"][str(name[k])] = vs
                else:
                    key = tuple(idx_list)
                    store["computed_by"].setdefault(str(name[k]), {})[key] = str(name[0])
                    store["vs"].setdefault(str(name[k]), {})[key] = vs
        else:
            self.add_named_set(ff, str(name), idx_list, int(N), fcn, hess, "", vs)
            store = getattr(self, ff).data
            if not idx_list:
                store["fcn"][str(name)] = fcn
                store["hess"][str(name)] = hess
                store["computed_by"][str(name)] = ""
                store["vs"][str(name)] = vs
            else:
                key = tuple(idx_list)
                store["fcn"].setdefault(str(name), {})[key] = fcn
                store["hess"].setdefault(str(name), {})[key] = hess
                store["computed_by"].setdefault(str(name), {})[key] = ""
                store["vs"].setdefault(str(name), {})[key] = vs
        getattr(self, ff).params = None
        self.prob_type = ""
        return self

    def add_nln_cost(self, name: str, idx: Any, N: Any = None, fcn: Any = None, varsets: Any = None):
        if isinstance(idx, list):
            if varsets is None:
                varsets = []
            idx_list = idx
        else:
            if varsets is None:
                varsets = fcn if fcn is not None else []
            fcn = N
            N = idx
            idx_list = []
        if int(N) != 1:
            raise ValueError(
                "@opt_model/add_nln_cost: not yet implemented for vector valued functions (i.e. N currently must equal 1)"
            )
        vs = self.varsets_cell2struct(varsets if varsets is not None else [])
        self.add_named_set("nlc", name, idx_list, int(N), fcn, vs)
        if not idx_list:
            self.nlc.data["fcn"][name] = fcn
            self.nlc.data["vs"][name] = vs
        else:
            key = tuple(idx_list)
            self.nlc.data["fcn"].setdefault(name, {})[key] = fcn
            self.nlc.data["vs"].setdefault(name, {})[key] = vs
        self.nlc.params = None
        self.prob_type = ""
        return self

    def params_var(self, name: str | None = None, idx: list[int] | None = None):
        if name is None:
            v0 = []
            vl = []
            vu = []
            vt = ""
            for entry in self.var.order:
                chunk_v0, chunk_vl, chunk_vu, chunk_vt = self.params_var(entry["name"], entry["idx"])
                v0.append(chunk_v0)
                vl.append(chunk_vl)
                vu.append(chunk_vu)
                vt += chunk_vt
            return (
                np.concatenate(v0) if v0 else np.array([]),
                np.concatenate(vl) if vl else np.array([]),
                np.concatenate(vu) if vu else np.array([]),
                vt,
            )
        if not idx:
            raw_v0 = self.var.data["v0"][name]
            raw_vl = self.var.data["vl"][name]
            raw_vu = self.var.data["vu"][name]
            raw_vt = self.var.data["vt"][name]
        else:
            key = tuple(idx)
            raw_v0 = self.var.data["v0"][name][key]
            raw_vl = self.var.data["vl"][name][key]
            raw_vu = self.var.data["vu"][name][key]
            raw_vt = self.var.data["vt"][name][key]
        N = self.getN("var", name, idx)
        vt = raw_vt * N if isinstance(raw_vt, str) and len(raw_vt) == 1 and N > 1 else str(raw_vt)
        return np.asarray(raw_v0, dtype=float), np.asarray(raw_vl, dtype=float), np.asarray(raw_vu, dtype=float), vt

    def params_lin_constraint(self, name: str | None = None, idx: list[int] | None = None):
        if name is not None:
            if not idx:
                if np.size(self.lin.idx.i1[name]) != 1:
                    raise ValueError(
                        f"@opt_model/params_lin_constraint: linear constraint set '{name}' requires an IDX_LIST arg"
                    )
                A = self.lin.data["A"][name]
                l = self.lin.data["l"][name]
                u = self.lin.data["u"][name]
                vs = self.lin.data["vs"][name]
                return A, l, u, vs, int(self.lin.idx.i1[name]), int(self.lin.idx.iN[name])
            key = tuple(idx)
            ref = tuple(i - 1 for i in idx)
            return (
                self.lin.data["A"][name][key],
                self.lin.data["l"][name][key],
                self.lin.data["u"][name][key],
                self.lin.data["vs"][name][key],
                int(self.lin.idx.i1[name][ref]),
                int(self.lin.idx.iN[name][ref]),
            )
        cache = self.lin.params
        if cache is None:
            nx = self.var.N
            nlin = self.lin.N
            A = sparse.lil_matrix((nlin, nx))
            l = -np.inf * np.ones(nlin)
            u = np.inf * np.ones(nlin)
            for entry in self.lin.order:
                Ak, lk, uk, vs, i1, iN = self.params_lin_constraint(entry["name"], entry["idx"])
                if Ak.shape[0]:
                    rows = slice(i1 - 1, iN)
                    if not vs:
                        if Ak.shape[1] == nx:
                            A[rows, :] = Ak
                        else:
                            A[rows, : Ak.shape[1]] = Ak
                    else:
                        jj = self.varsets_idx(vs) - 1
                        A[np.arange(i1 - 1, iN)[:, None], jj] = Ak.toarray()
                    l[rows] = lk
                    u[rows] = uk
            self.lin.params = {"A": A.tocsr(), "l": l, "u": u}
            cache = self.lin.params
        return cache["A"], cache["l"], cache["u"], [], 1, self.lin.N

    def params_quad_cost(self, name: str | None = None, idx: list[int] | None = None):
        if name is not None:
            if not idx:
                if np.size(self.qdc.idx.i1[name]) != 1:
                    raise ValueError(
                        f"@opt_model/params_quad_cost: quadratic cost set '{name}' requires an IDX_LIST arg"
                    )
                return (
                    self.qdc.data["Q"][name],
                    self.qdc.data["c"][name],
                    self.qdc.data["k"][name],
                    self.qdc.data["vs"][name],
                )
            key = tuple(idx)
            return (
                self.qdc.data["Q"][name][key],
                self.qdc.data["c"][name][key],
                self.qdc.data["k"][name][key],
                self.qdc.data["vs"][name][key],
            )
        cache = self.qdc.params
        if cache is None:
            nx = self.var.N
            Q = sparse.csc_matrix((nx, nx))
            c = np.zeros(nx)
            K = 0.0
            for entry in self.qdc.order:
                N = self.getN("qdc", entry["name"], entry["idx"])
                Qk, ck, kk, vs = self.params_quad_cost(entry["name"], entry["idx"])
                if vs:
                    jj = self.varsets_idx(vs) - 1
                else:
                    jj = np.arange(max(Qk.shape[0], ck.shape[0] if ck.size else 0))
                if Qk.shape == (0, 0):
                    pass
                elif Qk.shape[1] == 1:
                    Q = Q + sparse.csc_matrix((np.asarray(Qk).reshape(-1), (jj, jj)), shape=(nx, nx))
                else:
                    qfull = sparse.lil_matrix((nx, nx))
                    qfull[np.ix_(jj, jj)] = Qk
                    Q = Q + qfull.tocsr()
                if ck.size:
                    c[jj] += ck
                kk_arr = np.asarray(kk, dtype=float).reshape(-1)
                if kk_arr.size > 1:
                    K += float(np.sum(kk_arr))
                elif kk_arr.size == 1:
                    K += float(N * kk_arr[0])
            self.qdc.params = {"Q": Q, "c": c, "k": K}
            cache = self.qdc.params
        return cache["Q"], cache["c"], cache["k"], []

    def params_nln_constraint(self, iseq: int, name: str, idx: list[int] | None = None):
        om_nlx = self.nle if iseq else self.nli
        idx = [] if idx is None else idx
        if not idx:
            dims = np.shape(om_nlx.idx.i1[name])
            if np.prod(dims) != 1:
                raise ValueError(
                    f"@opt_model/params_nln_constraint: nonlinear constraint set '{name}' requires an IDX_LIST arg"
                )
            N = om_nlx.idx.N[name]
            fcn = om_nlx.data["fcn"].get(name, "")
            hess = om_nlx.data["hess"].get(name, "")
            vs = om_nlx.data["vs"].get(name, [])
            include = om_nlx.data.get("include", {}).get(name, "")
        else:
            key = tuple(idx)
            N = om_nlx.idx.N[name][tuple(i - 1 for i in idx)]
            fcn = om_nlx.data["fcn"].get(name, {}).get(key, "")
            hess = om_nlx.data["hess"].get(name, {}).get(key, "")
            vs = om_nlx.data["vs"].get(name, {}).get(key, [])
            include = ""
        return int(N), fcn, hess, vs, include

    def params_nln_cost(self, name: str, idx: list[int] | None = None):
        idx = [] if idx is None else idx
        if not idx:
            dims = np.shape(self.nlc.idx.i1[name])
            if np.prod(dims) != 1:
                raise ValueError(
                    f"@opt_model/params_nln_cost: general nonlinear cost set '{name}' requires an IDX_LIST arg"
                )
            return int(self.nlc.idx.N[name]), self.nlc.data["fcn"][name], self.nlc.data["vs"][name]
        key = tuple(idx)
        return (
            int(self.nlc.idx.N[name][tuple(i - 1 for i in idx)]),
            self.nlc.data["fcn"][name][key],
            self.nlc.data["vs"][name][key],
        )

    def eval_nln_constraint(self, x: np.ndarray, iseq: int, name: str | None = None, idx: list[int] | None = None):
        om_nlx = self.nle if iseq else self.nli
        x = np.asarray(x, dtype=float).reshape(-1)
        if name is None:
            g = np.full(om_nlx.N, np.nan)
            dg_rows = []
            dg_cols = []
            dg_data = []
            for entry in om_nlx.order:
                entry_name = entry["name"]
                entry_idx = entry["idx"]
                if not entry_idx:
                    if entry_name not in om_nlx.data["fcn"]:
                        continue
                    N = int(om_nlx.idx.N[entry_name])
                    include = om_nlx.data.get("include", {}).get(entry_name)
                    if include:
                        N += int(np.sum(include["N"]))
                    fcn = om_nlx.data["fcn"][entry_name]
                    i1 = int(om_nlx.idx.i1[entry_name])
                    iN = i1 + N - 1
                    vs = om_nlx.data["vs"][entry_name]
                else:
                    key = tuple(entry_idx)
                    ref = tuple(i - 1 for i in entry_idx)
                    N = int(om_nlx.idx.N[entry_name][ref])
                    if N == 0:
                        continue
                    fcn = om_nlx.data["fcn"][entry_name][key]
                    i1 = int(om_nlx.idx.i1[entry_name][ref])
                    iN = int(om_nlx.idx.iN[entry_name][ref])
                    vs = om_nlx.data["vs"][entry_name][key]
                if N:
                    xx = self.varsets_x(x, vs)
                    try:
                        gk, dgk = fcn(xx)
                        want_grad = True
                    except Exception:
                        gk = fcn(xx)
                        dgk = None
                        want_grad = False
                    g[i1 - 1 : iN] = np.asarray(gk, dtype=float).reshape(-1)
                    if want_grad and dgk is not None:
                        gradient = sparse.csc_matrix(dgk)
                        if not vs:
                            if int(np.asarray(gradient.shape)[1]) == self.var.N:
                                gradient = gradient.tocoo()
                                dg_rows.append(gradient.row + (i1 - 1))
                                dg_cols.append(gradient.col)
                                dg_data.append(gradient.data)
                            else:
                                gradient = gradient.tocoo()
                                dg_rows.append(gradient.row + (i1 - 1))
                                dg_cols.append(gradient.col)
                                dg_data.append(gradient.data)
                        else:
                            jj = self.varsets_idx(vs) - 1
                            gradient = gradient.tocoo()
                            dg_rows.append(gradient.row + (i1 - 1))
                            dg_cols.append(jj[gradient.col])
                            dg_data.append(gradient.data)
            if dg_rows:
                dg = sparse.coo_matrix(
                    (np.concatenate(dg_data), (np.concatenate(dg_rows), np.concatenate(dg_cols))),
                    shape=(om_nlx.N, self.var.N),
                ).tocsr()
            else:
                dg = sparse.csc_matrix((om_nlx.N, self.var.N))
            return g, dg
        idx = [] if idx is None else idx
        if not idx and np.prod(np.shape(om_nlx.idx.i1[name])) != 1:
            raise ValueError(
                f"@opt_model/eval_nln_constraint: nonlinear constraint set '{name}' requires an IDX_LIST arg"
            )
        _, fcn, _, vs, _ = self.params_nln_constraint(iseq, name, idx)
        xx = self.varsets_x(x, vs)
        return fcn(xx)

    def eval_nln_constraint_hess(self, x: np.ndarray, lam: np.ndarray, iseq: int):
        om_nlx = self.nle if iseq else self.nli
        x = np.asarray(x, dtype=float).reshape(-1)
        lam = np.asarray(lam, dtype=float).reshape(-1)
        d2G_rows = []
        d2G_cols = []
        d2G_data = []
        for entry in om_nlx.order:
            entry_name = entry["name"]
            entry_idx = entry["idx"]
            if not entry_idx:
                if entry_name not in om_nlx.data["hess"]:
                    continue
                N = int(om_nlx.idx.N[entry_name])
                include = om_nlx.data.get("include", {}).get(entry_name)
                if include:
                    N += int(np.sum(include["N"]))
                if N == 0:
                    continue
                d2G_fcn = om_nlx.data["hess"][entry_name]
                i1 = int(om_nlx.idx.i1[entry_name])
                iN = i1 + N - 1
                vs = om_nlx.data["vs"][entry_name]
            else:
                key = tuple(entry_idx)
                ref = tuple(i - 1 for i in entry_idx)
                N = int(om_nlx.idx.N[entry_name][ref])
                if N == 0:
                    continue
                d2G_fcn = om_nlx.data["hess"][entry_name][key]
                i1 = int(om_nlx.idx.i1[entry_name][ref])
                iN = int(om_nlx.idx.iN[entry_name][ref])
                vs = om_nlx.data["vs"][entry_name][key]
            xx = self.varsets_x(x, vs)
            d2Gk = sparse.csc_matrix(d2G_fcn(xx, lam[i1 - 1 : iN]))
            d2Gk = d2Gk.tocoo()
            if not vs:
                d2G_rows.append(d2Gk.row)
                d2G_cols.append(d2Gk.col)
                d2G_data.append(d2Gk.data)
            else:
                jj = self.varsets_idx(vs) - 1
                d2G_rows.append(jj[d2Gk.row])
                d2G_cols.append(jj[d2Gk.col])
                d2G_data.append(d2Gk.data)
        if d2G_rows:
            return sparse.coo_matrix(
                (np.concatenate(d2G_data), (np.concatenate(d2G_rows), np.concatenate(d2G_cols))),
                shape=(self.var.N, self.var.N),
            ).tocsr()
        return sparse.csc_matrix((self.var.N, self.var.N))

    def eval_nln_cost(self, x: np.ndarray, name: str | None = None, idx: list[int] | None = None):
        x = np.asarray(x, dtype=float).reshape(-1)
        if self.nlc.N == 0:
            nx = len(x)
            return 0.0, np.zeros(nx), sparse.csc_matrix((nx, nx))
        if name is None:
            f = 0.0
            nx = self.var.N
            df = np.zeros(nx)
            d2f = sparse.csc_matrix((nx, nx))
            for entry in self.nlc.order:
                N, fcn, vs = self.params_nln_cost(entry["name"], entry["idx"])
                if N != 1:
                    raise ValueError("@opt_model/eval_nln_cost: not yet implemented for vector valued functions")
                xx = self.varsets_x(x, vs)
                fk, dfk, d2fk = fcn(xx)
                f += float(fk)
                dfk = np.asarray(dfk, dtype=float).reshape(-1)
                d2fk = sparse.csc_matrix(d2fk)
                nk = len(dfk)
                if not vs:
                    if nk == nx:
                        df = df + dfk
                        d2f = d2f + d2fk
                    else:
                        df[:nk] += dfk
                        d2fk_all_cols = sparse.lil_matrix((nk, nx))
                        d2fk_all_cols[:, :nk] = d2fk
                        d2f_full = sparse.lil_matrix((nx, nx))
                        d2f_full[:, :nk] = d2fk_all_cols.T
                        d2f = d2f + d2f_full.tocsr()
                else:
                    jj = self.varsets_idx(vs) - 1
                    df[jj] += dfk
                    d2fk_all_cols = sparse.lil_matrix((nk, nx))
                    d2fk_all_cols[:, jj] = d2fk
                    d2f_full = sparse.lil_matrix((nx, nx))
                    d2f_full[:, jj] = d2fk_all_cols.T
                    d2f = d2f + d2f_full.tocsr()
            return f, df, d2f
        idx = [] if idx is None else idx
        if idx or np.prod(np.shape(self.nlc.idx.i1[name])) == 1:
            N, fcn, vs = self.params_nln_cost(name, idx)
            if N != 1:
                raise ValueError("@opt_model/eval_nln_cost: not yet implemented for vector valued functions")
            xx = self.varsets_x(x, vs)
            return fcn(xx)
        raise ValueError(
            f"@opt_model/eval_nln_cost: general nonlinear cost set '{name}' requires an IDX_LIST arg when requesting DF output"
        )

    def eval_quad_cost(self, x: np.ndarray, name: str | None = None, idx: list[int] | None = None):
        x = np.asarray(x, dtype=float).reshape(-1)
        if self.qdc.N == 0:
            return 0.0, np.array([]), sparse.csc_matrix((0, 0))
        if name is None:
            Q, c, k, vs = self.params_quad_cost()
            N = 1
        else:
            idx = [] if idx is None else idx
            if not idx and np.prod(np.shape(self.qdc.idx.i1[name])) != 1:
                raise ValueError(
                    f"@opt_model/eval_quad_cost: quadratic cost set '{name}' requires an IDX_LIST arg when requesting DF output"
                )
            Q, c, k, vs = self.params_quad_cost(name, idx)
            N = self.getN("qdc", name, idx)
        xx = self.varsets_x(x, vs, "vector")
        if N == 1:
            f = float(np.sum(k)) if np.size(k) else 0.0
            if c.size:
                f = f + float(c.T @ xx)
            if Q.shape != (0, 0):
                f = f + float((xx.T @ (Q @ xx)) / 2.0)
        else:
            if c.size == 0:
                f = np.asarray(Q).reshape(-1) * xx**2 / 2 + k
            elif Q.shape == (0, 0):
                f = c * xx + k
            else:
                f = np.asarray(Q).reshape(-1) * xx**2 / 2 + c * xx + k
        if c.size:
            df = c.copy()
        else:
            df = np.zeros_like(xx)
        if Q.shape != (0, 0):
            if N == 1:
                df = df + Q @ xx
            else:
                df = df + np.asarray(Q).reshape(-1) * xx
        if Q.shape == (0, 0):
            nx = len(xx)
            d2f = sparse.csc_matrix((nx, nx if N == 1 else 1))
        else:
            d2f = Q
        return f, df, d2f

    def is_mixed_integer(self) -> bool:
        _, _, _, vt = self.params_var()
        return any(ch in ("I", "B") for ch in vt)

    def problem_type(self, recheck: bool = False) -> str:
        if not self.prob_type or recheck:
            nleN = self.getN("nle")
            nliN = self.getN("nli")
            nlcN = self.getN("nlc")
            qdcN = self.getN("qdc")
            linN = self.getN("lin")
            varN = self.getN("var")
            if nlcN or qdcN:
                if nliN or nleN or nlcN:
                    prob = "NLP"
                else:
                    H, _, _, _ = self.params_quad_cost()
                    prob = "LP" if H.shape == (0, 0) or H.nnz == 0 else "QP"
            else:
                if nliN:
                    raise ValueError(
                        "@opt_model/problem_type: invalid problem - nonlinear inequality constraints with no costs"
                    )
                if nleN + linN == varN:
                    A, l, u, _, _, _ = self.params_lin_constraint()
                    if linN and np.any(l != u):
                        raise ValueError(
                            "@opt_model/problem_type: invalid problem - linear inequality constraints with no costs"
                        )
                    prob = "NLEQ" if nleN else "LEQ"
                else:
                    raise ValueError("@opt_model/problem_type: invalid problem - non-square system with no costs")
            if self.is_mixed_integer() and prob != "NLEQ":
                prob = f"MI{prob}"
            self.prob_type = prob
        return self.prob_type

    def solve(self, opt: dict[str, Any] | None = None):
        opt = {} if opt is None else dict(opt)
        pt = self.problem_type()
        if pt in {"LP", "QP"}:
            H, C, C0, _ = self.params_quad_cost()
            A, l, u, _, _, _ = self.params_lin_constraint()
            x0, xmin, xmax, _ = self.params_var()
            if "x0" in opt:
                x0 = opt["x0"]
            x, f, eflag, output, lambda_ = qps_master_full(H, C, A, l, u, xmin, xmax, x0, opt)
            f = f + C0
        elif pt == "NLP":
            A, l, u, _, _, _ = self.params_lin_constraint()
            x0, xmin, xmax, _ = self.params_var()
            if "x0" in opt:
                x0 = opt["x0"]
            f_fcn = lambda x: self.eval_costfcn(x)
            gh_fcn = lambda x: self.eval_consfcn(x)
            hess_fcn = lambda x, lambda_, cost_mult=1.0: self.eval_hessfcn(x, lambda_, cost_mult)
            x, f, eflag, output, lambda_ = nlps_master_full(f_fcn, x0, A, l, u, xmin, xmax, gh_fcn, hess_fcn, opt)
        else:
            raise NotImplementedError(f"{pt} solve not yet implemented")
        self.soln = {"eflag": eflag, "x": x, "f": f, "output": output, "lambda": lambda_}
        return x, f, eflag, output, lambda_

    def eval_costfcn(self, x: np.ndarray):
        from .nlp_costfcn import nlp_costfcn

        return nlp_costfcn(self, x, nargout=3)

    def eval_consfcn(self, x: np.ndarray):
        from .nlp_consfcn import nlp_consfcn

        return nlp_consfcn(self, x, nargout=4)

    def eval_hessfcn(self, x: np.ndarray, lambda_: dict[str, Any], cost_mult: float = 1.0):
        from .nlp_hessfcn import nlp_hessfcn

        return nlp_hessfcn(self, x, lambda_, cost_mult, nargout=1)

    def get_idx(self, *set_types: str):
        if not set_types:
            return self.var.idx, self.lin.idx, self.nle.idx, self.nli.idx, self.qdc.idx, self.nlc.idx
        return super().get_idx(*set_types)
