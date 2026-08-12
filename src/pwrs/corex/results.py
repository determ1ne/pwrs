# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

"""Structured public result types for MATPOWER-compatible analyses."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field, fields
from typing import ClassVar, NamedTuple, Self, cast, override

import numpy as np

from .types import ComplexArray, FloatArray, IntArray, Matrix


def _float_array(value: object) -> FloatArray:
    return np.asarray(value, dtype=np.float64)


def _int_array(value: object) -> IntArray:
    return np.asarray(value, dtype=np.int64)


def _index_map(value: object) -> IntArray | dict[int, int]:
    if isinstance(value, Mapping):
        return {int(key): int(item) for key, item in value.items()}
    return _int_array(value)


def _array(value: object) -> FloatArray | ComplexArray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        return np.asarray(array, dtype=np.complex128)
    return np.asarray(array, dtype=np.float64)


def _float_value(value: object) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _int_value(value: object) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _bool_value(value: object) -> bool:
    return bool(np.asarray(value).reshape(-1)[0])


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return cast(Mapping[str, object], value)


def _to_legacy(value: object) -> object:
    if isinstance(value, StructuredMapping):
        return value.to_dict()
    if isinstance(value, list):
        return [_to_legacy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_legacy(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _to_legacy(item) for key, item in value.items()}
    return value


@dataclass
class StructuredMapping(MutableMapping[str, object]):
    """Dataclass with a mapping compatibility layer and extension fields."""

    _extra: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _required_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def _field_names(cls) -> tuple[str, ...]:
        return tuple(item.name for item in fields(cls) if item.name != "_extra")

    def __getitem__(self, key: str) -> object:
        if key in self._field_names():
            value = getattr(self, key)
            if value is None:
                raise KeyError(key)
            return value
        try:
            return self._extra[key]
        except KeyError:
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: object) -> None:
        if key in self._field_names():
            setattr(self, key, value)
        else:
            self._extra[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._required_fields:
            raise KeyError(f"cannot delete required field {key!r}")
        if key in self._field_names():
            if getattr(self, key) is None:
                raise KeyError(key)
            setattr(self, key, None)
            return
        try:
            del self._extra[key]
        except KeyError:
            raise KeyError(key) from None

    def __iter__(self) -> Iterator[str]:
        for name in self._field_names():
            if getattr(self, name) is not None:
                yield name
        yield from self._extra

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __getattr__(self, name: str) -> object:
        extra = self.__dict__.get("_extra", {})
        if name in extra:
            return extra[name]
        raise AttributeError(name)

    def to_dict(self) -> dict[str, object]:
        """Return a MATPOWER-compatible mapping, including extension fields."""
        return {name: _to_legacy(self[name]) for name in self}

    def _store_extras(self, data: Mapping[str, object]) -> None:
        known = self._field_names()
        self._extra.update({key: value for key, value in data.items() if key not in known})


@dataclass
class ResultOrderStatus(StructuredMapping):
    on: IntArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    off: IntArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> ResultOrderStatus:
        result = cls(on=_int_array(data.get("on", [])), off=_int_array(data.get("off", [])))
        result._store_extras(data)
        return result


@dataclass
class ResultOrderEntity(StructuredMapping):
    e2i: IntArray | dict[int, int] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    i2e: IntArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    status: ResultOrderStatus = field(default_factory=ResultOrderStatus)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> ResultOrderEntity:
        status = ResultOrderStatus.from_mapping(_mapping(data.get("status", {}), "order status"))
        result = cls(
            e2i=_index_map(data.get("e2i", [])),
            i2e=_int_array(data.get("i2e", [])),
            status=status,
        )
        result._store_extras(data)
        return result


@dataclass
class ResultOrderBranch(StructuredMapping):
    status: ResultOrderStatus = field(default_factory=ResultOrderStatus)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> ResultOrderBranch:
        result = cls(status=ResultOrderStatus.from_mapping(_mapping(data.get("status", {}), "branch status")))
        result._store_extras(data)
        return result


@dataclass
class ResultOrderCase(StructuredMapping):
    bus: FloatArray | None = None
    branch: FloatArray | None = None
    gen: FloatArray | None = None
    gencost: FloatArray | None = None
    A: Matrix | None = None
    N: Matrix | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> ResultOrderCase:
        result = cls(
            bus=_float_array(data["bus"]) if "bus" in data else None,
            branch=_float_array(data["branch"]) if "branch" in data else None,
            gen=_float_array(data["gen"]) if "gen" in data else None,
            gencost=_float_array(data["gencost"]) if "gencost" in data else None,
            A=cast(Matrix, data.get("A")),
            N=cast(Matrix, data.get("N")),
        )
        result._store_extras(data)
        return result


@dataclass
class ResultOrder(StructuredMapping):
    state: str = "e"
    ext: ResultOrderCase | None = None
    int: ResultOrderCase | None = None
    bus: ResultOrderEntity = field(default_factory=ResultOrderEntity)
    gen: ResultOrderEntity = field(default_factory=ResultOrderEntity)
    branch: ResultOrderBranch = field(default_factory=ResultOrderBranch)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> ResultOrder:
        ext_value = data.get("ext")
        int_value = data.get("int")
        result = cls(
            state=str(data.get("state", "e")),
            ext=ResultOrderCase.from_mapping(_mapping(ext_value, "order.ext")) if ext_value is not None else None,
            int=ResultOrderCase.from_mapping(_mapping(int_value, "order.int")) if int_value is not None else None,
            bus=ResultOrderEntity.from_mapping(_mapping(data.get("bus", {}), "order.bus")),
            gen=ResultOrderEntity.from_mapping(_mapping(data.get("gen", {}), "order.gen")),
            branch=ResultOrderBranch.from_mapping(_mapping(data.get("branch", {}), "order.branch")),
        )
        result._store_extras(data)
        return result


@dataclass
class CpfEvent(StructuredMapping):
    eidx: int = 0
    name: str = ""
    zero: bool = False
    idx: IntArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    step_scale: float | FloatArray = 1.0
    log: bool = False
    msg: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> CpfEvent:
        scale = np.asarray(data.get("step_scale", 1.0))
        result = cls(
            eidx=_int_value(data.get("eidx", 0)),
            name=str(data.get("name", "")),
            zero=_bool_value(data.get("zero", False)),
            idx=_int_array(data.get("idx", [])),
            step_scale=_float_value(scale) if scale.size == 1 else _float_array(scale),
            log=_bool_value(data.get("log", False)),
            msg=str(data.get("msg", "")),
        )
        result._store_extras(data)
        return result


@dataclass
class CpfTrace(StructuredMapping):
    V: ComplexArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.complex128))
    V_hat: ComplexArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.complex128))
    lam: FloatArray = field(default_factory=lambda: np.empty((1, 0)))
    lam_hat: FloatArray = field(default_factory=lambda: np.empty((1, 0)))
    steps: FloatArray = field(default_factory=lambda: np.empty((1, 0)))
    iterations: int = 0
    max_lam: float = 0.0
    events: list[CpfEvent] = field(default_factory=list)
    done_msg: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> CpfTrace:
        event_value = data.get("events", [])
        raw_events = event_value if isinstance(event_value, list) else [event_value]
        events = [CpfEvent.from_mapping(_mapping(event, "CPF event")) for event in raw_events if event is not None]
        result = cls(
            V=np.asarray(data.get("V", np.empty((0, 0))), dtype=np.complex128),
            V_hat=np.asarray(data.get("V_hat", np.empty((0, 0))), dtype=np.complex128),
            lam=_float_array(data.get("lam", np.empty((1, 0)))),
            lam_hat=_float_array(data.get("lam_hat", np.empty((1, 0)))),
            steps=_float_array(data.get("steps", np.empty((1, 0)))),
            iterations=_int_value(data.get("iterations", 0)),
            max_lam=_float_value(data.get("max_lam", 0.0)),
            events=events,
            done_msg=str(data.get("done_msg", "")),
        )
        result._store_extras(data)
        return result


@dataclass
class BoundMultipliers(StructuredMapping):
    """Lower and upper multipliers; legacy keys are ``l`` and ``u``."""

    lower: FloatArray = field(default_factory=lambda: np.empty(0))
    upper: FloatArray = field(default_factory=lambda: np.empty(0))
    _required_fields: ClassVar[frozenset[str]] = frozenset({"lower", "upper"})

    def __getitem__(self, key: str) -> object:
        return super().__getitem__({"l": "lower", "u": "upper"}.get(key, key))

    def __setitem__(self, key: str, value: object) -> None:
        super().__setitem__({"l": "lower", "u": "upper"}.get(key, key), value)

    def __iter__(self) -> Iterator[str]:
        yield "l"
        yield "u"
        yield from self._extra

    def to_dict(self) -> dict[str, object]:
        return {"l": self.lower, "u": self.upper, **self._extra}

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> BoundMultipliers:
        result = cls(lower=_float_array(data.get("l", [])), upper=_float_array(data.get("u", [])))
        result._extra.update({key: value for key, value in data.items() if key not in {"l", "u"}})
        return result


@dataclass
class OpfMultipliers(StructuredMapping):
    var: BoundMultipliers | None = None
    nln: BoundMultipliers | None = None
    lin: BoundMultipliers | None = None
    nle: FloatArray | None = None
    nli: FloatArray | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> OpfMultipliers:
        def bounds(name: str) -> BoundMultipliers | None:
            value = data.get(name)
            return BoundMultipliers.from_mapping(_mapping(value, f"mu.{name}")) if value is not None else None

        result = cls(
            var=bounds("var"),
            nln=bounds("nln"),
            lin=bounds("lin"),
            nle=_float_array(data["nle"]) if "nle" in data else None,
            nli=_float_array(data["nli"]) if "nli" in data else None,
        )
        result._store_extras(data)
        return result


def _named_arrays(value: object, field_name: str) -> dict[str, FloatArray | ComplexArray]:
    return {name: _array(item) for name, item in _mapping(value, field_name).items()}


@dataclass
class NamedBoundMultipliers(StructuredMapping):
    """Named lower and upper multiplier vectors."""

    lower: dict[str, FloatArray | ComplexArray] = field(default_factory=dict)
    upper: dict[str, FloatArray | ComplexArray] = field(default_factory=dict)

    def __getitem__(self, key: str) -> object:
        return super().__getitem__({"l": "lower", "u": "upper"}.get(key, key))

    def __iter__(self) -> Iterator[str]:
        yield "l"
        yield "u"
        yield from self._extra

    def to_dict(self) -> dict[str, object]:
        return {"l": self.lower, "u": self.upper, **self._extra}

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> NamedBoundMultipliers:
        result = cls(
            lower=_named_arrays(data.get("l", {}), "lower named multipliers"),
            upper=_named_arrays(data.get("u", {}), "upper named multipliers"),
        )
        result._extra.update({key: value for key, value in data.items() if key not in {"l", "u"}})
        return result


@dataclass
class OpfVariableResults(StructuredMapping):
    """Named OPF variable values and bound multipliers."""

    values_by_name: dict[str, FloatArray | ComplexArray] = field(default_factory=dict)
    mu: NamedBoundMultipliers = field(default_factory=NamedBoundMultipliers)

    def __getitem__(self, key: str) -> object:
        return super().__getitem__("values_by_name" if key == "val" else key)

    def __iter__(self) -> Iterator[str]:
        yield "val"
        yield "mu"
        yield from self._extra

    def to_dict(self) -> dict[str, object]:
        return {"val": self.values_by_name, "mu": self.mu.to_dict(), **self._extra}

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> OpfVariableResults:
        result = cls(
            values_by_name=_named_arrays(data.get("val", {}), "variable values"),
            mu=NamedBoundMultipliers.from_mapping(_mapping(data.get("mu", {}), "variable multipliers")),
        )
        result._extra.update({key: value for key, value in data.items() if key not in {"val", "mu"}})
        return result


@dataclass
class OpfLinearConstraintResults(StructuredMapping):
    """Named linear-constraint multipliers."""

    mu: NamedBoundMultipliers = field(default_factory=NamedBoundMultipliers)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> OpfLinearConstraintResults:
        result = cls(mu=NamedBoundMultipliers.from_mapping(_mapping(data.get("mu", {}), "linear multipliers")))
        result._store_extras(data)
        return result


@dataclass
class OpfNonlinearEqualityResults(StructuredMapping):
    """Named nonlinear-equality multipliers; legacy key is ``lambda``."""

    multipliers: dict[str, FloatArray | ComplexArray] = field(default_factory=dict)

    def __getitem__(self, key: str) -> object:
        return super().__getitem__("multipliers" if key == "lambda" else key)

    def __iter__(self) -> Iterator[str]:
        yield "lambda"
        yield from self._extra

    def to_dict(self) -> dict[str, object]:
        return {"lambda": self.multipliers, **self._extra}

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> OpfNonlinearEqualityResults:
        result = cls(multipliers=_named_arrays(data.get("lambda", {}), "nonlinear equality multipliers"))
        result._extra.update({key: value for key, value in data.items() if key != "lambda"})
        return result


@dataclass
class OpfNonlinearInequalityResults(StructuredMapping):
    """Named nonlinear-inequality multipliers; legacy key is ``mu``."""

    multipliers: dict[str, FloatArray | ComplexArray] = field(default_factory=dict)

    def __getitem__(self, key: str) -> object:
        return super().__getitem__("multipliers" if key == "mu" else key)

    def __iter__(self) -> Iterator[str]:
        yield "mu"
        yield from self._extra

    def to_dict(self) -> dict[str, object]:
        return {"mu": self.multipliers, **self._extra}

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> OpfNonlinearInequalityResults:
        result = cls(multipliers=_named_arrays(data.get("mu", {}), "nonlinear inequality multipliers"))
        result._extra.update({key: value for key, value in data.items() if key != "mu"})
        return result


@dataclass
class PowerModelConstraintResult(StructuredMapping):
    """Dual and feasibility data for one named extension constraint group."""

    dual: FloatArray = field(default_factory=lambda: np.empty(0))
    native_dual: FloatArray = field(default_factory=lambda: np.empty(0))
    senses: tuple[str, ...] = ()
    violation: FloatArray = field(default_factory=lambda: np.empty(0))
    max_violation: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> PowerModelConstraintResult:
        senses_value = data.get("senses", ())
        if not isinstance(senses_value, (list, tuple)):
            raise TypeError("extension constraint senses must be a sequence")
        result = cls(
            dual=_float_array(data.get("dual", [])),
            native_dual=_float_array(data.get("native_dual", [])),
            senses=tuple(str(value) for value in senses_value),
            violation=_float_array(data.get("violation", [])),
            max_violation=_float_value(data.get("max_violation", 0.0)),
        )
        result._store_extras(data)
        return result


@dataclass
class PowerModelExtensionResult(StructuredMapping):
    """Structured values contributed by one PowerModels extension callback."""

    build_time: float = 0.0
    constraints: dict[str, PowerModelConstraintResult] = field(default_factory=dict)
    variables: dict[str, FloatArray | ComplexArray] = field(default_factory=dict)
    objective_terms: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> PowerModelExtensionResult:
        constraint_data = _mapping(data.get("constraints", {}), "extension constraints")
        variable_data = _mapping(data.get("variables", {}), "extension variables")
        objective_data = _mapping(data.get("objective_terms", {}), "extension objective terms")
        result = cls(
            build_time=_float_value(data.get("build_time", 0.0)),
            constraints={
                name: PowerModelConstraintResult.from_mapping(_mapping(value, name))
                for name, value in constraint_data.items()
            },
            variables={name: _array(value) for name, value in variable_data.items()},
            objective_terms={name: _float_value(value) for name, value in objective_data.items()},
        )
        result._store_extras(data)
        return result


def _named_numeric_values(value: object, field_name: str) -> dict[str, float | FloatArray | ComplexArray]:
    result: dict[str, float | FloatArray | ComplexArray] = {}
    for name, item in _mapping(value, field_name).items():
        array = np.asarray(item)
        result[name] = _float_value(array) if array.size == 1 and not np.iscomplexobj(array) else _array(array)
    return result


@dataclass
class CaseResult(StructuredMapping):
    """Common solved-case fields shared by PF, OPF and CPF results."""

    version: str = "2"
    baseMVA: float = 100.0
    bus: FloatArray = field(default_factory=lambda: np.empty((0, 0)))
    gen: FloatArray = field(default_factory=lambda: np.empty((0, 0)))
    branch: FloatArray = field(default_factory=lambda: np.empty((0, 0)))
    success: bool = False
    iterations: int = 0
    et: float = 0.0
    gencost: FloatArray | None = None
    dcline: FloatArray | None = None
    dclinecost: FloatArray | None = None
    areas: FloatArray | None = None
    gentype: list[str] | None = None
    genfuel: list[str] | None = None
    bus_name: list[str] | None = None
    branch_name: list[str] | None = None
    order: ResultOrder | None = None
    A: Matrix | None = None
    l: FloatArray | None = None
    u: FloatArray | None = None
    N: Matrix | None = None
    fparm: FloatArray | None = None
    H: Matrix | None = None
    Cw: FloatArray | None = None
    z0: FloatArray | None = None
    zl: FloatArray | None = None
    zu: FloatArray | None = None
    comments: list[str] | None = None
    userfcn: list[object] | None = None

    _required_fields: ClassVar[frozenset[str]] = frozenset({"baseMVA", "bus", "gen", "branch", "success", "et"})

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Self:
        missing = cls._required_fields.difference(data)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{cls.__name__} is missing required fields: {names}")
        result = cls(
            version=str(data.get("version", "2")),
            baseMVA=_float_value(data["baseMVA"]),
            bus=_float_array(data["bus"]),
            gen=_float_array(data["gen"]),
            branch=_float_array(data["branch"]),
            success=_bool_value(data["success"]),
            iterations=_int_value(data["iterations"]) if data.get("iterations") is not None else 0,
            et=_float_value(data["et"]),
            gencost=_float_array(data["gencost"]) if data.get("gencost") is not None else None,
            dcline=_float_array(data["dcline"]) if data.get("dcline") is not None else None,
            dclinecost=_float_array(data["dclinecost"]) if data.get("dclinecost") is not None else None,
            areas=_float_array(data["areas"]) if data.get("areas") is not None else None,
            gentype=cast(list[str] | None, data.get("gentype")),
            genfuel=cast(list[str] | None, data.get("genfuel")),
            bus_name=cast(list[str] | None, data.get("bus_name", data.get("busname"))),
            branch_name=cast(list[str] | None, data.get("branch_name", data.get("branchname"))),
            order=(
                ResultOrder.from_mapping(_mapping(data["order"], "order")) if data.get("order") is not None else None
            ),
            A=cast(Matrix, data.get("A")),
            l=_float_array(data["l"]) if data.get("l") is not None else None,
            u=_float_array(data["u"]) if data.get("u") is not None else None,
            N=cast(Matrix, data.get("N")),
            fparm=_float_array(data["fparm"]) if data.get("fparm") is not None else None,
            H=cast(Matrix, data.get("H")),
            Cw=_float_array(data["Cw"]) if data.get("Cw") is not None else None,
            z0=_float_array(data["z0"]) if data.get("z0") is not None else None,
            zl=_float_array(data["zl"]) if data.get("zl") is not None else None,
            zu=_float_array(data["zu"]) if data.get("zu") is not None else None,
            comments=cast(list[str] | None, data.get("comments")),
            userfcn=cast(list[object] | None, data.get("userfcn")),
        )
        result._store_extras(data)
        return result


@dataclass
class PowerFlowResult(CaseResult):
    """Solved AC or DC power-flow result."""

    iterations: int = 0
    _required_fields: ClassVar[frozenset[str]] = CaseResult._required_fields | {"iterations"}


@dataclass
class OptimalPowerFlowResult(CaseResult):
    """Solved AC or DC optimal-power-flow result."""

    f: float | FloatArray = 0.0
    x: FloatArray | ComplexArray | None = None
    mu: OpfMultipliers | None = None
    raw: Mapping[str, object] | None = None
    om: object | None = None
    var: OpfVariableResults | None = None
    lin: OpfLinearConstraintResults | None = None
    nle: OpfNonlinearEqualityResults | None = None
    nli: OpfNonlinearInequalityResults | None = None
    qdc: Mapping[str, float | FloatArray | ComplexArray] | None = None
    nlc: Mapping[str, float | FloatArray | ComplexArray] | None = None
    cost: Mapping[str, float | FloatArray | ComplexArray] | None = None
    extensions: dict[str, PowerModelExtensionResult] | None = None

    @classmethod
    @override
    def from_mapping(cls, data: Mapping[str, object]) -> Self:
        result = super().from_mapping(data)
        if not isinstance(result, cls):
            raise TypeError("unexpected result type")
        f_value = np.asarray(data.get("f", 0.0))
        result.f = _float_value(f_value) if f_value.size == 1 else _float_array(f_value)
        result.x = _array(data["x"]) if data.get("x") is not None else None
        result.mu = OpfMultipliers.from_mapping(_mapping(data["mu"], "mu")) if data.get("mu") is not None else None
        result.raw = _mapping(data["raw"], "raw") if data.get("raw") is not None else None
        result.om = data.get("om")
        result.var = (
            OpfVariableResults.from_mapping(_mapping(data["var"], "var")) if data.get("var") is not None else None
        )
        result.lin = (
            OpfLinearConstraintResults.from_mapping(_mapping(data["lin"], "lin"))
            if data.get("lin") is not None
            else None
        )
        result.nle = (
            OpfNonlinearEqualityResults.from_mapping(_mapping(data["nle"], "nle"))
            if data.get("nle") is not None
            else None
        )
        result.nli = (
            OpfNonlinearInequalityResults.from_mapping(_mapping(data["nli"], "nli"))
            if data.get("nli") is not None
            else None
        )
        for name in ("qdc", "nlc", "cost"):
            value = data.get(name)
            setattr(result, name, _named_numeric_values(value, name) if value is not None else None)
        extension_value = data.get("extensions")
        if extension_value is None and result.raw is not None:
            extension_value = result.raw.get("extensions")
        if extension_value is not None:
            extension_data = _mapping(extension_value, "extensions")
            result.extensions = {
                name: PowerModelExtensionResult.from_mapping(_mapping(value, name))
                for name, value in extension_data.items()
            }
        result._store_extras(data)
        return result


@dataclass
class ContinuationPowerFlowResult(CaseResult):
    """Continuation power-flow result with its typed continuation trace."""

    cpf: CpfTrace = field(default_factory=CpfTrace)

    @classmethod
    @override
    def from_mapping(cls, data: Mapping[str, object]) -> Self:
        result = super().from_mapping(data)
        if not isinstance(result, cls):
            raise TypeError("unexpected result type")
        result.cpf = CpfTrace.from_mapping(_mapping(data.get("cpf", {}), "cpf"))
        result._store_extras(data)
        return result


class PowerFlowExpandedResult(NamedTuple):
    baseMVA: float
    bus: FloatArray
    gen: FloatArray
    branch: FloatArray
    success: bool
    et: float


class OptimalPowerFlowExpandedResult(NamedTuple):
    baseMVA: float
    bus: FloatArray
    gen: FloatArray
    gencost: FloatArray | None
    branch: FloatArray
    f: float | FloatArray
    success: bool
    et: float


__all__ = [
    "BoundMultipliers",
    "CaseResult",
    "ContinuationPowerFlowResult",
    "CpfEvent",
    "CpfTrace",
    "NamedBoundMultipliers",
    "OpfLinearConstraintResults",
    "OpfMultipliers",
    "OpfNonlinearEqualityResults",
    "OpfNonlinearInequalityResults",
    "OpfVariableResults",
    "OptimalPowerFlowExpandedResult",
    "OptimalPowerFlowResult",
    "PowerFlowExpandedResult",
    "PowerFlowResult",
    "PowerModelConstraintResult",
    "PowerModelExtensionResult",
    "ResultOrder",
    "ResultOrderBranch",
    "ResultOrderCase",
    "ResultOrderEntity",
    "ResultOrderStatus",
    "StructuredMapping",
]
