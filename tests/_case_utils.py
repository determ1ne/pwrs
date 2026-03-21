import copy
import importlib
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "src" / "pwrs" / "data"


@lru_cache(maxsize=16)
def _load_case_cached(case_name: str):
    module = importlib.import_module(f".{case_name}", package="pwrs.data")
    return getattr(module, case_name)()


def load_case(case_name: str):
    return copy.deepcopy(_load_case_cached(case_name))


@lru_cache(maxsize=1)
def case_names() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in DATA_DIR.glob("case*.py")))


@lru_cache(maxsize=None)
def case_bus_count(case_name: str) -> int:
    return int(_load_case_cached(case_name)["bus"].shape[0])


def quick_case_names(max_buses: int = 300) -> list[str]:
    return [case_name for case_name in case_names() if case_bus_count(case_name) <= max_buses]


def large_case_names(max_buses: int = 300) -> list[str]:
    return [case_name for case_name in case_names() if case_bus_count(case_name) > max_buses]
