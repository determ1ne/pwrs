# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Callable

from ...corex import MatpowerCase
from .._loader import discover_case_names, load_case, register_case_modules

__all__ = discover_case_names(__name__, "case")  # pyright: ignore[reportUnsupportedDunderAll]

for _case_name in __all__:
    globals()[_case_name] = load_case(_case_name, __name__)

register_case_modules(__name__, __all__)


def __getattr__(name: str) -> Callable[[], MatpowerCase]:
    """Provide a typed fallback for dynamically discovered case functions."""
    if name in __all__:
        return load_case(name, __name__)
    raise AttributeError(name)
