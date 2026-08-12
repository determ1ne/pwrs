# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
from ..._loader import discover_case_names, load_case, register_case_modules

__all__ = discover_case_names(__name__, "pglib_opf_case")  # pyright: ignore[reportUnsupportedDunderAll]

for _case_name in __all__:
    globals()[_case_name] = load_case(_case_name, __name__)

register_case_modules(__name__, __all__)
