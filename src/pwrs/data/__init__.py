# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
from ._loader import register_case_modules
from .matpower import *  # noqa: F403
from .matpower import __all__ as _matpower_all

__all__ = list(_matpower_all)

register_case_modules(__name__, __all__, data_package_name="pwrs.data.matpower")
