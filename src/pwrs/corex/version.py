# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from importlib.metadata import PackageNotFoundError, version

try:
    version = version("pwrs")
except PackageNotFoundError:
    version = "unknown"
