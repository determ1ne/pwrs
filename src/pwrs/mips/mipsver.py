# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause


def mipsver(selector: str | None = None, nargout=1):
    info = {"Name": "MIPS", "Version": "1.0", "Release": "", "Date": "Python port"}
    if selector == "all":
        return info
    return info["Version"]
