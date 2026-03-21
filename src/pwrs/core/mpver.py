# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause


def mpver(*args, nargout=None):
    """Return or print the pwrs version record.

    Mimics MATPOWER's ``mpver`` behavior. When output arguments are requested
    it returns either the version string or the full version struct,
    otherwise it prints the formatted version banner.

    Parameters
    ----------
    *args
        Optional MATLAB-style selector arguments. Any non-empty argument list
        requests the full version record when outputs are requested.
    nargout : int, optional
        MATLAB compatibility flag controlling whether values are returned or
        the banner is printed.

    Returns
    -------
    str or dict or None
        Version string, version record dict, or ``None`` when printing.
    """
    v = {
        "Name": "pwrs",
        "Version": "7.1",
        "Release": "",
        "Date": "08-Oct-2020",
    }
    if nargout is None:
        nargout = 0
    if nargout > 0:
        if len(args) > 0:
            return v
        return v["Version"]
    print(f"\n{v['Name']:<22s} Version {v['Version']:<9s}  {v['Date']:>11s}\n")
    print("  pwrs is distributed under the 3-clause BSD License.")
    print("  Please see the LICENSE file for details.\n")
