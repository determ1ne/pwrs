# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import os
import sys
from pathlib import Path


def _resolve_callable(fname):
    if callable(fname):
        return fname
    if not isinstance(fname, str):
        raise TypeError("feval_w_path: FNAME must be a string")
    module = importlib.import_module("pwrs")
    if hasattr(module, fname):
        return getattr(module, fname)
    raise AttributeError(f"feval_w_path: module {fname!r} does not define function {fname!r}")


def feval_w_path(fpath, fname, *args, nargout=None):
    """Evaluate a function with an optional temporary search path.

    Resolves and executes ``fname`` either from the current Python/MATPOWER
    dispatch environment or from the supplied directory path, temporarily
    adjusting the working directory and import path to mimic MATPOWER's
    ``feval_w_path`` behavior.

    Parameters
    ----------
    fpath : str
        Directory path to add temporarily before resolving ``fname``. Use an
        empty string to resolve in the current environment.
    fname : str
        Function name to resolve and call.
    *args
        Positional arguments passed to the resolved function.
    nargout : int, optional
        MATLAB compatibility flag controlling tuple slicing for multiple
        outputs.

    Returns
    -------
    Any
        Function return value, or the leading ``nargout`` items when the
        underlying function returns a tuple.
    """
    if not isinstance(fpath, str):
        raise TypeError("feval_w_path: FPATH must be a string")

    if fpath == "":
        fcn = _resolve_callable(fname)
        result = fcn(*args)
    else:
        path = Path(fpath)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_dir():
            raise ValueError(f"feval_w_path: Sorry, '{fpath}' is not a valid directory path.")

        cwd = Path.cwd()
        path_str = str(path.resolve())
        sys_path_added = False
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            sys_path_added = True
        try:
            os.chdir(path_str)
            if not isinstance(fname, str):
                raise TypeError("feval_w_path: FNAME must be a string")
            if fname in sys.modules:
                module = importlib.reload(sys.modules[fname])
            else:
                module = importlib.import_module(fname)
            if not hasattr(module, fname):
                raise AttributeError(f"feval_w_path: module {fname!r} does not define function {fname!r}")
            result = getattr(module, fname)(*args)
        finally:
            os.chdir(cwd)
            if sys_path_added:
                sys.path.remove(path_str)

    if nargout is None or nargout <= 1:
        return result
    if not isinstance(result, tuple):
        raise TypeError(f"expected at least {nargout} outputs, got {type(result).__name__}")
    return result[:nargout]
