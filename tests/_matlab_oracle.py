from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.io import loadmat, savemat
from scipy.io.matlab import mat_struct

REPO_ROOT = Path(__file__).resolve().parents[1]


def _from_matlab(value: Any) -> Any:
    if isinstance(value, mat_struct):
        return {field: _from_matlab(getattr(value, field)) for field in value._fieldnames or []}
    if sparse.issparse(value):
        return value
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.ndim == 0:
            return _from_matlab(value.item())
        flat = [_from_matlab(item) for item in value.flat]
        return flat if value.ndim == 1 else np.array(flat, dtype=object).reshape(value.shape)
    return value


def _to_matlab(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_matlab(item) for key, item in value.items()}
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        arr = np.empty((1, len(value)), dtype=object)
        for idx, item in enumerate(value):
            arr[0, idx] = _to_matlab(item)
        return arr
    return value


def _matlab_quote(value: str) -> str:
    return value.replace("'", "''")


def _oracle_to_matlab(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _oracle_to_matlab(item) for key, item in value.items()}
    if sparse.issparse(value):
        if np.issubdtype(value.dtype, np.integer):
            return value.astype(float)
        return value
    if isinstance(value, np.ndarray):
        out = value.astype(float) if np.issubdtype(value.dtype, np.integer) else value
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        return out
    if isinstance(value, (list, tuple)):
        return type(value)(_oracle_to_matlab(item) for item in value)
    if isinstance(value, (int, np.integer)):
        return float(value)
    return value


def is_matlab_available() -> bool:
    if os.environ.get("DISABLE_MATLAB", "0") == "1":
        return False
    executable = "matlab.exe" if os.name == "nt" else "matlab"
    return shutil.which(executable) is not None


def matlab_call(function_name: str, *args: Any, nargout: int) -> list[Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input.mat"
        output_path = tmpdir_path / "output.mat"
        payload: dict[str, Any] = {
            "function_name": function_name,
            "nargin_value": np.array([[len(args)]], dtype=np.int64),
            "nargout_value": np.array([[nargout]], dtype=np.int64),
        }
        for idx, arg in enumerate(args, start=1):
            payload[f"arg{idx}"] = _to_matlab(_oracle_to_matlab(arg))
        savemat(input_path, payload, do_compression=False)

        matlab_cmd = (
            f"s=load('{_matlab_quote(str(input_path))}');"
            "args=cell(1,s.nargin_value);"
            "for k=1:s.nargin_value,args{k}=s.(sprintf('arg%d',k));end;"
            "[out{1:s.nargout_value}]=feval(s.function_name,args{:});"
            "payload.nout=s.nargout_value;"
            "for k=1:s.nargout_value,payload.(sprintf('out%d',k))=out{k};end;"
            f"save('{_matlab_quote(str(output_path))}','-struct','payload','-v7');"
        )
        env = os.environ.copy()
        env["MATPOWERPY_USE_PYTHON"] = "0"
        subprocess.run(
            ["matlab", "-batch", matlab_cmd],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        data = loadmat(output_path, struct_as_record=False, squeeze_me=True)
        return [_from_matlab(data[f"out{idx}"]) for idx in range(1, nargout + 1)]


def matlab_script(input_payload: dict[str, Any], script: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input.mat"
        output_path = tmpdir_path / "output.mat"
        payload = {key: _to_matlab(_oracle_to_matlab(value)) for key, value in input_payload.items()}
        savemat(input_path, payload, do_compression=False)

        matlab_cmd = f"infile='{_matlab_quote(str(input_path))}';outfile='{_matlab_quote(str(output_path))}';{script}"
        env = os.environ.copy()
        env["MATPOWERPY_USE_PYTHON"] = "0"
        subprocess.run(
            ["matlab", "-batch", matlab_cmd],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        data = loadmat(output_path, struct_as_record=False, squeeze_me=True)
        return {key: _from_matlab(value) for key, value in data.items() if not key.startswith("__")}


def as_dense(value: Any) -> np.ndarray:
    if sparse.issparse(value):
        return value.toarray()
    return np.asarray(value)
