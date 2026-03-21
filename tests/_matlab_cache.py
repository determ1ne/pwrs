import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from ._matlab_oracle import REPO_ROOT, _from_matlab, _matlab_quote


def run_matlab_cache_script(cache_dir: Path, script_stem: str, script_body: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        script_path = tmpdir_path / f"{script_stem}.m"
        script_path.write_text(script_body, encoding="ascii")
        script_ref = str(script_path).replace("'", "''")
        subprocess.run(
            ["matlab", "-batch", f"run('{script_ref}')"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    return cache_dir


def load_cached_matlab_case(cache_path: Path, *, error_message: str) -> dict[str, Any]:
    data = loadmat(cache_path, struct_as_record=False, squeeze_me=True)
    if "r" not in data:
        return {"__error__": str(data.get("errstr", error_message))}
    result = _from_matlab(data["r"])
    for key in ("bus", "gen", "branch", "gencost"):
        if key in result:
            value = np.asarray(result[key])
            if value.ndim == 1:
                result[key] = value.reshape(1, -1)
    return result
