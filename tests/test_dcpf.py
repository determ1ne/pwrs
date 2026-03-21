from pathlib import Path

import numpy as np
import pytest

from pwrs.core.idx_brch import PF, PT, QF, QT
from pwrs.core.idx_bus import VA, VM
from pwrs.core.idx_gen import PG, QG, VG
from pwrs.core.rundcpf import rundcpf

from ._case_utils import case_bus_count, case_names, load_case
from ._matlab_cache import load_cached_matlab_case, run_matlab_cache_script
from ._matlab_oracle import _matlab_quote, is_matlab_available

ALL_CASE_NAMES = list(case_names())
CASE_NAMES = [case_name for case_name in ALL_CASE_NAMES if case_bus_count(case_name) <= 300]
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "rundcpf"


def _ensure_matlab_rundcpf_cache(case_names):
    missing_case_names = [case_name for case_name in case_names if not (CACHE_DIR / f"{case_name}.mat").exists()]
    if not missing_case_names:
        return CACHE_DIR

    matlab_cases = ", ".join(f"'{_matlab_quote(case_name)}'" for case_name in missing_case_names)
    script_body = f"""
outdir = '{_matlab_quote(str(CACHE_DIR))}';
case_names = {{{matlab_cases}}};
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
for k = 1:numel(case_names)
    errstr = '';
    try
        r = rundcpf(case_names{{k}});
        save(fullfile(outdir, [case_names{{k}} '.mat']), 'r', 'errstr', '-v7');
    catch err
        errstr = getReport(err, 'basic', 'hyperlinks', 'off');
        save(fullfile(outdir, [case_names{{k}} '.mat']), 'errstr', '-v7');
    end
end
"""
    return run_matlab_cache_script(CACHE_DIR, "rundcpf_cache", script_body)


def _load_matlab_rundcpf_case(cache_dir, case_name):
    return load_cached_matlab_case(Path(cache_dir) / f"{case_name}.mat", error_message="MATLAB rundcpf failed")


def _assert_rundcpf_close(result, expected):
    np.testing.assert_allclose(result["bus"][:, [VM - 1, VA - 1]], expected["bus"][:, [VM - 1, VA - 1]], atol=1e-8, rtol=0)
    np.testing.assert_allclose(result["gen"][:, [PG - 1, QG - 1, VG - 1]], expected["gen"][:, [PG - 1, QG - 1, VG - 1]], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        result["branch"][:, [PF - 1, QF - 1, PT - 1, QT - 1]],
        expected["branch"][:, [PF - 1, QF - 1, PT - 1, QT - 1]],
        atol=1e-8,
        rtol=0,
    )


@pytest.fixture(scope="module")
def _matlab_rundcpf_cache():
    if not is_matlab_available():
        pytest.skip("MATLAB is not available")
    return _ensure_matlab_rundcpf_cache(CASE_NAMES)


@pytest.mark.parametrize("case_name", CASE_NAMES, ids=CASE_NAMES)
def test_rundcpf_matches_matlab(case_name, _matlab_rundcpf_cache):
    mpc = load_case(case_name)
    result = rundcpf(mpc, nargout=1)
    expected = _load_matlab_rundcpf_case(_matlab_rundcpf_cache, case_name)

    if "__error__" in expected:
        pytest.skip(expected["__error__"])

    assert result["success"] == expected["success"]
    if result["success"]:
        _assert_rundcpf_close(result, expected)
