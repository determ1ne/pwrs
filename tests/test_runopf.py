from pathlib import Path

import numpy as np
import pytest

from pwrs import mpoption, runopf
from pwrs.core.idx_brch import PF, PT, QF, QT
from pwrs.core.idx_bus import VA, VM
from pwrs.core.idx_gen import PG, QG, VG

from ._case_utils import case_bus_count, case_names, load_case
from ._matlab_cache import load_cached_matlab_case, run_matlab_cache_script
from ._matlab_oracle import _matlab_quote, is_matlab_available

CASE_NAMES = list(case_names())
CASE_BUS_COUNTS = {case_name: case_bus_count(case_name) for case_name in CASE_NAMES}
QUICK_CASE_NAMES = [case_name for case_name in CASE_NAMES if CASE_BUS_COUNTS[case_name] <= 300]
LARGE_CASE_NAMES = [case_name for case_name in CASE_NAMES if CASE_BUS_COUNTS[case_name] > 300]
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "runopf"


def _ensure_matlab_runopf_cache(case_names):
    missing_case_names = [case_name for case_name in case_names if not (CACHE_DIR / f"{case_name}.mat").exists()]
    if not missing_case_names:
        return CACHE_DIR

    matlab_cases = ", ".join(f"'{_matlab_quote(case_name)}'" for case_name in missing_case_names)
    script_body = f"""
outdir = '{_matlab_quote(str(CACHE_DIR))}';
case_names = {{{matlab_cases}}};
mpopt = mpoption('out.all', 0, 'verbose', 0, 'opf.ac.solver', 'MIPS');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
for k = 1:numel(case_names)
    errstr = '';
    try
        r = runopf(case_names{{k}}, mpopt);
        save(fullfile(outdir, [case_names{{k}} '.mat']), 'r', 'errstr', '-v7');
    catch err
        errstr = getReport(err, 'basic', 'hyperlinks', 'off');
        save(fullfile(outdir, [case_names{{k}} '.mat']), 'errstr', '-v7');
    end
end
"""
    return run_matlab_cache_script(CACHE_DIR, "runopf_cache", script_body)


def _load_matlab_runopf_case(cache_dir, case_name):
    return load_cached_matlab_case(Path(cache_dir) / f"{case_name}.mat", error_message="MATLAB runopf failed")


def _assert_runopf_primal_close(result, expected):
    np.testing.assert_allclose(result["f"], expected["f"], atol=1e-3, rtol=1e-5)
    np.testing.assert_allclose(
        result["bus"][:, [VM - 1, VA - 1]], expected["bus"][:, [VM - 1, VA - 1]], atol=1e-3, rtol=0
    )
    np.testing.assert_allclose(
        result["gen"][:, [PG - 1, QG - 1, VG - 1]], expected["gen"][:, [PG - 1, QG - 1, VG - 1]], atol=5e-2, rtol=0
    )
    np.testing.assert_allclose(
        result["branch"][:, [PF - 1, QF - 1, PT - 1, QT - 1]],
        expected["branch"][:, [PF - 1, QF - 1, PT - 1, QT - 1]],
        atol=5e-2,
        rtol=0,
    )


def _run_case(case_name, cache_dir):
    mpc = load_case(case_name)
    mpopt_value = mpoption("out.all", 0, "verbose", 0, "opf.ac.solver", "MIPS")
    try:
        result = runopf(mpc, mpopt_value, nargout=1)
    except Exception:
        result = {"success": 0}
    expected = _load_matlab_runopf_case(cache_dir, case_name)
    if "__error__" in expected:
        pytest.skip(expected["__error__"])
    assert result["success"] == expected["success"]
    if result["success"]:
        _assert_runopf_primal_close(result, expected)


@pytest.fixture(scope="module")
def _matlab_runopf_quick_cache():
    if not is_matlab_available():
        pytest.skip("MATLAB is not available")
    return _ensure_matlab_runopf_cache(QUICK_CASE_NAMES)


@pytest.fixture(scope="module")
def _matlab_runopf_large_cache():
    if not is_matlab_available():
        pytest.skip("MATLAB is not available")
    return _ensure_matlab_runopf_cache(LARGE_CASE_NAMES)


@pytest.mark.quick
@pytest.mark.parametrize("case_name", QUICK_CASE_NAMES, ids=QUICK_CASE_NAMES)
def test_runopf_matches_matlab_quick(case_name, _matlab_runopf_quick_cache):
    _run_case(case_name, _matlab_runopf_quick_cache)


@pytest.mark.slow
@pytest.mark.parametrize("case_name", LARGE_CASE_NAMES, ids=LARGE_CASE_NAMES)
def test_runopf_matches_matlab_large(case_name, _matlab_runopf_large_cache):
    _run_case(case_name, _matlab_runopf_large_cache)
