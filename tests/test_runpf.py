from pathlib import Path

import numpy as np
import pytest

from pwrs import mpoption
from pwrs.core.idx_brch import PF, PT, QF, QT
from pwrs.core.idx_bus import VA, VM
from pwrs.core.idx_gen import PG, QG, VG
from pwrs.core.runpf import runpf

from ._case_utils import case_bus_count, case_names, load_case
from ._matlab_cache import load_cached_matlab_case, run_matlab_cache_script
from ._matlab_oracle import _matlab_quote, is_matlab_available

CASE_NAMES = list(case_names())
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "runpf"
ALG_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "runpf_alg"
PF_ALGORITHMS = [
    "NR",
    "NR-SP",
    "NR-SC",
    "NR-SH",
    "NR-IP",
    "NR-IC",
    "NR-IH",
    "FDXB",
    "FDBX",
    "GS",
    "PQSUM",
    "ISUM",
    "YSUM",
]
CASE_BUS_COUNTS = {case_name: case_bus_count(case_name) for case_name in CASE_NAMES}
QUICK_CASE_NAMES = [case_name for case_name in CASE_NAMES if CASE_BUS_COUNTS[case_name] <= 300]
LARGE_CASE_NAMES = [case_name for case_name in CASE_NAMES if CASE_BUS_COUNTS[case_name] > 300]
QUICK_ALG_CASES = [
    ("NR-SP", "case9"),
    ("PQSUM", "case22"),
    ("YSUM", "case22"),
]
LARGE_ALG_CASES: list[tuple[str, str]] = []


def _ensure_matlab_runpf_cache(case_names):
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
        r = runpf(case_names{{k}});
        save(fullfile(outdir, [case_names{{k}} '.mat']), 'r', 'errstr', '-v7');
    catch err
        errstr = getReport(err, 'basic', 'hyperlinks', 'off');
        save(fullfile(outdir, [case_names{{k}} '.mat']), 'errstr', '-v7');
    end
end
"""
    return run_matlab_cache_script(CACHE_DIR, "runpf_cache", script_body)


def _ensure_matlab_runpf_alg_cache(alg_cases):
    missing = [(alg, case_name) for alg, case_name in alg_cases if not (ALG_CACHE_DIR / alg / f"{case_name}.mat").exists()]
    if not missing:
        return ALG_CACHE_DIR

    entries = []
    for alg, case_name in missing:
        entries.append(f"struct('alg', '{_matlab_quote(alg)}', 'case_name', '{_matlab_quote(case_name)}')")
    matlab_entries = ", ".join(entries)
    script_body = f"""
outdir = '{_matlab_quote(str(ALG_CACHE_DIR))}';
jobs = {{{matlab_entries}}};
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
for k = 1:numel(jobs)
    errstr = '';
    alg = jobs{{k}}.alg;
    case_name = jobs{{k}}.case_name;
    algdir = fullfile(outdir, alg);
    if ~exist(algdir, 'dir')
        mkdir(algdir);
    end
    try
        mpopt = mpoption('out.all', 0, 'verbose', 0, 'pf.alg', alg);
        r = runpf(case_name, mpopt);
        save(fullfile(algdir, [case_name '.mat']), 'r', 'errstr', '-v7');
    catch err
        errstr = getReport(err, 'basic', 'hyperlinks', 'off');
        save(fullfile(algdir, [case_name '.mat']), 'errstr', '-v7');
    end
end
"""
    return run_matlab_cache_script(ALG_CACHE_DIR, "runpf_alg_cache", script_body)


def _load_matlab_runpf_case(cache_dir, case_name):
    return load_cached_matlab_case(Path(cache_dir) / f"{case_name}.mat", error_message="MATLAB runpf failed")


def _assert_runpf_primal_close(result, expected):
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


@pytest.fixture(scope="module", autouse=True)
def _matlab_runpf_cache():
    return _ensure_matlab_runpf_cache(CASE_NAMES)


@pytest.fixture(scope="module")
def _matlab_runpf_alg_quick_cache():
    return _ensure_matlab_runpf_alg_cache(QUICK_ALG_CASES)


@pytest.fixture(scope="module")
def _matlab_runpf_alg_large_cache():
    return _ensure_matlab_runpf_alg_cache(LARGE_ALG_CASES)


@pytest.mark.parametrize("case_name", CASE_NAMES, ids=CASE_NAMES)
def test_runpf_matches_matlab(case_name, _matlab_runpf_cache):
    if not is_matlab_available():
        pytest.skip("MATLAB is not available")
        return
    mpc = load_case(case_name)

    result = runpf(mpc, nargout=1)
    expected = _load_matlab_runpf_case(_matlab_runpf_cache, case_name)

    if "__error__" in expected:
        pytest.skip(expected["__error__"])

    assert result["success"] == expected["success"]
    if result["success"]:
        _assert_runpf_primal_close(result, expected)


@pytest.mark.quick
@pytest.mark.parametrize("alg,case_name", QUICK_ALG_CASES, ids=[f"{alg}-{case_name}" for alg, case_name in QUICK_ALG_CASES])
def test_runpf_algorithms_match_matlab_quick(alg, case_name, _matlab_runpf_alg_quick_cache):
    if not is_matlab_available():
        pytest.skip("MATLAB is not available")
        return

    mpc = load_case(case_name)
    mpopt_value = mpoption("out.all", 0, "verbose", 0, "pf.alg", alg)
    result = runpf(mpc, mpopt_value, nargout=1)
    expected = _load_matlab_runpf_case(_matlab_runpf_alg_quick_cache / alg, case_name)

    if "__error__" in expected:
        pytest.skip(expected["__error__"])

    assert result["success"] == expected["success"]
    if result["success"]:
        _assert_runpf_primal_close(result, expected)


@pytest.mark.slow
@pytest.mark.parametrize("alg,case_name", LARGE_ALG_CASES, ids=[f"{alg}-{case_name}" for alg, case_name in LARGE_ALG_CASES])
def test_runpf_algorithms_match_matlab_large(alg, case_name, _matlab_runpf_alg_large_cache):
    if not is_matlab_available():
        pytest.skip("MATLAB is not available")
        return

    mpc = load_case(case_name)
    mpopt_value = mpoption("out.all", 0, "verbose", 0, "pf.alg", alg)
    result = runpf(mpc, mpopt_value, nargout=1)
    expected = _load_matlab_runpf_case(_matlab_runpf_alg_large_cache / alg, case_name)

    if "__error__" in expected:
        pytest.skip(expected["__error__"])

    assert result["success"] == expected["success"]
    if result["success"]:
        _assert_runpf_primal_close(result, expected)
