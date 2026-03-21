from pathlib import Path

import numpy as np
import pytest

from pwrs import mpoption
from pwrs.core.runcpf import runcpf

from ._case_utils import case_bus_count, case_names, load_case
from ._matlab_cache import load_cached_matlab_case, run_matlab_cache_script
from ._matlab_oracle import _matlab_quote, is_matlab_available

CASE_NAMES = list(case_names())
CASE_BUS_COUNTS = {case_name: case_bus_count(case_name) for case_name in CASE_NAMES}
QUICK_CASE_NAMES = [case_name for case_name in CASE_NAMES if CASE_BUS_COUNTS[case_name] <= 39]
CPF_QUICK_CASES = [(case_name, case_name) for case_name in QUICK_CASE_NAMES]
CPF_SPECIAL_CASES = [("case9", "case9target")]
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "runcpf"


def _cache_key(base_case_name, target_case_name):
    return f"{base_case_name}__{target_case_name}"


def _ensure_matlab_runcpf_cache(case_pairs):
    missing_pairs = [
        (base_case_name, target_case_name)
        for base_case_name, target_case_name in case_pairs
        if not (CACHE_DIR / f"{_cache_key(base_case_name, target_case_name)}.mat").exists()
    ]
    if not missing_pairs:
        return CACHE_DIR

    entries = ", ".join(
        f"struct('base_case_name', '{_matlab_quote(base_case_name)}', 'target_case_name', '{_matlab_quote(target_case_name)}')"
        for base_case_name, target_case_name in missing_pairs
    )
    script_body = f"""
outdir = '{_matlab_quote(str(CACHE_DIR))}';
jobs = {{{entries}}};
mpopt = mpoption('out.all', 0, 'verbose', 0);
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
for k = 1:numel(jobs)
    errstr = '';
    base_case_name = jobs{{k}}.base_case_name;
    target_case_name = jobs{{k}}.target_case_name;
    key = [base_case_name '__' target_case_name];
    try
        r = runcpf(base_case_name, target_case_name, mpopt);
        save(fullfile(outdir, [key '.mat']), 'r', 'errstr', '-v7');
    catch err
        errstr = getReport(err, 'basic', 'hyperlinks', 'off');
        save(fullfile(outdir, [key '.mat']), 'errstr', '-v7');
    end
end
"""
    return run_matlab_cache_script(CACHE_DIR, "runcpf_cache", script_body)


def _load_matlab_runcpf_case(cache_dir, base_case_name, target_case_name):
    return load_cached_matlab_case(
        Path(cache_dir) / f"{_cache_key(base_case_name, target_case_name)}.mat",
        error_message="MATLAB runcpf failed",
    )


@pytest.fixture(scope="module")
def _matlab_runcpf_quick_cache():
    if not is_matlab_available():
        pytest.skip("MATLAB is not available")
    return _ensure_matlab_runcpf_cache(CPF_QUICK_CASES)


@pytest.fixture(scope="module")
def _matlab_runcpf_special_cache():
    if not is_matlab_available():
        pytest.skip("MATLAB is not available")
    return _ensure_matlab_runcpf_cache(CPF_SPECIAL_CASES)


@pytest.mark.slow
@pytest.mark.parametrize("case_name", QUICK_CASE_NAMES, ids=QUICK_CASE_NAMES)
def test_runcpf_quick_identical_base_target(case_name, _matlab_runcpf_quick_cache):
    mpc = load_case(case_name)
    result = runcpf(mpc, mpc, mpoption("out.all", 0, "verbose", 0), nargout=1)
    expected = _load_matlab_runcpf_case(_matlab_runcpf_quick_cache, case_name, case_name)

    if "__error__" in expected:
        pytest.skip(expected["__error__"])

    assert result["success"] == expected["success"]
    assert "cpf" in result
    assert result["cpf"]["done_msg"] == expected["cpf"]["done_msg"]


def test_runcpf_case9_to_case9target_smoke(_matlab_runcpf_special_cache):
    result = runcpf(load_case("case9"), load_case("case9target"), mpoption("out.all", 0, "verbose", 0), nargout=1)
    expected = _load_matlab_runcpf_case(_matlab_runcpf_special_cache, "case9", "case9target")

    if "__error__" in expected:
        pytest.skip(expected["__error__"])

    assert result["success"] == expected["success"]
    assert "cpf" in result
    assert result["cpf"]["done_msg"] == expected["cpf"]["done_msg"]
    lam = np.asarray(result["cpf"]["lam"])
    expected_lam = np.asarray(expected["cpf"]["lam"])
    assert lam.size == expected_lam.size
    np.testing.assert_allclose(np.max(lam), np.max(expected_lam), atol=1e-8, rtol=0)
