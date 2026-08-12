# Configuration file for the Sphinx documentation builder.
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))


def git_value(*args: str) -> str | None:
    """Return a Git value for local builds, or ``None`` outside a checkout."""
    result = subprocess.run(
        ["git", *args],
        cwd=BASE_DIR,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


docs_branch = os.environ.get("DOCS_BRANCH") or git_value("branch", "--show-current")
docs_commit = os.environ.get("DOCS_COMMIT") or git_value("rev-parse", "HEAD")
docs_commit_url = os.environ.get("DOCS_COMMIT_URL")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pwrs"
copyright = "2026, Liangyu Zhang"
author = "Liangyu Zhang"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "case_comments",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

case_data_root = str(BASE_DIR / "src" / "pwrs" / "data")

templates_path = ["_templates"]
exclude_patterns = ["_build", "build"]

autodoc_default_options = {
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autosummary_generate = False
myst_heading_anchors = 3
# A few legacy NumPy-style docstrings contain MATLAB-era literal formatting
# that Sphinx cannot parse as strict reStructuredText. They remain visible in
# the generated API pages while their parser diagnostics are suppressed.
suppress_warnings = ["docutils", "ref"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = "pwrs documentation"
html_theme_options = {
    "navigation_with_keys": True,
    "footer_end": ["build-info", "theme-version"],
}
html_context = {
    "docs_branch": docs_branch or "unknown",
    "docs_commit": docs_commit or "unknown",
    "docs_commit_short": docs_commit[:7] if docs_commit else "unknown",
    "docs_commit_url": docs_commit_url,
}
