"""Generate Sphinx API pages for every Python module under ``src/pwrs``."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def module_name(source_root: Path, source: Path) -> str:
    relative = source.relative_to(source_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join((source_root.name, *parts))


def discover_modules(source_root: Path) -> list[str]:
    modules = {
        module_name(source_root, path)
        for path in source_root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    return sorted(modules)


def group_name(module: str) -> str:
    parts = module.split(".")
    return ".".join(parts[:2]) if len(parts) > 1 else module


def module_page(module: str) -> str:
    return f"module-{module}.rst"


def is_package_module(source_root: Path, module: str) -> bool:
    parts = module.split(".")[1:]
    return (source_root.joinpath(*parts, "__init__.py")).is_file()


def write_text_if_changed(path: Path, content: str) -> None:
    """Write *content* without invalidating Sphinx when it is unchanged."""
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def write_module_page(output: Path, module: str, source_root: Path) -> None:
    title = module
    directive = f".. automodule:: {module}\n"
    if not is_package_module(source_root, module):
        directive += "   :members:\n   :show-inheritance:\n   :member-order: bysource\n"
    write_text_if_changed(
        output / module_page(module),
        f"""{title}\n{'=' * len(title)}\n\n{directive}""",
    )


def write_group_page(output: Path, group: str, modules: list[str]) -> None:
    title = f"{group} modules"
    entries = "\n".join(f"   {module_page(module)[:-4]}" for module in modules)
    content = output.parent / "group-content" / f"{group}.inc"
    include = f".. include:: ../group-content/{content.name}\n\n" if content.is_file() else ""
    write_text_if_changed(
        output / f"group-{group}.rst",
        f"""{title}\n{'=' * len(title)}\n\n{include}.. toctree::\n   :maxdepth: 1\n\n{entries}\n""",
    )


def write_module_index(api_root: Path, groups: dict[str, list[str]]) -> None:
    entries = "\n".join(f"   generated/group-{group}" for group in sorted(groups))
    write_text_if_changed(
        api_root / "modules.rst",
        f"""Source modules\n==============\n\n.. toctree::\n   :maxdepth: 2\n\n{entries}\n""",
    )


def generate(source_root: Path, api_root: Path) -> list[str]:
    modules = [
        module
        for module in discover_modules(source_root)
        if not is_package_module(source_root, module)
    ]
    generated = api_root / "generated"
    generated.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[str]] = defaultdict(list)
    for module in modules:
        groups[group_name(module)].append(module)
        write_module_page(generated, module, source_root)
    for group, grouped_modules in groups.items():
        write_group_page(generated, group, grouped_modules)

    expected_pages = {module_page(module) for module in modules}
    expected_pages.update(f"group-{group}.rst" for group in groups)
    for path in generated.glob("*.rst"):
        if path.name not in expected_pages:
            path.unlink()

    write_module_index(api_root, groups)
    return modules


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    docs_root = Path(__file__).resolve().parent
    parser.add_argument(
        "--source",
        type=Path,
        default=docs_root.parent / "src" / "pwrs",
        help="source package root",
    )
    parser.add_argument(
        "--api-root",
        type=Path,
        default=docs_root / "source" / "api",
        help="Sphinx API source directory",
    )
    args = parser.parse_args()
    modules = generate(args.source, args.api_root)
    print(f"generated={len(modules)} modules in {args.api_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
