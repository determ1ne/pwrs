"""Sphinx directive for rendering comments from bundled case JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeGuard

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx

_DATASETS = {
    "matpower": ("MATPOWER", "case"),
    "pglibopf": ("PGLib-OPF", "pglib_opf_case"),
}


def _is_string_list(value: object) -> TypeGuard[list[str]]:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _load_comments(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    key = '"comments"'
    key_start = text.find(key)
    if key_start < 0:
        raise ValueError(f"{path} does not contain a comments field")

    value_start = text.find(":", key_start + len(key))
    if value_start < 0:
        raise ValueError(f"{path} has an invalid comments field")

    value_start += 1
    while value_start < len(text) and text[value_start].isspace():
        value_start += 1
    comments: object = json.JSONDecoder().raw_decode(text, value_start)[0]
    if not _is_string_list(comments):
        raise ValueError(f"{path}: comments must be a list of strings")
    return comments


def _pglib_intro(comments: list[str]) -> tuple[list[str], int]:
    """Keep descriptive PGLib comments, excluding generated data/log sections."""
    for index, line in enumerate(comments):
        if line.strip().lower().startswith("% bus data"):
            return comments[:index], len(comments) - index
    return comments, 0


class CaseCommentsDirective(Directive):
    """Render one subsection per case from its JSON ``comments`` field."""

    required_arguments = 1
    has_content = False

    def run(self) -> list[nodes.Node]:
        dataset = self.arguments[0].strip().lower()
        if dataset not in _DATASETS:
            names = ", ".join(sorted(_DATASETS))
            raise self.error(f"unknown case data set {dataset!r}; expected one of: {names}")

        title, prefix = _DATASETS[dataset]
        data_root = Path(self.state.document.settings.env.config.case_data_root)
        dataset_root = data_root / dataset
        paths = sorted(dataset_root.glob(f"{prefix}*.json"))
        if not paths:
            raise self.error(f"no {title} case JSON files found under {dataset_root}")

        result: list[nodes.Node] = []
        for path in paths:
            self.state.document.settings.env.note_dependency(str(path))
            comments = _load_comments(path)
            omitted = 0
            if dataset == "pglibopf":
                comments, omitted = _pglib_intro(comments)

            case_name = path.stem
            section = nodes.section(ids=[nodes.make_id(f"{dataset}-{case_name}")])
            section += nodes.title(text=case_name)

            import_path = f"pwrs.data.{dataset}.{case_name}()"
            python_path = nodes.paragraph()
            python_path += nodes.Text("Python: ")
            python_path += nodes.literal(text=import_path)
            section += python_path
            comment_block = nodes.literal_block(text="\n".join(comments).rstrip())
            comment_block["language"] = "text"
            section += comment_block
            if omitted:
                section += nodes.paragraph(
                    text=f"{omitted:,} generated data and conversion-log comment lines are omitted."
                )
            result.append(section)
        return result


def setup(app: Sphinx) -> dict[str, str | bool]:
    app.add_config_value("case_data_root", "", "env")
    app.add_directive("case-comments", CaseCommentsDirective)
    return {
        "version": "1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
