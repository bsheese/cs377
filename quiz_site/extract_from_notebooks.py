"""One-time migration: extract `all_quizzes` dicts from the practice-quiz
notebooks and emit human-readable Markdown into ../quizzes/.

After this runs, the Markdown files are the canonical source. The notebooks
are no longer used to author quizzes. Re-running is safe (it overwrites).

Usage:
    python quiz_site/extract_from_notebooks.py
"""
import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "quizzes"

# Friendly unit titles keyed by the unit folder name. Fallback prettifies the
# folder name if a key is missing.
TITLES = {
    "16_ml_intro": "16 · Intro to Machine Learning",
    "17_0_Preliminaries": "17.0 · Regression Preliminaries",
    "17_1_SLR": "17.1 · Simple Linear Regression",
    "17_2_MLR": "17.2 · Multiple Linear Regression",
    "17_3_Interactions": "17.3 · Interaction Terms",
    "18_1_Classification_Basics": "18.1 · Classification Basics",
    "18_2_LogisticRegression": "18.2 · Logistic Regression",
    "18_6_Ensemble": "18.6 · Ensemble Methods",
}


def _match_brace(src: str, open_idx: int) -> int:
    """Index just past the `}` matching the `{` at src[open_idx], skipping
    braces that appear inside string literals."""
    depth = 0
    i = open_idx
    quote = None
    while i < len(src):
        c = src[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces while slicing all_quizzes literal")


def find_all_quizzes(nb_path: Path) -> dict:
    """Pull the literal `all_quizzes = {...}` dict out of a notebook safely.

    Some notebooks bundle the dict in the same cell as helper code that does
    not parse as a standalone module, so we slice out just the dict literal
    (via brace matching) and eval that snippet rather than the whole cell.
    """
    nb = json.loads(nb_path.read_text())
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        m = re.search(r"^all_quizzes\s*=\s*", src, re.MULTILINE)
        if not m:
            continue
        open_idx = src.index("{", m.end())
        literal = src[open_idx:_match_brace(src, open_idx)]
        return ast.literal_eval(literal)
    raise ValueError(f"no all_quizzes dict found in {nb_path}")


def slugify(unit_dir: str) -> str:
    return unit_dir.lower()


def prettify(unit_dir: str) -> str:
    return unit_dir.replace("_", " ")


def to_markdown(title: str, quizzes: dict) -> str:
    lines = [f"# {title}", ""]
    for section, questions in quizzes.items():
        lines.append(f"## {section}")
        lines.append("")
        for q in questions:
            lines.append(f"### {q['question']}")
            answer = q["answer"]
            if answer not in q["options"]:
                raise ValueError(
                    f"answer not among options in section {section!r}: {answer!r}"
                )
            for opt in q["options"]:
                mark = "x" if opt == answer else " "
                lines.append(f"- [{mark}] {opt}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    OUT.mkdir(exist_ok=True)
    notebooks = sorted(REPO.glob("**/*practice_quiz*.ipynb"))
    if not notebooks:
        raise SystemExit("no practice_quiz notebooks found")
    for nb in notebooks:
        unit_dir = nb.parent.name
        slug = slugify(unit_dir)
        title = TITLES.get(unit_dir, prettify(unit_dir))
        quizzes = find_all_quizzes(nb)
        md = to_markdown(title, quizzes)
        (OUT / f"{slug}.md").write_text(md)
        n = sum(len(v) for v in quizzes.values())
        print(f"{slug}.md  ({len(quizzes)} sections, {n} questions)")


if __name__ == "__main__":
    main()
