"""Build the static quiz site from the canonical Markdown in ../quizzes/.

Parses each quizzes/*.md (# unit / ## section / ### question / - [ ] option,
where - [x] marks the correct answer) into quizzes.json, then copies the
static template alongside it into ../_site/.

    python quiz_site/build.py

The output dir (_site/) is disposable and gitignored; CI rebuilds it.
"""
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC = REPO / "quizzes"
TEMPLATE = HERE / "template"
OUT = REPO / "_site"

OPTION_RE = re.compile(r"^- \[([ xX])\]\s+(.*)$")


def parse_quiz(md_path: Path) -> dict:
    unit = {"id": md_path.stem, "title": md_path.stem, "sections": []}
    section = None
    question = None

    def flush_question():
        if question is not None:
            if question["answer"] is None:
                raise ValueError(
                    f"{md_path.name}: question has no [x] answer: "
                    f"{question['question']!r}"
                )
            section["questions"].append(question)

    for raw in md_path.read_text().splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            unit["title"] = line[2:].strip()
        elif line.startswith("## "):
            flush_question()
            question = None
            section = {"title": line[3:].strip(), "questions": []}
            unit["sections"].append(section)
        elif line.startswith("### "):
            flush_question()
            question = {"question": line[4:].strip(), "options": [], "answer": None}
        else:
            m = OPTION_RE.match(line)
            if m:
                if question is None:
                    raise ValueError(f"{md_path.name}: option outside a question")
                text = m.group(2).strip()
                question["options"].append(text)
                if m.group(1).lower() == "x":
                    question["answer"] = text
    flush_question()

    # Drop empty sections (e.g. trailing headings with no questions).
    unit["sections"] = [s for s in unit["sections"] if s["questions"]]
    return unit


def main():
    units = []
    for md in sorted(SRC.glob("*.md")):
        unit = parse_quiz(md)
        if unit["sections"]:
            units.append(unit)
    if not units:
        raise SystemExit(f"no quizzes parsed from {SRC}")

    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(TEMPLATE, OUT)
    (OUT / "quizzes.json").write_text(json.dumps(units, ensure_ascii=False, indent=2))

    n_sec = sum(len(u["sections"]) for u in units)
    n_q = sum(len(s["questions"]) for u in units for s in u["sections"])
    print(f"built _site/ : {len(units)} units, {n_sec} sections, {n_q} questions")


if __name__ == "__main__":
    main()
