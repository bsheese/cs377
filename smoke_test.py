#!/usr/bin/env python3
"""Execute every teaching notebook and report failures.

Practice quizzes are skipped (they ship unexecuted by design), as are
archived notebooks in old/ directories. Each notebook is copied into its own
temp subdirectory and executed there — nbconvert sets the kernel's working
directory to wherever the notebook file lives, so this keeps committed
outputs untouched AND stops side-effect writes (saved models, cached CSVs)
from leaking into the source tree. No included notebook reads a local
relative-path file, so the copy is safe.

Usage:
    python smoke_test.py                  # run everything (slow — some
                                          #   notebooks take ~30 minutes)
    python smoke_test.py 17 18_5          # only paths containing a pattern
    python smoke_test.py --list           # show what would run, don't run
    python smoke_test.py --timeout 900    # per-notebook timeout in seconds

Exit status is non-zero if any notebook fails or times out.
"""

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent
VENV_DIR = ROOT / "venv"

EXCLUDE_DIR_PARTS = {"venv", ".ipynb_checkpoints", "old", "old_material", "_site", ".ignore"}


def venv_warning():
    """None if running from ./venv; otherwise a banner to print.

    Jupyter picks a kernel via sys.prefix, which is set by *which python ran
    this script* — not by whether `source venv/bin/activate` was typed.
    (Checking sys.executable directly doesn't work: venv/bin/python3 is a
    symlink to the system interpreter, so resolving it collapses both paths
    to the same file. sys.prefix is what venvs actually redirect via
    pyvenv.cfg, so it's the correct signal.) If this process isn't running
    with ./venv's prefix, notebooks execute against whatever kernel is
    registered elsewhere on the machine, which can be missing or have
    outdated packages (e.g. an old jinja2 breaking pandas .style). Failures
    from that mismatch look exactly like real notebook bugs, so flag it
    loudly rather than let it masquerade as one.
    """
    try:
        running = pathlib.Path(sys.prefix).resolve()
        expected = VENV_DIR.resolve()
    except OSError:
        return None
    if running == expected:
        return None
    return (
        "\n"
        "############################################################\n"
        "#  WARNING: not running from this project's venv.\n"
        f"#  Expected prefix: {expected}\n"
        f"#  Actual prefix:   {running}\n"
        "#  Run `source venv/bin/activate` first — otherwise notebooks\n"
        "#  execute against whatever Jupyter kernel is registered\n"
        "#  elsewhere, and failures below may be environment artifacts,\n"
        "#  not real bugs.\n"
        "############################################################\n"
    )


def collect(patterns):
    nbs = []
    for p in sorted(ROOT.rglob("*.ipynb")):
        rel = p.relative_to(ROOT)
        if EXCLUDE_DIR_PARTS & set(rel.parts):
            continue
        if "practice_quiz" in p.name:
            continue
        if patterns and not any(pat in str(rel) for pat in patterns):
            continue
        nbs.append(rel)
    return nbs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="*", help="only run notebooks whose path contains one of these substrings")
    ap.add_argument("--list", action="store_true", help="list notebooks that would run, then exit")
    ap.add_argument("--timeout", type=int, default=2400, help="per-notebook timeout in seconds (default 2400)")
    args = ap.parse_args()

    warning = venv_warning()
    if warning:
        print(warning)

    nbs = collect(args.patterns)
    if not nbs:
        print("No notebooks matched.")
        return 1
    if args.list:
        for nb in nbs:
            print(nb)
        print(f"\n{len(nbs)} notebooks")
        return 0

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, nb in enumerate(nbs, 1):
            print(f"[{i}/{len(nbs)}] {nb} ... ", end="", flush=True)
            start = time.time()
            try:
                work_dir = pathlib.Path(tmp) / str(nb).replace("/", "__")
                work_dir.mkdir()
                nb_copy = work_dir / nb.name
                shutil.copy(ROOT / nb, nb_copy)
                proc = subprocess.run(
                    [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                     "--execute", str(nb_copy), "--output-dir", str(work_dir),
                     "--ExecutePreprocessor.timeout", str(args.timeout)],
                    capture_output=True, text=True, timeout=args.timeout + 60,
                )
                ok = proc.returncode == 0
                err = proc.stderr
            except subprocess.TimeoutExpired:
                ok, err = False, f"hard timeout after {args.timeout}s"
            elapsed = time.time() - start
            if ok:
                print(f"ok ({elapsed:.0f}s)")
            else:
                print(f"FAIL ({elapsed:.0f}s)")
                # keep the last relevant traceback lines for the summary
                tail = "\n".join(line for line in err.splitlines() if line.strip())[-2000:]
                failures.append((nb, tail))

    print(f"\n{len(nbs) - len(failures)}/{len(nbs)} notebooks passed.")
    for nb, tail in failures:
        print(f"\n=== FAILED: {nb} ===\n{tail}")
    if failures and warning:
        print(warning)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
