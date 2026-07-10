#!/usr/bin/env python3
"""Execute every teaching notebook and report failures.

Practice quizzes are skipped (they ship unexecuted by design), as are
archived notebooks in old/ directories. Notebooks are executed with
--inplace OFF: output goes to a temp copy, so committed outputs are
never touched.

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
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent

EXCLUDE_DIR_PARTS = {"venv", ".ipynb_checkpoints", "old", "old_material", "_site", ".ignore"}


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
                proc = subprocess.run(
                    [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                     "--execute", str(ROOT / nb), "--output-dir", tmp,
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
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
