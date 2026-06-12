"""Tail-integrity sweep for v2.

Walks every .py file in the production tree, checks:
  * AST parses
  * No NUL bytes anywhere (the Edit-tool truncation pattern leaves
    trailing nulls that grep flags as "binary file matches")
  * Ends with a newline

Vendor / third-party trees are skipped (swap_models/simswap/, LatentSync/).

Run after any large-scale edit pass or before pushing a release.
Exit code 0 = clean, 1 = problems found (logged to stderr).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = ("__pycache__", "_snapshots", "swap_models",
              "LatentSync", "models", "recordings",
              ".pytest_cache", ".git")


def main() -> int:
    null_byte = []
    ast_error = []
    no_newline = []
    n_scanned = 0

    for p in PROJECT_ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        n_scanned += 1
        try:
            raw = p.read_bytes()
        except OSError as exc:
            ast_error.append((str(p.relative_to(PROJECT_ROOT)),
                              "read error: " + str(exc)))
            continue
        nbytes = raw.count(b"\x00")
        if nbytes:
            null_byte.append((str(p.relative_to(PROJECT_ROOT)), nbytes))
        try:
            ast.parse(raw.decode("utf-8", errors="replace"),
                      filename=str(p))
        except SyntaxError as exc:
            ast_error.append((str(p.relative_to(PROJECT_ROOT)),
                              "line " + str(exc.lineno) + ": " + str(exc.msg)))
        if len(raw) > 0 and not raw.endswith(b"\n"):
            no_newline.append((str(p.relative_to(PROJECT_ROOT)),
                               len(raw)))

    if null_byte or ast_error:
        print("TAIL-INTEGRITY FAIL", file=sys.stderr)
        if null_byte:
            print("  null bytes in:", file=sys.stderr)
            for rel, n in null_byte:
                print("    " + rel + " (" + str(n) + " null bytes)",
                      file=sys.stderr)
        if ast_error:
            print("  AST errors in:", file=sys.stderr)
            for rel, msg in ast_error:
                print("    " + rel + ": " + msg, file=sys.stderr)
        return 1

    print("OK -- " + str(n_scanned) + " .py files scanned, all parse, "
          "no NULs.")
    if no_newline:
        print("WARN -- " + str(len(no_newline))
              + " files missing trailing newline:")
        for rel, sz in no_newline:
            print("    " + rel + " (" + str(sz) + " bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
