# -*- coding: utf-8 -*-
"""Minimal test runner used when pytest is unavailable.

It understands exactly two things: module-level ``test_*`` functions, and an
optional ``rng`` argument (a seeded ``numpy.random.Generator``).  Anything the
suite needs beyond that (fixtures, parametrisation) it does not support -- use
pytest for the real thing.

    python src/percolation/tests/run_tests.py            # fast tests only
    python src/percolation/tests/run_tests.py --all      # include @slow
"""

import argparse
import importlib.util
import inspect
import pathlib
import sys
import time
import traceback

import numpy as np


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also run @pytest.mark.slow tests")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parents[1]))  # .../src

    passed, failed, skipped = 0, [], 0
    for path in sorted(here.glob("test_*.py")):
        mod = load(path)
        print(f"\n{path.name}")
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            slow = bool(getattr(fn, "__slow__", False)) or \
                any(getattr(m, "name", "") == "slow" for m in getattr(fn, "pytestmark", []))
            if slow and not args.all:
                skipped += 1
                print(f"  skip  {name}  (slow)")
                continue
            kwargs = {}
            if "rng" in inspect.signature(fn).parameters:
                kwargs["rng"] = np.random.default_rng(20261104)
            t0 = time.perf_counter()
            try:
                fn(**kwargs)
                passed += 1
                print(f"  ok    {name}  ({time.perf_counter() - t0:.2f}s)")
            except Exception as exc:  # noqa: BLE001
                failed.append((path.name, name, exc))
                print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
                if args.verbose:
                    traceback.print_exc()

    print(f"\n{passed} passed, {len(failed)} failed, {skipped} skipped")
    for f in failed:
        print(f"  FAILED {f[0]}::{f[1]} -> {type(f[2]).__name__}: {f[2]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
