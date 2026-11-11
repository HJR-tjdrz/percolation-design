# -*- coding: utf-8 -*-
"""Tiny stand-in for the parts of pytest the test modules use.

Only ``pytest.mark.slow`` and ``pytest.fixture`` are needed.  Real pytest is
still preferred when it is installed; this module is imported as a fallback.
"""

import types

mark = types.SimpleNamespace()


def _slow(fn):
    fn.__slow__ = True
    return fn


mark.slow = _slow


def fixture(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def deco(fn):
        return fn
    return deco
