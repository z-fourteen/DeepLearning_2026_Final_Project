"""Bootstrap: fix Windows GBK encoding for all project scripts.

Every entry script under scripts/ does:
    sys.path.insert(0, PROJECT_ROOT)
    from src.data / src.models / src.training import ...

So this __init__ is guaranteed to run first, giving us a single place
to patch stdout/stderr encoding before any print() happens.
"""

import os
import sys

# --- Force UTF-8 I/O on Windows ---
if sys.platform == "win32":
    # 1. Environment variable (affects child processes too)
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # 2. Reconfigure existing streams (Python 3.10+)
    _stdout = sys.stdout
    _stderr = sys.stderr
    if hasattr(_stdout, "reconfigure"):
        try:
            _stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(_stderr, "reconfigure"):
        try:
            _stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # 3. Fallback for older Python: wrap with TextIOWrapper
    if (
        getattr(_stdout, "encoding", None) not in {None, "utf-8", "utf-8-sig"}
        and hasattr(_stdout, "buffer")
    ):
        import io

        try:
            sys.stdout = io.TextIOWrapper(
                _stdout.buffer, encoding="utf-8", errors="replace"
            )
        except Exception:
            pass
    if (
        getattr(_stderr, "encoding", None) not in {None, "utf-8", "utf-8-sig"}
        and hasattr(_stderr, "buffer")
    ):
        import io

        try:
            sys.stderr = io.TextIOWrapper(
                _stderr.buffer, encoding="utf-8", errors="replace"
            )
        except Exception:
            pass

# --- Patch tqdm: force ASCII progress bars on Windows ---
if sys.platform == "win32":
    try:
        import tqdm as _tqdm_mod

        _orig_tqdm_cls = _tqdm_mod.tqdm

        class _AsciiTqdm(_orig_tqdm_cls):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("ascii", True)
                super().__init__(*args, **kwargs)

        _tqdm_mod.tqdm = _AsciiTqdm
    except ImportError:
        pass
