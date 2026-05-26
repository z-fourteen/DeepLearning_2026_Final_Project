from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.ingest.agent import ensure_directories, load_yaml  # noqa: E402


def main() -> None:
    project_root = PROJECT_ROOT
    config = load_yaml(project_root / "configs" / "data.yaml")
    ensure_directories(config, project_root)
    print("Pipeline directories initialized.")


if __name__ == "__main__":
    main()
