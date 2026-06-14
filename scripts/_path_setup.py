"""Make project-root imports stable when scripts are executed directly."""
from __future__ import annotations

import sys
from pathlib import Path


def add_project_root_to_path() -> Path:
    """Add the repository root to sys.path and return it."""
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


PROJECT_ROOT = add_project_root_to_path()
