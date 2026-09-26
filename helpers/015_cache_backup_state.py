from __future__ import annotations

from pathlib import Path


BACKUP_DIRTY_MARKER = ".github-backup-dirty"


def mark_cache_backup_dirty(root_path: str) -> None:
    root = Path(root_path)
    root.mkdir(parents=True, exist_ok=True)
    (root / BACKUP_DIRTY_MARKER).write_text("1\n", encoding="utf-8")


def is_cache_backup_dirty(root_path: str) -> bool:
    return (Path(root_path) / BACKUP_DIRTY_MARKER).is_file()


def clear_cache_backup_dirty(root_path: str) -> None:
    marker = Path(root_path) / BACKUP_DIRTY_MARKER
    if marker.exists():
        marker.unlink()
