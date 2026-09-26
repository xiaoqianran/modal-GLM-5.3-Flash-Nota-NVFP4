from __future__ import annotations

import os
import hashlib
import json
import shutil
import tempfile
import re
import threading
from pathlib import Path


def startup_plan_fingerprints(persistent_dir: str) -> dict[str, str]:
    """Inventory valid v1 plans by content, excluding temp/corrupt files."""
    records = {}
    for path in Path(persistent_dir).glob("startup_plan_*.json"):
        try:
            plan = json.loads(path.read_text(encoding="utf-8"))
            fingerprint = plan.get("fingerprint")
            if (
                plan.get("schema") != 1
                or not isinstance(fingerprint, str)
                or re.fullmatch(r"[0-9a-f]{16}", fingerprint) is None
                or path.name != f"startup_plan_{fingerprint}.json"
                or type(plan.get("kv_cache_memory_bytes")) is not int
                or plan["kv_cache_memory_bytes"] <= 0
                or type(plan.get("free_memory_baseline")) is not int
                or plan["free_memory_baseline"] < 0
            ):
                continue
            payload = json.dumps(plan, sort_keys=True, separators=(",", ":"))
            records[path.name] = hashlib.sha256(payload.encode()).hexdigest()
        except (OSError, ValueError, AttributeError):
            continue
    return records


def changed_startup_plans(before: dict[str, str], persistent_dir: str) -> list[str]:
    """A different valid fingerprint or changed payload both require a commit."""
    after = startup_plan_fingerprints(persistent_dir)
    return sorted(name for name, digest in after.items() if before.get(name) != digest)


def step_018_prepare_startup_plan_cache(
    *,
    local_cache_root: str,
    persistent_dir: str,
) -> dict[str, str]:
    """只把 vLLM startup_plan 子目录持久化，避免把整个 VLLM_CACHE_ROOT 放到 9P。"""
    local_root = Path(local_cache_root)
    persistent_root = Path(persistent_dir)
    local_root.mkdir(parents=True, exist_ok=True)
    persistent_root.mkdir(parents=True, exist_ok=True)

    before = startup_plan_fingerprints(persistent_dir)
    startup_plan_link = local_root / "startup_plan"

    if startup_plan_link.is_symlink():
        current_target = startup_plan_link.resolve(strict=False)
        if current_target != persistent_root.resolve():
            startup_plan_link.unlink()
    elif startup_plan_link.exists():
        if startup_plan_link.is_dir():
            shutil.rmtree(startup_plan_link)
        else:
            startup_plan_link.unlink()

    if not startup_plan_link.exists():
        os.symlink(persistent_root, startup_plan_link, target_is_directory=True)

    print(
        "[018_STARTUP_PLAN_CACHE] "
        f"local={startup_plan_link} persistent={persistent_root} "
        f"valid_files={len(before)} actual_hit=unknown",
        flush=True,
    )
    return before


class StartupPlanObserver:
    """Only vLLM's apply log confirms that its fingerprint/free-memory gate passed."""

    def __init__(self) -> None:
        self.applied_fingerprints: set[str] = set()
        self.saved_paths: set[str] = set()
        self.rejected = False
        self._lock = threading.Lock()

    def result(self) -> tuple[str, list[str]]:
        with self._lock:
            applied = sorted(self.applied_fingerprints)
            # No apply log yet is not proof of a miss (log reader may be behind).
            hit = "true" if applied else "false" if self.rejected else "unknown"
            return hit, applied

    def observe(self, line: str) -> None:
        match = re.search(r"Applying persisted startup plan \(fingerprint ([0-9a-f]{16})\)", line)
        if match:
            with self._lock:
                self.applied_fingerprints.add(match.group(1))
            print(f"[018_STARTUP_PLAN_APPLIED] actual_hit=true fingerprint={match.group(1)}", flush=True)
        match = re.search(r"Saved startup plan to (.+?)(?:\x1b\[.*)?$", line.strip())
        if match:
            with self._lock:
                self.saved_paths.add(match.group(1))
            print(f"[018_STARTUP_PLAN_SAVED] path={match.group(1)} volume_committed=false", flush=True)
        if "Startup plan not applied:" in line or "Ignoring unreadable startup plan" in line:
            with self._lock:
                self.rejected = True
            print("[018_STARTUP_PLAN_REJECTED] actual_hit=false", flush=True)


def _modelinfo_files(root: Path) -> dict[str, bytes]:
    records = {}
    for path in root.glob("*.json"):
        try:
            data = path.read_bytes()
            value = json.loads(data)
            if isinstance(value.get("hash"), str) and isinstance(value.get("modelinfo"), dict):
                records[path.name] = data
        except (OSError, ValueError, AttributeError):
            continue
    return records


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare_modelinfo_cache(*, local_cache_root: str, persistent_root: str,
                            capabilities_path: str) -> str:
    capabilities = json.loads(Path(capabilities_path).read_text(encoding="utf-8"))
    if capabilities.get("modelinfo_cache_dir") != "modelinfos":
        raise RuntimeError("Image model registry cache layout was not verified")
    namespace = capabilities["cache_namespace"]
    if re.fullmatch(r"[0-9a-f]{64}", namespace) is None:
        raise RuntimeError("Invalid image modelinfo namespace")
    seed = Path(persistent_root) / namespace
    local = Path(local_cache_root) / "modelinfos"
    local.mkdir(parents=True, exist_ok=True)
    records = _modelinfo_files(seed)
    for name, data in records.items():
        _atomic_write(local / name, data)
    print(f"[018_MODELINFO_STAGE] files={len(records)} namespace={namespace} "
          f"local={local} actual_hit=unknown", flush=True)
    return str(seed)


def sync_modelinfo_cache(*, local_cache_root: str, persistent_dir: str) -> bool:
    """Publish only small model-info JSONs; native source-hash validation stays intact."""
    before = _modelinfo_files(Path(persistent_dir))
    records = _modelinfo_files(Path(local_cache_root) / "modelinfos")
    changed = [name for name, data in records.items() if before.get(name) != data]
    for name in changed:
        _atomic_write(Path(persistent_dir) / name, records[name])
    print(f"[018_MODELINFO_SYNC] changed_files={len(changed)}", flush=True)
    return bool(changed)
