from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheArtifact:
    """描述一个可由 Modal Volume 或 GitHub Release 提供的运行时缓存。"""

    name: str
    local_path: str
    volume_name: str
    release_asset: str
    required_globs: tuple[str, ...]


def _github_request(
    url: str,
    *,
    method: str = "GET",
    token: str | None = None,
    data: bytes | None = None,
    content_type: str | None = None,
    accept: str = "application/vnd.github+json",
) -> urllib.request.Request:
    headers = {
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "modal-glm53-cache-manager",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if content_type:
        headers["Content-Type"] = content_type
    return urllib.request.Request(url, data=data, headers=headers, method=method)


def _read_json(request: urllib.request.Request) -> dict:
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _path_has_required(root: Path, required_globs: tuple[str, ...]) -> bool:
    if not root.exists():
        return False
    ignored = {".staged-cache-dirty", ".github-backup-dirty"}
    return any(
        any(
            path.is_file() and path.name not in ignored
            for path in root.glob(pattern)
        )
        for pattern in required_globs
    )


def _cache_ready(artifact: CacheArtifact) -> bool:
    return _path_has_required(Path(artifact.local_path), artifact.required_globs)


def _availability_path(directory: str, artifact: CacheArtifact, repo: str,
                       tag: str, scope: str) -> Path:
    identity = json.dumps([repo, tag, scope, artifact.name, artifact.release_asset])
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return Path(directory) / f"{digest}.json"


def _record_availability(path: Path | None, state: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".availability-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"schema": 1, "state": state, "checked_at": time.time()}, stream)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_availability(path: Path | None) -> str:
    if path is not None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") == 1:
                return payload.get("state", "unknown")
        except (OSError, ValueError, AttributeError):
            pass
    return "unknown"


def cache_fingerprint(artifact: CacheArtifact) -> tuple[tuple[str, int, str], ...]:
    """Return a content fingerprint so mtime-only rewrites do not republish assets."""
    root = Path(artifact.local_path)
    if not root.exists():
        return ()

    records: dict[str, tuple[str, int, str]] = {}
    for pattern in artifact.required_globs:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
            rel = path.relative_to(root).as_posix()
            records[rel] = (rel, size, digest)
    return tuple(records[key] for key in sorted(records))


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe cache archive member: {member.name}")
        tar.extractall(destination)


def step_009_restore_cache(
    artifact: CacheArtifact,
    *,
    github_repo: str,
    release_tag: str,
    availability_dir: str | None = None,
    availability_scope: str = "",
    remote_policy: str = "refresh",
) -> str:
    """优先使用 Modal Volume；缺失时再从 GitHub Release 恢复缓存。"""
    if remote_policy not in {"refresh", "deployment"}:
        raise ValueError(f"Unknown remote cache policy: {remote_policy}")
    availability = (
        _availability_path(availability_dir, artifact, github_repo, release_tag, availability_scope)
        if availability_dir else None
    )
    if _cache_ready(artifact):
        if remote_policy == "refresh":
            _record_availability(availability, "local")
        print(f"[009_CACHE_VOLUME_SEED] name={artifact.name}", flush=True)
        return "modal"

    if remote_policy == "deployment":
        state = _read_availability(availability)
        if state not in {"available", "local"}:
            print(f"[009_CACHE_REMOTE_SKIP] name={artifact.name} "
                  f"deployment_state={state} reason=optional_cache_not_available", flush=True)
            return "miss"

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        if remote_policy == "refresh":
            _record_availability(availability, "unknown")
        print(
            f"[009_CACHE_MISS] name={artifact.name} reason=no_github_token",
            flush=True,
        )
        return "miss"

    release_url = (
        f"https://api.github.com/repos/{github_repo}/releases/tags/"
        f"{urllib.parse.quote(release_tag, safe='')}"
    )

    release_loaded = False
    try:
        release = _read_json(_github_request(release_url, token=token))
        release_loaded = True
        asset = next(
            (
                item
                for item in release.get("assets", [])
                if item.get("name") == artifact.release_asset
            ),
            None,
        )
        if asset is None:
            _record_availability(availability, "absent")
            print(
                f"[009_CACHE_MISS] name={artifact.name} reason=asset_not_found",
                flush=True,
            )
            return "miss"

        _record_availability(availability, "available")

        with tempfile.TemporaryDirectory(prefix="glm53-cache-") as temp_dir:
            temp_root = Path(temp_dir)
            archive = temp_root / artifact.release_asset
            extracted = temp_root / "extracted"
            request = _github_request(
                asset["url"],
                token=token,
                accept="application/octet-stream",
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                with archive.open("wb") as output:
                    shutil.copyfileobj(response, output)
            _safe_extract(archive, extracted)

            if not _path_has_required(extracted, artifact.required_globs):
                raise RuntimeError("downloaded archive failed cache validation")

            destination = Path(artifact.local_path)
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(extracted, destination, dirs_exist_ok=True)

    except urllib.error.HTTPError as exc:
        if remote_policy == "refresh":
            _record_availability(availability, "absent" if exc.code == 404 and not release_loaded else "unknown")
        reason = "release_not_found" if exc.code == 404 else f"github_http_{exc.code}"
        print(
            f"[009_CACHE_MISS] name={artifact.name} reason={reason}",
            flush=True,
        )
        return "miss"
    except Exception as exc:
        if remote_policy == "refresh":
            _record_availability(availability, "unknown")
        print(
            f"[009_CACHE_MISS] name={artifact.name} "
            f"reason=github_restore_failed error={type(exc).__name__}",
            flush=True,
        )
        return "miss"

    print(
        f"[009_CACHE_GITHUB_HIT] name={artifact.name} asset={artifact.release_asset}",
        flush=True,
    )
    return "github"
