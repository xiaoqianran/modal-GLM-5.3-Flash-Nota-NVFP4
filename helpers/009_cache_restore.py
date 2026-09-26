from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
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
    required_glob: str


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


def _path_has_required(root: Path, required_glob: str) -> bool:
    return root.exists() and any(root.glob(required_glob))


def _cache_ready(artifact: CacheArtifact) -> bool:
    return _path_has_required(Path(artifact.local_path), artifact.required_glob)


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
) -> str:
    """优先使用 Modal Volume；缺失时再从 GitHub Release 恢复缓存。"""
    if _cache_ready(artifact):
        print(f"[009_CACHE_VOLUME_HIT] name={artifact.name}", flush=True)
        return "modal"

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print(
            f"[009_CACHE_MISS] name={artifact.name} reason=no_github_token",
            flush=True,
        )
        return "miss"

    release_url = (
        f"https://api.github.com/repos/{github_repo}/releases/tags/"
        f"{urllib.parse.quote(release_tag, safe='')}"
    )

    try:
        release = _read_json(_github_request(release_url, token=token))
        asset = next(
            (
                item
                for item in release.get("assets", [])
                if item.get("name") == artifact.release_asset
            ),
            None,
        )
        if asset is None:
            print(
                f"[009_CACHE_MISS] name={artifact.name} reason=asset_not_found",
                flush=True,
            )
            return "miss"

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

            if not _path_has_required(extracted, artifact.required_glob):
                raise RuntimeError("downloaded archive failed cache validation")

            destination = Path(artifact.local_path)
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(extracted, destination, dirs_exist_ok=True)

    except urllib.error.HTTPError as exc:
        reason = "release_not_found" if exc.code == 404 else f"github_http_{exc.code}"
        print(
            f"[009_CACHE_MISS] name={artifact.name} reason={reason}",
            flush=True,
        )
        return "miss"
    except Exception as exc:
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
