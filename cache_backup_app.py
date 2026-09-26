from __future__ import annotations

import importlib

import modal

catalog = importlib.import_module("helpers.014_cache_catalog")
publish_module = importlib.import_module("helpers.010_cache_publish")
staging_module = importlib.import_module("helpers.012_cache_staging")
backup_state = importlib.import_module("helpers.015_cache_backup_state")

ALL_CACHE_ARTIFACTS = catalog.ALL_CACHE_ARTIFACTS
STAGED_CACHE_NAMES = catalog.STAGED_CACHE_NAMES
GITHUB_REPO = catalog.GITHUB_REPO
CACHE_RELEASE_TAG = catalog.CACHE_RELEASE_TAG
CACHE_BACKUP_APP_NAME = catalog.CACHE_BACKUP_APP_NAME
BACKUP_WORKER_VERSION = "2"

step_010_publish_cache = publish_module.step_010_publish_cache
step_012_cache_dirty = staging_module.step_012_cache_dirty
step_012_clear_dirty = staging_module.step_012_clear_dirty
is_cache_backup_dirty = backup_state.is_cache_backup_dirty
clear_cache_backup_dirty = backup_state.clear_cache_backup_dirty

cache_volumes = {
    artifact.name: modal.Volume.from_name(
        artifact.volume_name,
        create_if_missing=True,
    )
    for artifact in ALL_CACHE_ARTIFACTS
}
cache_mounts = {
    artifact.local_path: cache_volumes[artifact.name]
    for artifact in ALL_CACHE_ARTIFACTS
}

github_secret = modal.Secret.from_name("github")
cache_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir("helpers", "/root/helpers")
)

app = modal.App(CACHE_BACKUP_APP_NAME)


def _selected_artifacts(artifact_names: list[str] | None):
    if artifact_names is None:
        return ALL_CACHE_ARTIFACTS

    requested = set(artifact_names)
    known = {artifact.name for artifact in ALL_CACHE_ARTIFACTS}
    unknown = requested - known
    if unknown:
        raise ValueError(f"Unknown cache artifacts: {sorted(unknown)}")
    return tuple(
        artifact
        for artifact in ALL_CACHE_ARTIFACTS
        if artifact.name in requested
    )


def _backup(
    artifact_names: list[str] | None = None,
    *,
    replace_names: list[str] | None = None,
    force: bool = False,
) -> dict[str, bool]:
    print(
        f"[CACHE_BACKUP_WORKER] version={BACKUP_WORKER_VERSION}",
        flush=True,
    )
    replace_requested = set(replace_names or ())
    results: dict[str, bool] = {}

    for artifact in _selected_artifacts(artifact_names):
        generic_dirty = is_cache_backup_dirty(artifact.local_path)
        staged_dirty = (
            artifact.name in STAGED_CACHE_NAMES
            and step_012_cache_dirty(artifact.local_path)
        )
        replace_existing = (
            force
            or artifact.name in replace_requested
            or generic_dirty
            or staged_dirty
        )

        print(
            "[CACHE_GITHUB_BACKUP_START] "
            f"name={artifact.name} "
            f"replace={str(replace_existing).lower()} "
            f"generic_dirty={str(generic_dirty).lower()} "
            f"staged_dirty={str(staged_dirty).lower()}",
            flush=True,
        )

        published = step_010_publish_cache(
            artifact,
            github_repo=GITHUB_REPO,
            release_tag=CACHE_RELEASE_TAG,
            replace_existing=replace_existing,
        )
        results[artifact.name] = published

        if not published:
            continue

        marker_cleared = False
        if generic_dirty:
            clear_cache_backup_dirty(artifact.local_path)
            marker_cleared = True
        if staged_dirty:
            step_012_clear_dirty(artifact.local_path)
            marker_cleared = True
        if marker_cleared:
            cache_volumes[artifact.name].commit()
            print(
                "[CACHE_GITHUB_BACKUP_MARKERS_CLEARED] "
                f"name={artifact.name}",
                flush=True,
            )

        print(
            "[CACHE_GITHUB_BACKUP_DONE] "
            f"name={artifact.name} "
            f"replace={str(replace_existing).lower()}",
            flush=True,
        )

    return results


@app.function(
    image=cache_image,
    cpu=2,
    memory=8192,
    timeout=3600,
    max_containers=1,
    scaledown_window=60,
    secrets=[github_secret],
    volumes=cache_mounts,
)
def backup_runtime_caches(
    artifact_names: list[str] | None = None,
    replace_names: list[str] | None = None,
):
    """独立 CPU worker：仅在事件触发时备份可移植缓存。"""
    return _backup(
        artifact_names,
        replace_names=replace_names,
        force=False,
    )


@app.function(
    image=cache_image,
    cpu=2,
    memory=8192,
    timeout=3600,
    max_containers=1,
    scaledown_window=60,
    secrets=[github_secret],
    volumes=cache_mounts,
)
def backup_all_force():
    """人工/部署阶段使用：把所有已就绪 cache 强制刷新到 GitHub Release。"""
    return _backup(force=True)
