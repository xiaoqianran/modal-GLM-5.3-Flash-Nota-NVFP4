"""Verify the installed image's startup contracts and add timing-only probes."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import py_compile
import sys
from pathlib import Path


SITE_PACKAGES = Path("/usr/local/lib/python3.12/dist-packages")
CAPABILITIES = Path("/root/glm53-startup-capabilities.json")
TRACE_IMPORT = "from helpers import startup_trace as _glm53_trace"
STARTUP_PLAN_TOLERANCE_ENV = "GLM53_STARTUP_PLAN_FREE_MEMORY_TOLERANCE_MIB"
STARTUP_PLAN_TOLERANCE_DEFAULT_MIB = 256
STARTUP_PLAN_TOLERANCE_MAX_MIB = 1024


def _function(tree, owner: str | None, name: str):
    scope = tree
    if owner:
        matches = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner]
        if len(matches) != 1:
            raise RuntimeError(f"Expected class {owner}")
        scope = matches[0]
    matches = [n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected function {owner}.{name}")
    return matches[0]


def _decorate(text: str, owner: str | None, name: str, phase: str) -> str:
    node = _function(ast.parse(text), owner, name)
    lines = text.splitlines(keepends=True)
    lines.insert(node.lineno - 1, " " * node.col_offset + f'@_glm53_trace.trace_call("{phase}")\n')
    return "".join(lines)


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Image startup source changed: expected one {old!r}")
    return text.replace(old, new, 1)


def _add_import(text: str) -> str:
    tree = ast.parse(text)
    index = 0
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)) or (
                isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            index = node.end_lineno
        else:
            break
    lines = text.splitlines(keepends=True)
    lines.insert(index, "\n" + TRACE_IMPORT + "\n")
    return "".join(lines)


def patch_startup_plan_tolerance(text: str) -> str:
    """Allow small allocator jitter without removing vLLM's OOM safety gate."""
    old = '''    if current_free_memory < baseline:\n        logger.info(\n            "Startup plan not applied: current free memory (%.2f GiB) is "\n            "below the recorded baseline (%.2f GiB); falling back to full "\n            "memory profiling.",\n            current_free_memory / (1 << 30),\n            baseline / (1 << 30),\n        )\n        return None\n    return kv_bytes\n'''
    new = f'''    try:\n        tolerance_mib = int(os.getenv(\n            "{STARTUP_PLAN_TOLERANCE_ENV}",\n            "{STARTUP_PLAN_TOLERANCE_DEFAULT_MIB}",\n        ))\n    except ValueError:\n        tolerance_mib = {STARTUP_PLAN_TOLERANCE_DEFAULT_MIB}\n    tolerance_mib = min(max(tolerance_mib, 0), {STARTUP_PLAN_TOLERANCE_MAX_MIB})\n    tolerance_bytes = tolerance_mib * (1 << 20)\n    deficit = baseline - current_free_memory\n    if deficit > tolerance_bytes:\n        logger.info(\n            "Startup plan not applied: current free memory (%.2f GiB) is "\n            "below the recorded baseline (%.2f GiB) by %.1f MiB, exceeding "\n            "the %d MiB tolerance; falling back to full memory profiling.",\n            current_free_memory / (1 << 30),\n            baseline / (1 << 30),\n            deficit / (1 << 20),\n            tolerance_mib,\n        )\n        return None\n    if deficit > 0:\n        logger.info(\n            "Startup plan free-memory jitter accepted: current %.2f GiB, "\n            "recorded baseline %.2f GiB, deficit %.1f MiB within %d MiB "\n            "tolerance.",\n            current_free_memory / (1 << 30),\n            baseline / (1 << 30),\n            deficit / (1 << 20),\n            tolerance_mib,\n        )\n    return kv_bytes\n'''
    return _replace_once(text, old, new)


def patch_sources(root: Path, versions: dict[str, str]) -> dict:
    paths = {
        "registry": root / "model_executor/models/registry.py",
        "plan": root / "v1/worker/startup_plan.py",
        "engine": root / "v1/engine/core.py",
        "process": root / "v1/engine/utils.py",
        "worker": root / "v1/worker/gpu_worker.py",
        "model_runner": root / "v1/worker/gpu_model_runner.py",
        "cli": root / "entrypoints/cli/main.py",
    }
    source = {key: path.read_text(encoding="utf-8") for key, path in paths.items()}
    registry = source["registry"]
    # Fail the image build instead of silently staging an unrelated directory.
    cache_dir = _function(ast.parse(registry), "_LazyRegisteredModel", "_get_cache_dir")
    if '"modelinfos"' not in ast.get_source_segment(registry, cache_dir):
        raise RuntimeError("Unverified modelinfos layout")
    for method in ("_load_modelinfo_from_cache", "_get_modelinfo_module_hash"):
        _function(ast.parse(registry), "_LazyRegisteredModel", method)
    if 'mi_dict["hash"] != module_hash' not in registry:
        raise RuntimeError("Unverified registry source-hash validation")
    plan = source["plan"]
    if "PLAN_SCHEMA_VERSION = 1" not in plan:
        raise RuntimeError("Unverified startup plan schema")
    for marker in ("Applying persisted startup plan (fingerprint %s)",
                   "Saved startup plan to %s", "current_free_memory < baseline"):
        if marker not in plan:
            raise RuntimeError(f"Unverified startup plan contract: {marker}")
    plan = patch_startup_plan_tolerance(plan)
    compile(plan, str(paths["plan"]), "exec")
    paths["plan"].write_text(plan, encoding="utf-8")
    source["plan"] = plan
    # Removing timing decorations gives stable identities on repeated patch runs.
    identity = {"versions": versions, "plan_schema": 1}
    info_class = next(n for n in ast.parse(registry).body
                      if isinstance(n, ast.ClassDef) and n.name == "_ModelInfo")
    identity["modelinfo_schema"] = hashlib.sha256(ast.dump(info_class).encode()).hexdigest()
    # Native model module hashes remain the final validity check. Registry layout
    # and installed library versions namespace seeds across image revisions.
    identity["registry_contract"] = "modelinfos-source-hash-v1"
    namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    targets = {
        "registry": [("_LazyRegisteredModel", "inspect_model_cls", "registry_inspect"),
                     ("_LazyRegisteredModel", "_load_modelinfo_from_cache", "registry_cache_lookup"),
                     ("_LazyRegisteredModel", "load_model_cls", "model_class_import"),
                     (None, "_run_in_subprocess", "registry_subprocess"),
                     (None, "_run", "registry_child")],
        "engine": [("EngineCoreProc", "run_engine_core", "engine_entry"),
                   ("EngineCore", "__init__", "engine_init")],
        "worker": [("Worker", "init_device", "worker_init_device")],
        "model_runner": [("GPUModelRunner", "profile_run", "model_profile_run"),
                         ("GPUModelRunner", "capture_model", "model_capture")],
    }
    targets["worker"] += [
        ("Worker", "determine_available_memory", "determine_available_memory"),
        ("Worker", "initialize_from_config", "initialize_from_config"),
        ("Worker", "compile_or_warm_up_model", "compile_or_warm_up_model"),
    ]
    for key in ("registry", "engine", "worker", "model_runner", "process", "cli"):
        text = source[key]
        if TRACE_IMPORT in text:
            continue
        for owner, name, phase in targets.get(key, []):
            text = _decorate(text, owner, name, phase)
        if key == "registry":
            text = _replace_once(text, "        # check if the subprocess is successful\n",
                                 "        _glm53_trace.forward_registry_trace(returned)\n\n"
                                 "        # check if the subprocess is successful\n")
        elif key == "process":
            text = _replace_once(text, "                    proc.start()\n",
                                 "                    _glm53_trace.start_engine_process(proc)\n")
        elif key == "cli":
            node = _function(ast.parse(text), None, "main")
            imports = []
            for statement in node.body:
                if not isinstance(statement, (ast.Import, ast.ImportFrom)):
                    break
                imports.append(statement)
            if not imports:
                raise RuntimeError("Expected CLI command import block")
            lines = text.splitlines(keepends=True)
            start, end = imports[0].lineno - 1, imports[-1].end_lineno
            lines[start:end] = ['    with _glm53_trace.span("cli_command_imports"):\n'] + [
                "    " + line if line.strip() else line for line in lines[start:end]
            ]
            text = "".join(lines)
        text = _add_import(text)
        compile(text, str(paths[key]), "exec")
        paths[key].write_text(text, encoding="utf-8")
    # The patch above changes mtimes/content of several large vLLM modules.
    # Precompile those exact files into the image so every new container does
    # not pay source->bytecode compilation again on first import.
    for path in paths.values():
        py_compile.compile(str(path), doraise=True)
    return {"modelinfo_cache_dir": "modelinfos", "cache_namespace": namespace,
            "startup_plan_schema": 1,
            "startup_plan_free_memory_tolerance_default_mib": STARTUP_PLAN_TOLERANCE_DEFAULT_MIB,
            "startup_plan_free_memory_tolerance_max_mib": STARTUP_PLAN_TOLERANCE_MAX_MIB,
            "patched_sources_precompiled": True,
            "versions": versions,
            "source_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}}


def main() -> None:
    shebang = Path("/usr/local/bin/vllm").read_text().splitlines()[0]
    if shebang != f"#!{sys.executable}":
        raise RuntimeError(f"vLLM interpreter changed: {shebang}, patch runner={sys.executable}")
    versions = {}
    for package in ("vllm", "torch", "transformers", "flashinfer-python"):
        versions[package] = importlib.metadata.version(package)
    capabilities = patch_sources(SITE_PACKAGES / "vllm", versions)
    capabilities["runtime_python"] = sys.executable
    CAPABILITIES.write_text(json.dumps(capabilities, indent=2), encoding="utf-8")
    print("[IMAGE_STARTUP_CAPABILITIES] " + json.dumps(capabilities, sort_keys=True))


if __name__ == "__main__":
    main()
