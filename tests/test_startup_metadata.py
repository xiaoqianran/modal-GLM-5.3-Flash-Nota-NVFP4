import contextlib
import importlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


metadata = importlib.import_module("helpers.018_startup_acceleration")
restore = importlib.import_module("helpers.009_cache_restore")


class TemporaryFiles(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def write_plan(self, fingerprint="a" * 16, kv=100):
        path = self.root / f"startup_plan_{fingerprint}.json"
        path.write_text(json.dumps({"schema": 1, "fingerprint": fingerprint,
                                    "kv_cache_memory_bytes": kv, "free_memory_baseline": 1000}))
        return path


class PlanTests(TemporaryFiles):
    def test_old_plan_does_not_hide_new_configuration(self):
        self.write_plan()
        before = metadata.startup_plan_fingerprints(str(self.root))
        new = self.write_plan("b" * 16)
        self.assertEqual(metadata.changed_startup_plans(before, str(self.root)), [new.name])

    def test_changed_payload_commits_but_formatting_and_mtime_do_not(self):
        path = self.write_plan()
        before = metadata.startup_plan_fingerprints(str(self.root))
        path.write_text(json.dumps(json.loads(path.read_text()), indent=4, sort_keys=True))
        self.assertEqual(metadata.changed_startup_plans(before, str(self.root)), [])
        self.write_plan(kv=200)
        self.assertEqual(metadata.changed_startup_plans(before, str(self.root)), [path.name])

    def test_invalid_partial_and_wrong_fingerprint_plans_are_excluded(self):
        valid = self.write_plan()
        self.write_plan("b" * 16, kv=-1)
        (self.root / ("startup_plan_" + "c" * 16 + ".json")).write_text(valid.read_text())
        (self.root / "startup_plan_broken.json").write_text("{")
        (self.root / "startup_plan_partial.json.tmp.12").write_text(valid.read_text())
        self.assertEqual(list(metadata.startup_plan_fingerprints(str(self.root))), [valid.name])

    def test_only_native_apply_is_an_actual_hit(self):
        observer = metadata.StartupPlanObserver()
        observer.observe("Saved startup plan to /tmp/startup_plan_aaaaaaaaaaaaaaaa.json")
        self.assertEqual(observer.result()[0], "unknown")
        observer.observe("Startup plan not applied: current free memory is below baseline")
        self.assertEqual(observer.result()[0], "false")
        observer.observe("Applying persisted startup plan (fingerprint aaaaaaaaaaaaaaaa): kv_cache_memory_bytes=100")
        self.assertEqual(observer.result(), ("true", ["a" * 16]))

    def test_background_commit_with_old_plan_and_new_plan(self):
        import app
        self.write_plan()
        before = metadata.startup_plan_fingerprints(str(self.root))
        self.write_plan("b" * 16)
        with patch.object(app, "VLLM_STARTUP_PLAN_CACHE", str(self.root)), \
             patch.object(app, "VLLM_LOCAL_CACHE_ROOT", str(self.root / "local")), \
             patch.object(app, "project_volume") as volume:
            thread = app._start_startup_plan_commit(before, str(self.root / "seed"), metadata.StartupPlanObserver())
            thread.join(5)
            self.assertFalse(thread.is_alive())
            volume.commit.assert_called_once()
            after = metadata.startup_plan_fingerprints(str(self.root))
            volume.reset_mock()
            app._start_startup_plan_commit(after, str(self.root / "seed"), metadata.StartupPlanObserver()).join(5)
            volume.commit.assert_not_called()


class ModelInfoTests(TemporaryFiles):
    def test_stage_to_local_and_publish_only_changed_valid_json(self):
        namespace = "a" * 64
        caps = self.root / "capabilities.json"
        caps.write_text(json.dumps({"cache_namespace": namespace, "modelinfo_cache_dir": "modelinfos"}))
        persistent = self.root / "seed"
        seed = persistent / namespace
        seed.mkdir(parents=True)
        old = {"hash": "source-hash", "modelinfo": {"is_text_generation_model": True}}
        (seed / "main.json").write_text(json.dumps(old))
        (seed / "broken.json").write_text("{")
        local = self.root / "runtime"
        actual_seed = metadata.prepare_modelinfo_cache(
            local_cache_root=str(local), persistent_root=str(persistent), capabilities_path=str(caps))
        self.assertEqual(actual_seed, str(seed))
        self.assertFalse((local / "modelinfos").is_symlink())
        self.assertFalse((local / "modelinfos/broken.json").exists())
        self.assertFalse(metadata.sync_modelinfo_cache(local_cache_root=str(local), persistent_dir=str(seed)))
        (local / "modelinfos/mtp.json").write_text(json.dumps(old))
        self.assertTrue(metadata.sync_modelinfo_cache(local_cache_root=str(local), persistent_dir=str(seed)))
        self.assertTrue((seed / "mtp.json").is_file())

    def test_image_namespace_must_not_escape_persistent_root(self):
        caps = self.root / "capabilities.json"
        caps.write_text(json.dumps({"cache_namespace": "../other", "modelinfo_cache_dir": "modelinfos"}))
        with self.assertRaises(RuntimeError):
            metadata.prepare_modelinfo_cache(local_cache_root=str(self.root / "runtime"),
                                            persistent_root=str(self.root / "seed"), capabilities_path=str(caps))


class AvailabilityTests(TemporaryFiles):
    def setUp(self):
        super().setUp()
        self.artifact = restore.CacheArtifact("flashinfer-jit", str(self.root / "jit"), "volume", "jit-v1.tgz", ("**/*.so",))
        self.options = dict(github_repo="owner/repo", release_tag="cache-v1",
                            availability_dir=str(self.root / "inventory"), availability_scope="image-v1")
        env = patch.dict(os.environ, {"GITHUB_TOKEN": "test-placeholder"})
        env.start()
        self.addCleanup(env.stop)

    def restore(self, policy):
        return restore.step_009_restore_cache(self.artifact, **self.options, remote_policy=policy)

    def test_absent_release_asset_is_not_queried_again_on_gpu(self):
        with patch.object(restore, "_read_json", return_value={"assets": []}) as query:
            self.assertEqual(self.restore("refresh"), "miss")
            query.assert_called_once()
            query.reset_mock()
            self.assertEqual(self.restore("deployment"), "miss")
            query.assert_not_called()
            self.restore("refresh")  # Explicit disaster recovery/deployment bypass.
            query.assert_called_once()

    def test_unknown_inventory_does_not_query_optional_asset(self):
        with patch.object(restore, "_read_json") as query:
            self.restore("deployment")
            query.assert_not_called()

    def test_volume_data_wins_over_negative_inventory(self):
        with patch.object(restore, "_read_json", return_value={"assets": []}) as query:
            self.restore("refresh")
            local = Path(self.artifact.local_path)
            local.mkdir()
            (local / "kernel.so").write_bytes(b"cache")
            query.reset_mock()
            self.assertEqual(self.restore("deployment"), "modal")
            query.assert_not_called()

    def test_network_failure_is_unknown_not_confirmed_absence(self):
        with patch.object(restore, "_read_json", side_effect=TimeoutError):
            self.restore("refresh")
        inventory = next((self.root / "inventory").glob("*.json"))
        self.assertEqual(json.loads(inventory.read_text())["state"], "unknown")

    def test_inventory_identity_includes_release_asset_and_image(self):
        args = (str(self.root), self.artifact, "owner/repo", "v1")
        self.assertNotEqual(restore._availability_path(*args, "image-1"),
                            restore._availability_path(*args, "image-2"))

    def test_cpu_restore_commits_negative_inventory_even_without_download(self):
        import app
        with patch.object(app, "_restore_cache_artifacts", return_value={"flashinfer-jit": "miss"}), \
             patch.object(app, "project_volume") as volume:
            app._restore_runtime_caches()
            volume.commit.assert_called_once()


class TraceTests(unittest.TestCase):
    def test_instrumentation_preserves_return_and_exceptions(self):
        from helpers.startup_trace import trace_call
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            fn = trace_call("registry_cache_lookup")(lambda self, key: key)
            self.assertEqual(fn(SimpleNamespace(class_name="Model"), "value"), "value")
            fail = trace_call("failure")(lambda: 1 / 0)
            with self.assertRaises(ZeroDivisionError):
                fail()
        self.assertIn('"actual_hit": true', output.getvalue())
        self.assertIn('"status": "error"', output.getvalue())

    def test_captured_registry_child_traces_are_forwarded_selectively(self):
        from helpers.startup_trace import forward_registry_trace
        output = io.StringIO()
        result = SimpleNamespace(stdout=b'other output\n[STARTUP_TRACE] {"pid": 7}\n',
                                 stderr=b'import time: 100 | 100 | module\nother stderr\n')
        with contextlib.redirect_stdout(output):
            forward_registry_trace(result)
        self.assertNotIn("other", output.getvalue())
        self.assertIn('"pid": 7', output.getvalue())
        self.assertIn("REGISTRY_IMPORT_TIME", output.getvalue())

    def test_engine_spawn_failure_restores_parent_environment(self):
        from helpers.startup_trace import start_engine_process
        process = Mock(name="engine")
        process.start.side_effect = RuntimeError("failed to spawn")
        process.name = "EngineCore"
        with patch.dict(os.environ, {"GLM53_ENGINE_SPAWN_AT": "prior"}):
            with self.assertRaises(RuntimeError):
                start_engine_process(process)
            self.assertEqual(os.environ["GLM53_ENGINE_SPAWN_AT"], "prior")

    def test_command_uses_verified_image_interpreter(self):
        import app
        command = app._build_vllm_command("/tmp/model")
        self.assertEqual(command[:3], ["/usr/bin/python3", "-m", "helpers.020_vllm_bootstrap"])


if __name__ == "__main__":
    unittest.main()
