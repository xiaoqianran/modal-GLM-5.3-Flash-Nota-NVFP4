import contextlib
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from filelock import FileLock

from helpers.startup_prefetch import start_prefetch


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = patch.dict(os.environ, {"VLLM_PREFETCH_STATE_DIR": str(self.root / "state")})
        env.start()
        self.addCleanup(env.stop)
        self.shard = self.root / "model.safetensors"
        self.shard.write_bytes(b"weights" * 100)

    def start(self, paths=None):
        return start_prefetch(paths or [str(self.shard)], num_prefetch_threads=2, block_size=32)

    def test_other_process_does_not_wait_on_inflight_owner(self):
        state = self.root / "state"
        state.mkdir()
        digest = hashlib.sha256(str(self.shard.absolute()).encode()).hexdigest()
        with FileLock(str(state / f"{digest}.lock")):
            result = subprocess.run(
                [sys.executable, "-c",
                 "from helpers.startup_prefetch import start_prefetch; "
                 "import sys; assert start_prefetch([sys.argv[1]], "
                 "num_prefetch_threads=1, block_size=32) is None", str(self.shard)],
                capture_output=True, text=True, timeout=5,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("state=inflight action=load-without-wait", result.stdout)

    def test_completed_prefetch_is_reused(self):
        thread = self.start()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(self.start())

    def test_failure_does_not_publish_success_and_can_retry(self):
        self.shard.unlink()
        self.start().join(5)
        self.assertFalse(list((self.root / "state").glob("*.done")))
        self.shard.write_bytes(b"retry")
        self.start().join(5)
        self.assertIsNone(self.start())

    def test_overlapping_shard_subsets_read_each_file_once(self):
        other = self.root / "other.safetensors"
        other.write_bytes(b"other")
        entered, release = threading.Event(), threading.Event()
        real_open = open
        reads = []

        def slow_open(path, *args, **kwargs):
            reads.append(path)
            if path == str(self.shard):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test reader was not released")
            return real_open(path, *args, **kwargs)

        with patch("helpers.startup_prefetch.open", side_effect=slow_open):
            first = self.start()
            try:
                self.assertTrue(entered.wait(5))
                second = self.start([str(self.shard), str(other)])
            finally:
                release.set()
            first.join(5)
            second.join(5)
        self.assertEqual(reads.count(str(self.shard)), 1)
        self.assertEqual(reads.count(str(other)), 1)

    def test_bootstrap_prefetch_precedes_vllm_import_and_has_fresh_namespace(self):
        bootstrap = importlib.import_module("helpers.020_vllm_bootstrap")
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": self.shard.name}})
        )
        import builtins
        real_import = builtins.__import__
        events = []

        def import_hook(name, *args, **kwargs):
            if name == "vllm.entrypoints.cli.main":
                self.assertEqual(events, ["prefetch"])
                return type("CLI", (), {"main": staticmethod(lambda: events.append("cli"))})
            return real_import(name, *args, **kwargs)

        with patch.object(sys, "argv", ["bootstrap", "serve", str(self.root),
                                       "--safetensors-load-strategy", "prefetch"]), \
             patch("helpers.startup_prefetch.start_prefetch", side_effect=lambda *a, **k: events.append("prefetch")), \
             patch("builtins.__import__", side_effect=import_hook), \
             patch.object(bootstrap.tempfile, "mkdtemp", return_value=str(self.root / "fresh")):
            bootstrap.main()
        self.assertEqual(events, ["prefetch", "cli"])
        self.assertEqual(os.environ["VLLM_PREFETCH_STATE_DIR"], str(self.root / "fresh"))


class CacheTests(unittest.TestCase):
    def test_partial_hit_is_not_reported_as_full_hit_and_save_notifies(self):
        module = importlib.import_module("helpers.005_kernel_jit")
        saves = []
        observer = module.step_005_create_kernel_jit_observer(
            0, on_flashinfer_saved=lambda: saves.append(True),
        )
        for new, previous, full in [(3, 60, "false"), (0, 63, "true"), (63, 0, "false")]:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                observer.observe(f"Saved 63 configs to /cache (" 
                                 f"{new} new, {previous} from previous config)")
            self.assertIn(f"actual_cache_hit={full}", output.getvalue())
        self.assertEqual(len(saves), 3)

    def test_flashinfer_commit_independent_of_bulk_sync(self):
        import app
        from unittest.mock import Mock
        volume = Mock()
        for after, expected in [((), 0), (("new-config",), 1)]:
            volume.reset_mock()
            with patch.object(app, "RUNTIME_CACHE_SYNC_ENABLED", False), \
                 patch.object(app, "cache_fingerprint", return_value=after), \
                 patch.object(app, "project_volume", volume), \
                 patch.object(app, "mark_cache_backup_dirty") as dirty, \
                 patch.object(app, "_spawn_cache_backup") as backup:
                thread = app._start_flashinfer_cache_commit(())
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(volume.commit.call_count, expected)
                self.assertEqual(dirty.call_count, expected)
                self.assertEqual(backup.call_count, expected)


if __name__ == "__main__":
    unittest.main()
