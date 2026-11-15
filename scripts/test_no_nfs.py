#!/usr/bin/env python3
import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class NoNfsTests(unittest.TestCase):
    def test_missing_sha_gate_fails_before_preflight(self):
        env = os.environ.copy()
        env.update(NO_NFS="1", VERIFY_SHA="0", ABLIT="0")
        p = subprocess.run(["./scripts/prepare.sh"], cwd=ROOT, env=env, text=True,
                           capture_output=True, timeout=20)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("requires VERIFY_SHA=1", p.stdout + p.stderr)
        self.assertNotIn("Preflight checks", p.stdout + p.stderr)

    def test_ablit_is_refused_before_preflight(self):
        env = os.environ.copy()
        env.update(NO_NFS="1", VERIFY_SHA="1", ABLIT="1")
        p = subprocess.run(["./scripts/prepare.sh"], cwd=ROOT, env=env, text=True,
                           capture_output=True, timeout=20)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("stock weights only", p.stdout + p.stderr)
        self.assertNotIn("Preflight checks", p.stdout + p.stderr)

    def test_snapshot_filters_copy_only_vision_and_remove_stale_shards(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            src = root / "src"
            dst = root / "dst"
            src.mkdir()
            dst.mkdir()
            (src / "config.json").write_text("{}")
            (src / "model-00039-of-00039.safetensors").write_bytes(b"vision")
            (src / "model-00001-of-00039.safetensors").write_bytes(b"not-needed")
            (dst / "model-00038-of-00039.safetensors").write_bytes(b"stale")
            subprocess.run([
                "rsync", "-aL", "--delete-excluded", "--prune-empty-dirs",
                "--include=*/", "--include=model-00039-of-00039.safetensors",
                "--exclude=model-*.safetensors", "--include=*", str(src) + "/", str(dst) + "/",
            ], check=True)
            self.assertEqual((dst / "config.json").read_text(), "{}")
            self.assertEqual((dst / "model-00039-of-00039.safetensors").read_bytes(), b"vision")
            self.assertFalse((dst / "model-00001-of-00039.safetensors").exists())
            self.assertFalse((dst / "model-00038-of-00039.safetensors").exists())

    def test_disk_budget_covers_stale_rank_repair_and_one_filesystem(self):
        script = (ROOT / "scripts/prepare-local-worker.sh").read_text()
        self.assertIn("stat -c %d '$NFS_VOLUME'", script)
        self.assertIn("stat -c %d '$WORKER_ENGRAM_DIR'", script)
        self.assertIn("stat -c %d '$WORKER_CACHE_DIR'", script)
        self.assertIn("50 * 1024 ** 3 + 120 * 1024 ** 3", script)
        self.assertNotIn("|| worker_need=", script)
        self.assertIn("set -o pipefail; tf-dsv41-rank-cache", script)
        self.assertIn("^[0-9a-f]{20}$", script)

    def test_start_dry_run_uses_absolute_worker_host_paths(self):
        env = os.environ.copy()
        env.update(DRY_RUN="1", PREPARE="0")
        p = subprocess.run(["./start.sh"], cwd=ROOT, env=env, text=True,
                           capture_output=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        output = p.stdout + p.stderr
        self.assertIn("local HF at /home/sakkout/.cache/dsv41-tensorfold/worker-hf", output)
        self.assertIn("-v /home/sakkout/.cache/dsv41-tensorfold/worker-hf:/hf:ro", output)
        self.assertIn("-v /home/sakkout/.cache/dsv41-tensorfold/engram-1bae0e10:/engram:ro", output)
        self.assertIn("-v /home/sakkout/.cache/dsv41-tensorfold/rank-cache-45ee5d0f:/rank-cache:ro", output)
        self.assertNotIn("docker volume create", output)
        self.assertNotIn("-v $HOME/.cache/dsv41-tensorfold", output)


if __name__ == "__main__":
    unittest.main()
