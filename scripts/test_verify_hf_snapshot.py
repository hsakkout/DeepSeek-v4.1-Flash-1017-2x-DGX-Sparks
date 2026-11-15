#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_hf_snapshot", ROOT / "scripts/verify_hf_snapshot.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class VerifySnapshotTests(unittest.TestCase):
    def fixture(self, root):
        snap = root / "snapshot"
        snap.mkdir()
        small = b"small\n"
        large = b"large-weight-bytes"
        (snap / "small.txt").write_bytes(small)
        (snap / "large.bin").write_bytes(large)
        entries = [
            {"path": "small.txt", "size": len(small), "blob_id": verifier.git_blob(small)},
            {"path": "large.bin", "size": len(large), "blob_id": "unused",
             "lfs": {"size": len(large), "sha256": hashlib.sha256(large).hexdigest()}},
        ]
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(entries))
        return snap, manifest

    def test_git_and_lfs_hashes_pass(self):
        with tempfile.TemporaryDirectory() as td:
            snap, manifest = self.fixture(pathlib.Path(td))
            result = verifier.verify(manifest, snap)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["files"], 2)
            self.assertEqual(result["bytes"], 24)

    def test_size_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            snap, manifest = self.fixture(pathlib.Path(td))
            (snap / "large.bin").write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verifier.verify(manifest, snap)

    def test_extra_file_fails(self):
        with tempfile.TemporaryDirectory() as td:
            snap, manifest = self.fixture(pathlib.Path(td))
            (snap / "extra").write_text("unexpected")
            with self.assertRaisesRegex(ValueError, "file-set mismatch"):
                verifier.verify(manifest, snap)


if __name__ == "__main__":
    unittest.main()
