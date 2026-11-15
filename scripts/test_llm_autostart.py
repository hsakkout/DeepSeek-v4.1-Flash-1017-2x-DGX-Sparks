#!/usr/bin/env python3
import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/llm_autostart.sh"


class AutostartTests(unittest.TestCase):
    def run_script(self, state, selected, extra=None):
        env = os.environ.copy()
        env.update(DRY_RUN="1", GLM_AUTOSTART_STATE=str(state), LLM_SELECTED_FILE=str(state / "selected"))
        (state / "selected").write_text(selected + "\n")
        if extra:
            env.update(extra)
        return subprocess.run([str(SCRIPT)], env=env, text=True, capture_output=True, timeout=60)

    def test_invalid_selection_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            p = self.run_script(pathlib.Path(td), "bogus")
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("Invalid saved model selection", p.stdout + p.stderr)

    def test_current_handled_boot_pair_is_not_actioned(self):
        with tempfile.TemporaryDirectory() as td:
            state = pathlib.Path(td)
            pair = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            worker = subprocess.check_output([
                "ssh", "-o", "BatchMode=yes", "sakkout@192.168.200.2",
                "cat /proc/sys/kernel/random/boot_id",
            ], text=True).strip()
            handled = state / "handled"
            handled.write_text(pair + " " + worker + "\n")
            original = handled.read_bytes()
            p = self.run_script(state, "glm")
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertEqual(handled.read_bytes(), original)
            self.assertNotIn("would start", p.stdout + p.stderr)

    def test_new_boot_glm_dry_run_uses_existing_gates(self):
        with tempfile.TemporaryDirectory() as td:
            state = pathlib.Path(td)
            p = self.run_script(state, "glm")
            output = p.stdout + p.stderr
            self.assertEqual(p.returncode, 0, output)
            self.assertIn("would start glm", output)
            self.assertFalse((state / "handled").exists())

    def test_new_boot_deepseek_dry_run_selects_deepseek_without_preparation(self):
        with tempfile.TemporaryDirectory() as td:
            state = pathlib.Path(td)
            p = self.run_script(state, "dsv41", {"MIN_AVAIL_GIB": "1"})
            output = p.stdout + p.stderr
            self.assertEqual(p.returncode, 0, output)
            self.assertIn("would start dsv41 with preparation disabled", output)
            self.assertFalse((state / "handled").exists())
            self.assertIn("DEFAULT_MIN=112", SCRIPT.read_text())


if __name__ == "__main__":
    unittest.main()
