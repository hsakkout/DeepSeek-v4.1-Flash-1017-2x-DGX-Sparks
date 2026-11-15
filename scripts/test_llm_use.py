#!/usr/bin/env python3
import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("llm_use", ROOT / "scripts" / "llm_use.py")
llm_use = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(llm_use)


class FakeDriver:
    def __init__(self, current="glm"):
        self.current = current
        self.selected_client = current
        self.fail_start = set()
        self.fail_verify = set()
        self.calls = []
        self.stopped_checked = 0

    def current_model(self):
        return self.current

    def drain(self, model):
        self.calls.append(("drain", model))

    def stop(self, model):
        self.calls.append(("stop", model))
        if self.current == model:
            self.current = None

    def assert_stopped(self):
        self.stopped_checked += 1
        if self.current is not None:
            raise llm_use.SwitchError("a rank is still running")

    def start(self, model):
        self.calls.append(("start", model))
        if model in self.fail_start:
            raise llm_use.SwitchError("injected start failure")
        self.current = model

    def verify(self, model):
        self.calls.append(("verify", model))
        if model in self.fail_verify or self.current != model:
            raise llm_use.SwitchError("injected verification failure")

    def update_clients(self, model):
        self.calls.append(("clients", model))
        self.selected_client = model


class SwitchCoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = pathlib.Path(self.tmp.name)
        self.cores = []

    def tearDown(self):
        for core in self.cores:
            core.close()
        self.tmp.cleanup()

    def core(self, driver):
        core = llm_use.SwitchCore(self.state, driver)
        self.cores.append(core)
        return core

    def test_successful_switch_commits_only_after_verification_and_clients(self):
        d = FakeDriver("glm")
        self.core(d).switch("dsv41")
        self.assertEqual(d.current, "dsv41")
        self.assertEqual(d.selected_client, "dsv41")
        self.assertEqual((self.state / "selected").read_text(), "dsv41\n")
        self.assertEqual(d.calls, [
            ("drain", "glm"), ("stop", "glm"), ("start", "dsv41"),
            ("verify", "dsv41"), ("clients", "dsv41"),
        ])

    def test_failed_target_start_restores_running_source_and_selection(self):
        d = FakeDriver("glm")
        d.fail_start.add("dsv41")
        (self.state / "selected").write_text("glm\n")
        with self.assertRaisesRegex(llm_use.SwitchError, "rolled back to glm"):
            self.core(d).switch("dsv41")
        self.assertEqual(d.current, "glm")
        self.assertEqual(d.selected_client, "glm")
        self.assertEqual((self.state / "selected").read_text(), "glm\n")
        self.assertIn(("start", "glm"), d.calls)
        self.assertIn(("verify", "glm"), d.calls)

    def test_failed_target_verification_also_rolls_back(self):
        d = FakeDriver("glm")
        d.fail_verify.add("dsv41")
        with self.assertRaisesRegex(llm_use.SwitchError, "rolled back to glm"):
            self.core(d).switch("dsv41")
        self.assertEqual(d.current, "glm")
        self.assertEqual(d.calls[-1], ("clients", "glm"))

    def test_start_from_manually_stopped_state_does_not_restart_source_on_failure(self):
        d = FakeDriver(None)
        d.fail_start.add("dsv41")
        (self.state / "selected").write_text("glm\n")
        with self.assertRaisesRegex(llm_use.SwitchError, "left stopped"):
            self.core(d).switch("dsv41")
        self.assertIsNone(d.current)
        self.assertNotIn(("start", "glm"), d.calls)
        self.assertEqual((self.state / "selected").read_text(), "glm\n")

    def test_already_running_target_is_idempotent(self):
        d = FakeDriver("dsv41")
        self.core(d).switch("dsv41")
        self.assertNotIn(("stop", "dsv41"), d.calls)
        self.assertNotIn(("start", "dsv41"), d.calls)
        self.assertIn(("verify", "dsv41"), d.calls)

    def test_unknown_current_model_is_refused(self):
        d = FakeDriver("unexpected-model")
        with self.assertRaisesRegex(llm_use.SwitchError, "Refusing"):
            self.core(d).switch("glm")
        self.assertEqual(d.calls, [])

    def test_restart_restarts_the_same_model(self):
        d = FakeDriver("glm")
        self.core(d).switch("glm", force_restart=True)
        self.assertIn(("stop", "glm"), d.calls)
        self.assertIn(("start", "glm"), d.calls)

    def test_invalid_selection_is_rejected(self):
        d = FakeDriver(None)
        (self.state / "selected").write_text("bogus\n")
        with self.assertRaisesRegex(llm_use.SwitchError, "selection"):
            self.core(d).switch("dsv41")


class LocalDriverTests(unittest.TestCase):
    @mock.patch.object(llm_use.subprocess, "run")
    def test_lowercase_docker_absence_is_absent_not_unknown(self, run):
        run.return_value = mock.Mock(returncode=1, stdout="", stderr="error: no such object: glm53-spark\n")
        with tempfile.TemporaryDirectory() as td:
            driver = llm_use.LocalDriver(pathlib.Path(td))
            self.assertEqual(driver.rank_state("glm", "head"), "absent")
    def test_client_adapter_is_invoked_in_apply_mode(self):
        with tempfile.TemporaryDirectory() as td:
            driver = llm_use.LocalDriver(pathlib.Path(td))
            driver.run = mock.Mock()
            driver.update_clients("dsv41")
            args = driver.run.call_args.args[0]
            self.assertIn("--apply", args)


if __name__ == "__main__":
    unittest.main()
