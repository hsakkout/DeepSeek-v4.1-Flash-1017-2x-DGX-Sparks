#!/usr/bin/env python3
import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("llm_switch_clients", ROOT / "scripts" / "llm_switch_clients.py")
clients = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clients)

YAML_SOURCE = """# preserve this comment
model:
  provider: ollama-cloud
  default: kimi-k3.1-pro
  context_length: 1048576
fallback_providers:
  - provider: custom:glm53-exl3
    model: GLM-5.3-Flash-EXL3
custom_providers:
  - name: glm53-exl3
    base_url: http://127.0.0.1:8888/v1
    model: GLM-5.3-Flash-EXL3
    api_key: ${env:GLM53_API_KEY}
    api_mode: chat_completions
    models:
      GLM-5.3-Flash-EXL3:
        context_length: 600000
"""


class ClientTests(unittest.TestCase):
    def test_hermes_retargets_only_local_references_and_preserves_comments(self):
        doc = clients.load_yaml(YAML_SOURCE)
        count = clients.retarget_hermes(doc, "dsv41")
        self.assertEqual(count, 1)
        self.assertEqual(doc["model"]["provider"], "ollama-cloud")
        self.assertEqual(doc["model"]["context_length"], 1048576)
        self.assertEqual(doc["fallback_providers"][0]["provider"], "custom:dsv41-exl3")
        self.assertEqual(doc["fallback_providers"][0]["model"], "DeepSeek-V4.1-Flash-1017")
        target = clients.provider(doc, "dsv41-exl3")
        self.assertEqual(target["models"]["DeepSeek-V4.1-Flash-1017"]["context_length"], 524288)
        self.assertEqual(target["api_key"], "${env:GLM53_API_KEY}")
        self.assertIn("# preserve this comment", clients.dump_yaml(doc))

    def test_switch_back_restores_glm_context(self):
        doc = clients.load_yaml(YAML_SOURCE)
        clients.retarget_hermes(doc, "dsv41")
        clients.retarget_hermes(doc, "glm")
        self.assertEqual(doc["fallback_providers"][0]["provider"], "custom:glm53-exl3")
        target = clients.provider(doc, "glm53-exl3")
        self.assertEqual(target["models"]["GLM-5.3-Flash-EXL3"]["context_length"], 600000)

    def test_local_primary_context_tracks_served_window(self):
        source = YAML_SOURCE.replace("provider: ollama-cloud", "provider: custom:glm53-exl3").replace(
            "default: kimi-k3.1-pro", "default: GLM-5.3-Flash-EXL3")
        doc = clients.load_yaml(source)
        clients.retarget_hermes(doc, "dsv41")
        self.assertEqual(doc["model"]["default"], "DeepSeek-V4.1-Flash-1017")
        self.assertEqual(doc["model"]["context_length"], 524288)

    def test_pi_adds_real_deepseek_provider_without_changing_cloud_default(self):
        doc = {"providers": {"glm-exl3": {
            "baseUrl": "http://127.0.0.1:8888/v1", "apiKey": "local-command",
            "models": [{"id": "GLM-5.3-Flash-EXL3", "name": "GLM", "contextWindow": 600000,
                        "maxTokens": 32768, "input": ["text", "image"]}],
        }}}
        settings = {"defaultProvider": "ollama-cloud", "defaultModel": "kimi-k3"}
        clients.retarget_pi(doc, settings, "dsv41")
        model = doc["providers"]["dsv41-exl3"]["models"][0]
        self.assertEqual(model["id"], "DeepSeek-V4.1-Flash-1017")
        self.assertEqual(model["contextWindow"], 524288)
        self.assertEqual(model["input"], ["text"])
        self.assertEqual(settings["defaultProvider"], "ollama-cloud")

    def test_apply_writes_atomically_and_creates_backup(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            config = root / ".hermes/config.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(YAML_SOURCE)
            config.chmod(0o600)
            changes = clients.plan(root, "dsv41")
            backup = clients.apply(changes, root / "state")
            self.assertNotEqual(config.read_text(), YAML_SOURCE)
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(backup.glob("*-config.yaml"))[0].read_text(), YAML_SOURCE)
            doc = clients.load_yaml(config.read_text())
            self.assertEqual(doc["fallback_providers"][0]["provider"], "custom:dsv41-exl3")

    def test_newline_only_pi_settings_difference_is_not_a_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            config = root / ".hermes/config.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(YAML_SOURCE)
            pi = root / ".pi/agent"
            pi.mkdir(parents=True)
            (pi / "models.json").write_text(json.dumps({"providers": {"glm-exl3": {
                "models": [{"id": "GLM-5.3-Flash-EXL3", "contextWindow": 600000}]
            }}}))
            settings = pi / "settings.json"
            settings.write_text('{"defaultProvider":"ollama-cloud","defaultModel":"kimi-k3"}')
            changes = clients.plan(root, "dsv41")
            self.assertNotIn(settings, changes)

    def test_hindsight_render_preserves_other_settings(self):
        original = "export HINDSIGHT_API_LLM_MODEL=GLM-5.3-Flash-EXL3\nOTHER=value\n"
        rendered = clients.render_hindsight(original, "dsv41")
        self.assertIn("export HINDSIGHT_API_LLM_MODEL=DeepSeek-V4.1-Flash-1017\n", rendered)
        self.assertIn("OTHER=value\n", rendered)
        with self.assertRaisesRegex(ValueError, "missing"):
            clients.render_hindsight("OTHER=value\n", "dsv41")

    @mock.patch.object(clients, "update_hindsight")
    def test_dry_run_does_not_write(self, hindsight):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            config = root / ".hermes/config.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(YAML_SOURCE)
            changed = clients.plan(root, "dsv41")
            self.assertIn(config, changed)
            self.assertEqual(config.read_text(), YAML_SOURCE)


if __name__ == "__main__":
    unittest.main()
