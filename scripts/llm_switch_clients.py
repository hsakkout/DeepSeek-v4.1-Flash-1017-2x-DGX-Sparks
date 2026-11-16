#!/usr/bin/env python3
"""Plan or apply local-model client selection without touching cloud defaults."""
from __future__ import annotations

import argparse
import copy
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from ruamel.yaml import YAML

TARGETS = {
    "glm": {
        "provider": "glm53-exl3", "model": "GLM-5.3-Flash-EXL3",
        "context": 600000, "pi_provider": "glm-exl3", "pi_name": "GLM-5.3-Flash EXL3 (2x DGX Spark TP2)",
    },
    "dsv41": {
        "provider": "dsv41-exl3", "model": "DeepSeek-V4.1-Flash-1017",
        "context": 524288, "pi_provider": "dsv41-exl3", "pi_name": "DeepSeek-V4.1-Flash-1017 (2x DGX Spark TP2)",
    },
}
LOCAL_PROVIDERS = {"custom:glm53-exl3", "custom:dsv41-exl3"}
LOCAL_PI_PROVIDERS = {"glm-exl3", "dsv41-exl3"}


def yaml_object(typ="rt"):
    y = YAML(typ=typ)
    y.preserve_quotes = True
    return y


def load_yaml(text):
    return yaml_object().load(text)


def dump_yaml(doc) -> str:
    out = StringIO()
    yaml_object().dump(doc, out)
    return out.getvalue()


def provider(doc, name):
    entries = doc.get("custom_providers") or []
    if not isinstance(entries, list):
        raise ValueError("custom_providers must be a list")
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    raise ValueError(f"missing custom provider {name}")


def retarget_hermes(doc, target_name: str) -> int:
    target = TARGETS[target_name]
    entries = doc.get("custom_providers") or []
    if not isinstance(entries, list):
        raise ValueError("custom_providers must be a list")
    source = provider(doc, "glm53-exl3")
    try:
        selected = provider(doc, target["provider"])
    except ValueError:
        selected = copy.deepcopy(source)
        entries.append(selected)
    selected["name"] = target["provider"]
    selected["base_url"] = source.get("base_url", "http://127.0.0.1:8888/v1")
    selected["model"] = target["model"]
    selected["models"] = {target["model"]: {"context_length": target["context"]}}

    primary = doc.get("model") or {}
    primary_local = isinstance(primary, dict) and primary.get("provider") in LOCAL_PROVIDERS
    count = 0

    def walk(node):
        nonlocal count
        if isinstance(node, dict):
            if node.get("provider") in LOCAL_PROVIDERS:
                node["provider"] = "custom:" + target["provider"]
                if "model" in node:
                    node["model"] = target["model"]
                if "context_length" in node:
                    node["context_length"] = target["context"]
                count += 1
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(doc)
    if primary_local:
        primary["default"] = target["model"]
        primary["context_length"] = target["context"]
    return count


def retarget_pi(doc, settings, target_name: str) -> None:
    target = TARGETS[target_name]
    providers = doc.setdefault("providers", {})
    source = providers.get("glm-exl3")
    if not isinstance(source, dict):
        raise ValueError("missing Pi provider glm-exl3")
    selected = copy.deepcopy(source)
    selected["models"] = copy.deepcopy(source.get("models") or [])
    if not selected["models"]:
        raise ValueError("Pi glm-exl3 has no model entry")
    model = selected["models"][0]
    model["id"] = target["model"]
    model["name"] = target["pi_name"]
    model["contextWindow"] = target["context"]
    model["maxTokens"] = 32768
    if target_name == "dsv41":
        model["input"] = ["text"]
    providers[target["pi_provider"]] = selected
    if settings.get("defaultProvider") in LOCAL_PI_PROVIDERS:
        settings["defaultProvider"] = target["pi_provider"]
        settings["defaultModel"] = target["model"]


def plan(root: Path, target_name: str):
    if target_name not in TARGETS:
        raise ValueError("target must be glm or dsv41")
    root = Path(root)
    changes = {}
    reference_count = 0
    config_paths = [root / ".hermes/config.yaml", root / ".hermes/profiles/sandbox/config.yaml"]
    for path in config_paths:
        if not path.is_file():
            continue
        original = path.read_text()
        doc = load_yaml(original)
        reference_count += retarget_hermes(doc, target_name)
        rendered = dump_yaml(doc)
        if rendered != original:
            changes[path] = rendered

    pi_models = root / ".pi/agent/models.json"
    pi_settings = root / ".pi/agent/settings.json"
    if pi_models.is_file():
        original = pi_models.read_text()
        doc = json.loads(original)
        settings = json.loads(pi_settings.read_text()) if pi_settings.is_file() else {}
        original_settings = copy.deepcopy(settings)
        retarget_pi(doc, settings, target_name)
        rendered = json.dumps(doc, indent=2) + "\n"
        if rendered != original:
            changes[pi_models] = rendered
        if pi_settings.is_file() and settings != original_settings:
            changes[pi_settings] = json.dumps(settings, indent=2) + "\n"

    if reference_count == 0:
        raise ValueError("no Hermes local-provider references were found; refusing to claim client selection")
    return changes


def apply(changes, state_dir: Path) -> Path:
    backup_dir = Path(state_dir) / "client-backups" / (time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}")
    backup_dir.mkdir(parents=True, exist_ok=False)
    originals = {}
    written = []
    try:
        for index, (path, content) in enumerate(changes.items()):
            path = Path(path)
            backup = backup_dir / f"{index:02d}-{path.name}"
            shutil.copy2(path, backup)
            originals[path] = backup
            mode = path.stat().st_mode & 0o777
            fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.chmod(tmp_name, mode)
                os.replace(tmp_name, path)
                written.append(path)
                if path.read_text() != content:
                    raise OSError(f"read-back verification failed for {path}")
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
    except Exception:
        for path in reversed(written):
            shutil.copy2(originals[path], path)
        raise
    return backup_dir


HINDSIGHT_HOST = "sakkout@10.0.1.20"
HINDSIGHT_ENV = "/home/sakkout/.config/hindsight/server.env"
HINDSIGHT_SERVICE = "hindsight-api.service"


def render_hindsight(original: str, target_name: str) -> str:
    key = "HINDSIGHT_API_LLM_MODEL"
    value = TARGETS[target_name]["model"]
    found = False
    lines = []
    for line in original.splitlines(keepends=True):
        stripped = line.lstrip()
        prefix = "export " if stripped.startswith("export ") else ""
        body = stripped[len(prefix):]
        if body.startswith(key + "="):
            newline = "\n" if line.endswith("\n") else ""
            lines.append(prefix + key + "=" + value + newline)
            found = True
        else:
            lines.append(line)
    if not found:
        raise ValueError(f"{key} is missing from the Node 2 Hindsight environment")
    return "".join(lines)


def update_hindsight(target_name: str, apply_changes: bool) -> None:
    """Atomically retarget the relocated Node 2 service and verify its live process."""
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", HINDSIGHT_HOST]

    def remote(command: str, *, input_text=None, timeout=30):
        return subprocess.run(ssh + [command], input=input_text, text=True, capture_output=True,
                              timeout=timeout, check=True).stdout

    original = remote("cat " + HINDSIGHT_ENV)
    rendered = render_hindsight(original, target_name)
    expected = TARGETS[target_name]["model"]
    if rendered == original:
        print(f"Hindsight Node 2 already selects {expected}; no restart needed")
        return
    if not apply_changes:
        print(f"Dry-run: Hindsight Node 2 would select {expected} and restart {HINDSIGHT_SERVICE}")
        return

    stamp = time.strftime("%Y%m%dT%H%M%S")
    backup = HINDSIGHT_ENV + ".bak-" + stamp
    tmp = HINDSIGHT_ENV + ".llm-switch.tmp"
    remote(f"cp --preserve=mode,timestamps {HINDSIGHT_ENV} {backup}")

    def write_remote(content: str):
        remote(f"umask 077; cat > {tmp} && chmod 600 {tmp} && mv {tmp} {HINDSIGHT_ENV}",
               input_text=content, timeout=30)
        if remote("cat " + HINDSIGHT_ENV) != content:
            raise OSError("Hindsight Node 2 read-back verification failed")

    def restart_and_verify(expected_model: str):
        remote(f"systemctl --user restart {HINDSIGHT_SERVICE}", timeout=60)
        deadline = time.monotonic() + 45
        while True:
            try:
                remote("curl -fsS --max-time 3 http://127.0.0.1:18780/health >/dev/null", timeout=8)
                break
            except subprocess.CalledProcessError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        command = (
            f"pid=$(systemctl --user show {HINDSIGHT_SERVICE} -p MainPID --value) && "
            "python3 -c 'import os,sys; d=dict(x.split(\"=\",1) for x in "
            "open(\"/proc/\"+sys.argv[1]+\"/environ\",\"rb\").read().decode(errors=\"replace\").split(\"\\0\") "
            "if \"=\" in x); print(d.get(\"HINDSIGHT_API_LLM_MODEL\",\"\"))' \"$pid\""
        )
        actual = remote(command).strip()
        if actual != expected_model:
            raise OSError(f"Hindsight live model verification failed: {actual!r} != {expected_model!r}")

    write_remote(rendered)
    try:
        restart_and_verify(expected)
    except Exception:
        write_remote(original)
        restart_and_verify(TARGETS["glm"]["model"] if "GLM-5.3-Flash-EXL3" in original
                           else TARGETS["dsv41"]["model"])
        raise
    print(f"Hindsight Node 2 verified {expected}; backup: {HINDSIGHT_HOST}:{backup}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=sorted(TARGETS))
    parser.add_argument("--apply", action="store_true", help="write changes; default is a read-only dry-run")
    parser.add_argument("--root", type=Path, default=Path.home(), help=argparse.SUPPRESS)
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".local/state/llm-switch")
    args = parser.parse_args(argv)
    try:
        changes = plan(args.root, args.target)
        if not changes:
            print(f"Clients already select {args.target}; no file changes needed")
        elif args.apply:
            backup = apply(changes, args.state_dir)
            print(f"Updated {len(changes)} client file(s); backup: {backup}")
        else:
            print(f"Dry-run: {len(changes)} client file(s) would change")
            for path in changes:
                print(path)
        update_hindsight(args.target, args.apply)
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
