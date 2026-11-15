#!/usr/bin/env python3
"""Transactional GLM/DeepSeek switching for the two-Spark cluster."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

MODELS = {
    "glm": {
        "recipe": Path.home() / "glm53-1104",
        "container": "glm53-spark",
        "model_id": "GLM-5.3-Flash-EXL3",
    },
    "dsv41": {
        "recipe": Path.home() / "dsv41-1017-tf",
        "container": "dsv41-1017-exl3-tf",
        "model_id": "DeepSeek-V4.1-Flash-1017",
    },
}
WORKER = "sakkout@192.168.200.2"
BASE_URL = "http://127.0.0.1:8888"
KEY_FILE = Path.home() / ".config/glm53/api-key"


class SwitchError(RuntimeError):
    pass


class SwitchCore:
    def __init__(self, state_dir: Path, driver):
        self.state_dir = Path(state_dir)
        self.driver = driver
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_file = (self.state_dir / "switch.lock").open("a+")

    def close(self) -> None:
        self.lock_file.close()

    def selected(self) -> str:
        path = self.state_dir / "selected"
        value = path.read_text().strip() if path.exists() else "glm"
        if value not in MODELS:
            raise SwitchError(f"Invalid saved model selection: {value!r}")
        return value

    def commit(self, model: str) -> None:
        tmp = self.state_dir / f"selected.tmp.{os.getpid()}"
        tmp.write_text(model + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.state_dir / "selected")

    def switch(self, target: str, force_restart: bool = False) -> None:
        if target not in MODELS:
            raise SwitchError(f"Unknown target {target!r}; choose glm or dsv41")
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = self.driver.current_model()
        if current not in (None, *MODELS):
            raise SwitchError(f"Refusing to switch from unexpected responder {current!r}")
        source = current or self.selected()
        was_running = current is not None

        if current == target and not force_restart:
            self.driver.verify(target)
            self.driver.update_clients(target)
            self.commit(target)
            return

        try:
            if current is not None:
                self.driver.drain(current)
                self.driver.stop(current)
            self.driver.assert_stopped()
            self.driver.start(target)
            self.driver.verify(target)
            self.driver.update_clients(target)
            self.commit(target)
            return
        except Exception as target_error:
            rollback_errors = []
            try:
                self.driver.stop(target)
                self.driver.assert_stopped()
                if was_running:
                    self.driver.start(source)
                    self.driver.verify(source)
                    self.driver.update_clients(source)
            except Exception as rollback_error:
                rollback_errors.append(str(rollback_error))
            if rollback_errors:
                raise SwitchError(
                    f"Target {target} failed: {target_error}; ROLLBACK TO {source} FAILED: "
                    + "; ".join(rollback_errors)
                ) from target_error
            outcome = f"rolled back to {source}" if was_running else "left stopped"
            raise SwitchError(f"Target {target} failed: {target_error}; {outcome}") from target_error

    def manual_stop(self) -> None:
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = self.driver.current_model()
        if current is None:
            self.driver.assert_stopped()
            return
        if current not in MODELS:
            raise SwitchError(f"Refusing to stop unexpected responder {current!r}")
        self.driver.drain(current)
        self.driver.stop(current)
        self.driver.assert_stopped()


class LocalDriver:
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.log_path = self.state_dir / "switch.log"
        self.api = os.environ.get("LLM_SWITCH_URL", BASE_URL).rstrip("/")
        self.worker = os.environ.get("LLM_SWITCH_WORKER", WORKER)
        self.key_file = Path(os.environ.get("LLM_SWITCH_KEY_FILE", KEY_FILE))

    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with self.log_path.open("a") as f:
            f.write(line + "\n")

    def run(self, args, *, cwd=None, env=None, timeout=300) -> None:
        shown = " ".join(str(x) for x in args)
        self.log(f"RUN {shown}")
        with self.log_path.open("a") as out:
            result = subprocess.run(
                [str(x) for x in args], cwd=cwd, env=env, stdout=out,
                stderr=subprocess.STDOUT, timeout=timeout, check=False,
            )
        if result.returncode:
            raise SwitchError(f"command exited {result.returncode}: {shown}; see {self.log_path}")

    def rank_state(self, model: str, node: str) -> str:
        container = MODELS[model]["container"]
        local = ["docker", "inspect", "-f", "{{.State.Running}}", container]
        if node == "worker":
            local = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", self.worker] + local
        p = subprocess.run(local, text=True, capture_output=True, timeout=20, check=False)
        if p.returncode:
            combined = (p.stdout + p.stderr).lower()
            if "no such object" in combined or "no such container" in combined:
                return "absent"
            return "unknown"
        value = p.stdout.strip()
        return "running" if value == "true" else "stopped" if value == "false" else "unknown"

    def all_rank_states(self):
        return {
            model: {node: self.rank_state(model, node) for node in ("head", "worker")}
            for model in MODELS
        }

    def assert_stopped(self) -> None:
        states = self.all_rank_states()
        bad = {
            f"{model}.{node}": state
            for model, nodes in states.items()
            for node, state in nodes.items() if state != "absent"
        }
        if bad:
            raise SwitchError(f"Ranks are not verifiably absent on both nodes: {bad}")

    def key(self) -> str:
        if not self.key_file.is_file() or (self.key_file.stat().st_mode & 0o777) != 0o600:
            raise SwitchError(f"API key file is missing or not mode 600: {self.key_file}")
        key = self.key_file.read_text().strip()
        if not key:
            raise SwitchError("API key file is empty")
        return key

    def request(self, path: str, *, payload=None, auth=True, timeout=20):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.api + path, data=data)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if auth:
            req.add_header("Authorization", "Bearer " + self.key())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode()
                return response.status, json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                parsed = json.loads(body) if body else {}
            except json.JSONDecodeError:
                parsed = {"raw": body[:500]}
            return e.code, parsed
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise SwitchError(f"API request {path} failed: {e}") from e

    def endpoint_model(self):
        try:
            status, health = self.request("/health", auth=False, timeout=5)
        except SwitchError:
            return None
        if status != 200 or health.get("status") not in (None, "ok"):
            return None
        auth_status, body = self.request("/v1/models", timeout=10)
        if auth_status != 200:
            raise SwitchError(f"Authenticated /v1/models returned HTTP {auth_status}")
        ids = [m.get("id") for m in body.get("data", []) if isinstance(m, dict)]
        if len(ids) != 1:
            raise SwitchError(f"Expected exactly one served model id, got {ids}")
        return ids[0]

    def current_model(self):
        endpoint = self.endpoint_model()
        states = self.all_rank_states()
        if endpoint is None:
            if all(state == "absent" for nodes in states.values() for state in nodes.values()):
                return None
            raise SwitchError(f"API is down but rank state is not cleanly stopped: {states}")
        actual = next((name for name, spec in MODELS.items() if spec["model_id"] == endpoint), endpoint)
        if actual not in MODELS:
            return actual
        if any(state != "running" for state in states[actual].values()):
            raise SwitchError(f"Endpoint reports {actual} but its ranks are not both running: {states[actual]}")
        other = next(name for name in MODELS if name != actual)
        if any(state != "absent" for state in states[other].values()):
            raise SwitchError(f"Both models have residual containers: {states}")
        return actual

    def connections(self):
        last = None
        for auth in (True, False):
            try:
                status, body = self.request("/health", auth=auth, timeout=5)
            except SwitchError as e:
                last = e
                continue
            if status != 200:
                continue
            live = body.get("live") if isinstance(body, dict) else None
            value = live.get("connections") if isinstance(live, dict) else body.get("requests_running")
            if isinstance(value, int) and value >= 0:
                return value
        if last:
            raise last
        return None

    def drain(self, model: str) -> None:
        timeout = int(os.environ.get("LLM_DRAIN_TIMEOUT_S", "600"))
        deadline = time.monotonic() + timeout
        while True:
            count = self.connections()
            if count is None:
                grace = int(os.environ.get("LLM_UNKNOWN_DRAIN_GRACE_S", "30"))
                self.log(
                    f"WARNING: {model} does not expose an active-request count; "
                    f"using the explicit {grace}s quiet-period fallback before stopping"
                )
                time.sleep(grace)
                return
            if count == 0:
                return
            if time.monotonic() >= deadline:
                raise SwitchError(f"{model} still has {count} active requests after {timeout}s")
            self.log(f"Waiting for {count} active request(s) to finish")
            time.sleep(5)

    def stop(self, model: str) -> None:
        self.run(["./stop.sh"], cwd=MODELS[model]["recipe"], timeout=180)

    def start(self, model: str) -> None:
        env = os.environ.copy()
        if model == "dsv41":
            env["PREPARE"] = "0"
        timeout = int(os.environ.get("LLM_START_TIMEOUT_S", "3000"))
        self.run(["./start.sh"], cwd=MODELS[model]["recipe"], env=env, timeout=timeout)

    def verify(self, model: str) -> None:
        states = self.all_rank_states()[model]
        if any(state != "running" for state in states.values()):
            raise SwitchError(f"{model} ranks failed verification: {states}")
        status, _ = self.request("/health", auth=False, timeout=10)
        if status != 200:
            raise SwitchError(f"/health returned HTTP {status}")
        unauth, _ = self.request("/v1/models", auth=False, timeout=10)
        if unauth != 401:
            raise SwitchError(f"Unauthenticated /v1/models returned HTTP {unauth}, expected 401")
        actual = self.endpoint_model()
        if actual != MODELS[model]["model_id"]:
            raise SwitchError(f"Wrong responder: {actual!r}, expected {MODELS[model]['model_id']!r}")
        payload = {
            "model": MODELS[model]["model_id"],
            "messages": [{"role": "user", "content": "What is 2+2? Answer with one digit only."}],
            "temperature": 0,
            "max_tokens": 128,
            "stream": False,
        }
        code, body = self.request("/v1/chat/completions", payload=payload, timeout=180)
        if code != 200:
            raise SwitchError(f"Correctness request returned HTTP {code}: {body}")
        try:
            choices = body.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("missing choices")
            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise ValueError("missing message")
            content = message.get("content") or ""
            if not isinstance(content, str):
                raise ValueError("non-text content")
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise SwitchError(f"Malformed correctness response: {body}") from e
        if "4" not in content:
            raise SwitchError(f"Correctness response did not contain the required digit: {message}")
        record = self.state_dir / "last-verification.json"
        tmp = record.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "model": model, "model_id": actual, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "auth_required": True, "correctness": "2+2 response contained 4",
        }, indent=2) + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, record)
        self.log(f"Verified {actual}: both ranks, health, authentication, and correctness")

    def update_clients(self, model: str) -> None:
        script = MODELS["dsv41"]["recipe"] / "scripts/llm_switch_clients.py"
        self.run([sys.executable, str(script), model, "--apply"], timeout=120)


def status(driver: LocalDriver) -> None:
    selected_path = driver.state_dir / "selected"
    selected = selected_path.read_text().strip() if selected_path.exists() else "glm"
    try:
        current = driver.current_model()
        connections = driver.connections() if current in MODELS else None
    except SwitchError as e:
        current = f"ambiguous: {e}"
        connections = None
    print(f"selected={selected}")
    print(f"actual={current or 'stopped'}")
    print(f"connections={connections if connections is not None else 'unknown'}")
    for model, nodes in driver.all_rank_states().items():
        print(f"{model}: head={nodes['head']} worker={nodes['worker']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["glm", "dsv41", "status", "stop", "restart"])
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".local/state/llm-switch")
    args = parser.parse_args(argv)
    driver = LocalDriver(args.state_dir)
    core = SwitchCore(args.state_dir, driver)
    try:
        if args.action == "status":
            status(driver)
        elif args.action == "stop":
            core.manual_stop()
            driver.log("Stopped; saved selection unchanged")
        elif args.action == "restart":
            core.switch(core.selected(), force_restart=True)
            driver.log(f"Restarted and selected {core.selected()}")
        else:
            core.switch(args.action)
            driver.log(f"Selected and verified {args.action}")
        return 0
    except BrokenPipeError:
        # A consumer such as grep -q may close the pipe after finding a status line.
        # Do not log a spurious error or emit a second traceback during shutdown.
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
        return 0
    except (SwitchError, subprocess.TimeoutExpired, OSError) as e:
        driver.log(f"ERROR: {e}")
        return 1
    finally:
        core.close()


if __name__ == "__main__":
    raise SystemExit(main())
