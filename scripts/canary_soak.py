#!/usr/bin/env python3
"""Two-worker mixed DeepSeek canary workload with two-node memory telemetry."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import checklib as C

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather of a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def memory(host=None):
    command = "free -k"
    if host:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, command],
                                text=True, capture_output=True, timeout=15, check=True)
        text = result.stdout
    else:
        text = subprocess.check_output(["free", "-k"], text=True)
    rows = {line.split(":", 1)[0]: line.split()[1:] for line in text.splitlines() if ":" in line}
    mem = rows["Mem"]
    swap = rows["Swap"]
    return {"available_gib": round(int(mem[5]) / 1048576, 3),
            "swap_used_gib": round(int(swap[1]) / 1048576, 3)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=int, default=1800)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--sample-seconds", type=int, default=30)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    C._api_key_file = os.environ.get("API_KEY_FILE")
    started = time.time()
    started_mono = time.monotonic()
    deadline = started_mono + args.duration
    lock = threading.Lock()
    sequence = 0
    stop = threading.Event()
    requests = []
    telemetry = []
    errors = []

    def worker(worker_id):
        nonlocal sequence
        server = C.Server("http://127.0.0.1:8888", timeout=300, seed=1000 + worker_id)
        while not stop.is_set() and time.monotonic() < deadline:
            with lock:
                i = sequence
                sequence += 1
            tokens = 8000 if i % 5 == 0 else 2000
            text = C.words(C.words_for(tokens), 50000 + i) + f"\nRequest {i}: summarize the prose in one sentence."
            messages = [{"role": "user", "content": text}]
            body = server.body(messages, 96)
            if i % 17 == 0:
                messages = [{"role": "user", "content": f"What is the weather in city{i % 7}? Use the tool."}]
                body = server.body(messages, 256)
                body["tools"] = TOOLS
            t0 = time.monotonic()
            reply = server.post_json("/v1/chat/completions", body, timeout=300)
            elapsed = time.monotonic() - t0
            row = {"i": i, "worker": worker_id, "seconds": round(elapsed, 3),
                   "tool": i % 17 == 0, "prompt_tokens": (reply.get("usage") or {}).get("prompt_tokens"),
                   "completion_tokens": (reply.get("usage") or {}).get("completion_tokens")}
            ok = isinstance(reply.get("choices"), list) and bool(reply["choices"]) and not reply.get("error")
            if ok and row["tool"]:
                calls = reply["choices"][0].get("message", {}).get("tool_calls") or []
                ok = bool(calls) and calls[0].get("function", {}).get("name") == "get_weather"
            elif ok:
                ok = bool(reply["choices"][0].get("message", {}).get("content", "").strip())
            row["ok"] = ok
            if not ok:
                row["error"] = str(reply.get("error") or reply)[:300]
            with lock:
                requests.append(row)
                if not ok:
                    errors.append(row)
                    if len(errors) >= 3:
                        stop.set()
            if len(requests) % 20 == 0:
                with lock:
                    print(f"requests={len(requests)} errors={len(errors)} elapsed={int(time.monotonic()-started_mono)}s", flush=True)

    def sampler():
        while not stop.is_set() and time.monotonic() < deadline:
            row = {"time": time.time(), "head": memory(),
                   "worker": memory("sakkout@192.168.200.2")}
            with lock:
                telemetry.append(row)
            print(f"memory head={row['head']['available_gib']} GiB worker={row['worker']['available_gib']} GiB "
                  f"swap_head={row['head']['swap_used_gib']} GiB", flush=True)
            time.sleep(args.sample_seconds)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers + 1) as pool:
        futures = [pool.submit(worker, i) for i in range(args.workers)] + [pool.submit(sampler)]
        for future in futures:
            future.result()
    stop.set()
    latencies = [r["seconds"] for r in requests if r["ok"]]
    result = {
        "status": "pass" if not errors and C.Server("http://127.0.0.1:8888").alive() else "fail",
        "started": started, "duration_s": round(time.time() - started, 3),
        "workers": args.workers, "requests": len(requests), "errors": errors,
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in requests),
        "latency_s": {"p50": percentile(latencies, .5), "p95": percentile(latencies, .95),
                      "p99": percentile(latencies, .99), "max": max(latencies) if latencies else None},
        "min_available_gib": {node: min((t[node]["available_gib"] for t in telemetry), default=None)
                              for node in ("head", "worker")},
        "max_swap_used_gib": {node: max((t[node]["swap_used_gib"] for t in telemetry), default=None)
                              for node in ("head", "worker")},
        "telemetry": telemetry, "request_rows": requests,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("telemetry", "request_rows")}, indent=2))
    if result["status"] != "pass":
        raise SystemExit("soak FAILED")


if __name__ == "__main__":
    main()
