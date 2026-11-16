#!/usr/bin/env python3
"""Cancel a cold 30K-token prefill, then prove the pair answers a fresh request."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import checklib as C


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--tokens", type=int, default=30000)
    p.add_argument("--cancel-after", type=float, default=2.0)
    args = p.parse_args()
    C._api_key_file = os.environ.get("API_KEY_FILE")
    server = C.Server("http://127.0.0.1:8888")
    prompt = C.words(C.words_for(args.tokens), 918273) + "\nWhen you finish reading, answer DONE."
    messages = [{"role": "user", "content": prompt}]
    started = time.monotonic()
    cancelled = server.stream(messages, max_tokens=128, cancel_after=args.cancel_after, keep_text=True)
    time.sleep(3)
    retry = server.post_json("/v1/chat/completions", server.body(
        [{"role": "user", "content": "What is 2+2? Answer with one digit only."}], 64), timeout=120)
    content = retry.get("choices", [{}])[0].get("message", {}).get("content", "")
    result = {
        "cancelled": cancelled,
        "cancel_wall_s": time.monotonic() - started,
        "retry_content": content,
        "retry_finish": retry.get("choices", [{}])[0].get("finish_reason"),
        "healthy_after": server.alive(),
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cancelled": cancelled.get("cancelled"), "cancel_error": cancelled.get("error"),
                      "retry_content": content, "retry_finish": result["retry_finish"],
                      "healthy_after": result["healthy_after"]}, indent=2))
    if not cancelled.get("cancelled") or cancelled.get("error") or "4" not in content or not result["healthy_after"]:
        raise SystemExit("cancellation/retry FAILED")
    print("cancellation/retry: PASS")


if __name__ == "__main__":
    main()
