#!/usr/bin/env python3
"""Verify an HF snapshot's exact file set, sizes, Git blob ids and LFS SHA-256 values."""
import argparse
import hashlib
import json
from pathlib import Path


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def file_hash(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(manifest_path: Path, snapshot: Path):
    remote = json.loads(manifest_path.read_text())
    expected = {entry["path"]: entry for entry in remote if "size" in entry}
    local_files = {str(p.relative_to(snapshot)) for p in snapshot.rglob("*") if p.is_file()}
    if set(expected) != local_files:
        raise ValueError(f"file-set mismatch: missing={sorted(set(expected)-local_files)} extra={sorted(local_files-set(expected))}")
    results = []
    total = 0
    for name, entry in sorted(expected.items()):
        path = snapshot / name
        size = path.stat().st_size
        if size != entry["size"]:
            raise ValueError(f"size mismatch for {name}: {size} != {entry['size']}")
        if "lfs" in entry:
            algorithm = "sha256"
            expected_hash = entry["lfs"]["sha256"]
        else:
            algorithm = "git-blob-sha1"
            expected_hash = entry["blob_id"]
        if algorithm == "sha256":
            actual_hash = file_hash(path, "sha256")
        else:
            data = path.read_bytes()
            actual_hash = git_blob(data)
        if actual_hash != expected_hash:
            raise ValueError(f"hash mismatch for {name}: {actual_hash} != {expected_hash}")
        results.append({"path": name, "size": size, "algorithm": algorithm, "hash": actual_hash})
        total += size
        print(f"OK {name} {size} {algorithm} {actual_hash}", flush=True)
    return {"status": "ok", "snapshot": str(snapshot), "files": len(results), "bytes": total, "results": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("snapshot", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = verify(args.manifest, args.snapshot)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"VERIFIED files={result['files']} bytes={result['bytes']}")


if __name__ == "__main__":
    main()
