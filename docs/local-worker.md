# Worker-local preparation without NFS

This fork adds a `NO_NFS=1` preparation path for the two-Spark deployment. It does not install or configure NFS, change firewall rules, or create Docker NFS volumes. The upstream NFS path remains the default when `NO_NFS` is unset.

## Layout

The head keeps the complete pinned checkpoint, official Engram snapshot, kernel kit, served-model view, and both generated rank-cache files. Keeping rank 1 on the head lets preparation verify the worker copy by SHA-256 and costs about another 106 GB of head storage beyond the standard NFS layout.

The worker receives only:

- checkpoint metadata and the runtime vision shard `model-00039-of-00039.safetensors`, dereferenced from the HF snapshot;
- the served-model view, preserving relative symlinks into that partial snapshot;
- the kernel kit and token map;
- both official Engram shards plus their config and index in `WORKER_ENGRAM_DIR`;
- rank 1's generated weight file in `WORKER_CACHE_DIR`.

`NFS_VOLUME` is an absolute worker host path in this mode. Despite the historical variable name, `start.sh` bind-mounts it directly at `/hf:ro`; no NFS server is involved.

## Configure and prepare

`scripts/local.sh` contains:

```bash
NO_NFS=1
NFS_SERVER=
NFS_VOLUME=/home/sakkout/.cache/dsv41-tensorfold/worker-hf
HEAD_NFS_SERVER=
```

Run preparation only with every model rank stopped:

```bash
VERIFY_SHA=1 ./scripts/prepare.sh
```

`NO_NFS=1` requires `VERIFY_SHA=1`. Preparation hashes the head's Engram shards, copies them over ConnectX with resumable `rsync`, and verifies their exact sizes and SHA-256 values on the worker. It also compares rank 1's generated weight-file hash and the runtime vision-shard hash across the two nodes.

After preparation, normal starts can skip all setup:

```bash
PREPARE=0 ./start.sh
```

The prepared marker includes the worker-local path. The normal start-time `worker_files_check` reads the worker's config, kit manifest, and token map through the same host-path mount and image used by the rank.

## Reruns and updates

Preparation is resumable. Large Engram copies use `--partial --inplace` and are always content-verified on rerun. Rank 1 is copied to a temporary worker file, hashed, and atomically renamed; a stale final-named file is repaired rather than trusted. The small runtime tree gets checksum synchronization. Dedicated worker directories may be synchronized with `--delete` (and `--delete-excluded` inside the dedicated snapshot), so do not place unrelated files there.

The no-NFS branch is intentionally limited to stock `ABLIT=0` weights. It requires `VERIFY_SHA=1` and absolute worker paths; these are set in `scripts/local.sh` and validated before image builds or downloads. All three worker destinations must be on one filesystem. The disk check always reserves a complete temporary rank copy, even when the existing final file proves valid and no transfer is needed, so stale-rank repair cannot exceed the approved budget.

A checkpoint, kit, template, token-map, or image change must be followed by a fresh preparation. Do not edit the worker's served files independently. A changed rank-cache identity produces a new filename; retain old rank files until the new pair has passed qualification and rollback testing.

## Verification

- `bash -n scripts/prepare.sh scripts/prepare-local-worker.sh start.sh`
- `DRY_RUN=1 PREPARE=0 ./start.sh` must show `local HF at ...` and a host-path `-v ...:/hf:ro` for rank 1.
- The original `scripts/test_scripts.py`, `test_checkpoint_pin.py`, and `test_systemd_unit.py` suites must remain green.
- At first real launch, verify NCCL/RDMA, memory headroom, model id, authentication, and correctness separately; a dry-run proves only configuration discovery.
