#!/usr/bin/env bash
# Sourced by prepare.sh after rank 0 is ready. Build a worker-local runtime tree without NFS.
# The full checkpoint stays on the head. The worker gets only its documented runtime reads, its Engram tables and rank 1.
set -euo pipefail

[[ "${NO_NFS:-0}" == 1 ]] || die "prepare-local-worker.sh requires NO_NFS=1"
[[ "$ABLIT" == 0 ]] || die "NO_NFS=1 currently supports stock weights only (ABLIT=0)"
[[ "${VERIFY_SHA:-0}" == 1 ]] || die "NO_NFS=1 requires VERIFY_SHA=1 so the head's Engram tables are hashed first"
[[ "$NFS_VOLUME" == /* && "$NFS_VOLUME" != / && "$NFS_VOLUME" != "$HF_CACHE" ]] ||
  die "NO_NFS=1 needs NFS_VOLUME to be a dedicated absolute worker-local directory, not '$NFS_VOLUME'"
[[ "$WORKER_ENGRAM_DIR" == /* && "$WORKER_CACHE_DIR" == /* ]] ||
  die "NO_NFS=1 needs absolute WORKER_ENGRAM_DIR and WORKER_CACHE_DIR paths"
command -v rsync >/dev/null || die "NO_NFS=1 needs rsync on the head"
worker 1 'command -v rsync >/dev/null' || die "NO_NFS=1 needs rsync on $(worker_host 1)"

whost=$(worker_host 1)
MODEL_REL="hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)"
VISION_SHARD=model-00039-of-00039.safetensors
wwant=$(in_hf --entrypoint bash "$IMAGE" -c "set -o pipefail; tf-dsv41-rank-cache $SNAP_C /tmp/x 1 2 --index-only | sha256sum | cut -c1-20")
[[ "$wwant" =~ ^[0-9a-f]{20}$ ]] || die "rank 1's index identity command failed or returned an invalid identity"
wrank="$DATA_DIR/rank-cache/rank1of2-$wwant.bin"
wrank_remote="$WORKER_CACHE_DIR/rank1of2-$wwant.bin"

log "Worker: local runtime tree at $NFS_VOLUME (no NFS, no full checkpoint copy)"
worker 1 "mkdir -p '$NFS_VOLUME/$MODEL_REL' '$NFS_VOLUME/$KIT_SUB/kit' '$NFS_VOLUME/$DATA_SUB/model' '$WORKER_ENGRAM_DIR' '$WORKER_CACHE_DIR'"
runtime_dev=$(worker 1 "stat -c %d '$NFS_VOLUME'")
engram_dev=$(worker 1 "stat -c %d '$WORKER_ENGRAM_DIR'")
cache_dev=$(worker 1 "stat -c %d '$WORKER_CACHE_DIR'")
[[ "$runtime_dev" == "$engram_dev" && "$runtime_dev" == "$cache_dev" ]] ||
  die "NO_NFS=1 requires NFS_VOLUME, WORKER_ENGRAM_DIR and WORKER_CACHE_DIR on one filesystem so its disk budget is valid"

# Budget before any large worker write. Existing files reduce the Engram requirement; reserve space for the runtime
# tree, a new rank file and 50 GiB of slack. The head separately budgets its additional rank-1 copy below.
need=0
for f in "${shards[@]}"; do need=$(( need + $(stat -L -c %s "$ESNAP/$f") )); done
have=0
for f in "${shards[@]}"; do
  size=$(worker 1 "stat -L -c %s '$WORKER_ENGRAM_DIR/$f' 2>/dev/null || echo 0")
  [[ "$size" =~ ^[0-9]+$ ]] && have=$(( have + size ))
done
# Always budget a complete temporary rank copy: a stale final-named file must coexist with its replacement until
# the hash-verified atomic rename, even when a valid existing rank would not need a transfer.
worker_need=$(( need - have + 20 * 1024 ** 3 + 50 * 1024 ** 3 + 120 * 1024 ** 3 ))
worker_free=$(worker 1 "df -B1 --output=avail '$WORKER_CACHE_DIR' | tail -1")
(( worker_free > worker_need )) ||
  die "$(worker_host 1) has $(( worker_free / 1024 ** 3 )) GiB free; no-NFS preparation needs $(( worker_need / 1024 ** 3 )) GiB including reserves"
if [[ ! -f "$wrank" ]]; then
  head_free=$(df -B1 --output=avail "$DATA_DIR/rank-cache" | tail -1)
  (( head_free > 130 * 1024 ** 3 )) ||
    die "the head has $(( head_free / 1024 ** 3 )) GiB free; rank 1 needs a 130 GiB generation budget"
fi

# Snapshot files are symlinks into HF's blob store. Dereference only the files the worker actually reads;
# the index still names every shard, but the engine's documented worker runtime reads only the vision shard.
rsync -aL --delete-excluded --prune-empty-dirs \
  --include='*/' --include="$VISION_SHARD" --exclude='model-*.safetensors' --include='*' \
  "$SNAP/" "$whost:$NFS_VOLUME/$MODEL_REL/"
# The kit and token map are regular files. The served view keeps relative symlinks into the worker's partial snapshot.
rsync -aLc --delete --exclude='.git' "$KIT_DIR/kit/" "$whost:$NFS_VOLUME/$KIT_SUB/kit/"
rsync -aLc "$DATA_DIR/dsv41_token_map.json" "$whost:$NFS_VOLUME/$DATA_SUB/dsv41_token_map.json"
rsync -ac --delete "$DATA_DIR/model/" "$whost:$NFS_VOLUME/$DATA_SUB/model/"
# A checksum pass on the snapshot makes reruns content-sensitive and removes stale excluded shards.
rsync -aLc --delete-excluded --prune-empty-dirs \
  --include='*/' --include="$VISION_SHARD" --exclude='model-*.safetensors' --include='*' \
  "$SNAP/" "$whost:$NFS_VOLUME/$MODEL_REL/"

log "Worker: Engram tables from the head over ConnectX (local disk at runtime)"
rsync -aL --partial --inplace --info=stats2 \
  "$ESNAP/model.safetensors.index.json" "$ESNAP/config.json" \
  "${shards[@]/#/$ESNAP/}" "$whost:$WORKER_ENGRAM_DIR/"
for f in model.safetensors.index.json config.json "${shards[@]}"; do
  head_sha=$(sha256sum "$ESNAP/$f" | cut -d' ' -f1)
  remote_sha=$(worker 1 "sha256sum '$WORKER_ENGRAM_DIR/$f' | cut -d' ' -f1")
  [[ "$head_sha" == "$remote_sha" ]] || die "$f differs between the head and $(worker_host 1)"
  if [[ -n "${want_sha[$f]:-}" ]]; then
    [[ "$remote_sha" == "${want_sha[$f]}" ]] || die "$f on $(worker_host 1) does not match the official SHA-256"
    remote_size=$(worker 1 "stat -L -c %s '$WORKER_ENGRAM_DIR/$f'")
    [[ "$remote_size" == "${want_size[$f]}" ]] || die "$f on $(worker_host 1) has the wrong size"
  fi
done
log "Worker: Engram config, index, exact shard sizes and SHA-256 values verified"

# Rank 1 is derived from the full checkpoint on the head. Copy to a temporary worker name, hash it, then rename atomically.
if [[ ! -f "$wrank" ]]; then
  log "Head: writing rank 1's weight file from the checkpoint (~106 GB)"
  in_hf --entrypoint tf-dsv41-rank-cache "$IMAGE" "$SNAP_C" "/hf/$DATA_SUB/rank-cache" 1 2 >/dev/null ||
    die "writing rank 1's weight file on the head failed"
fi
[[ -f "$wrank" ]] || die "rank-cache tool did not produce $wrank"
head_rank_sha=$(sha256sum "$wrank" | cut -d' ' -f1)
  wrank_part="$WORKER_CACHE_DIR/.rank1of2-$wwant.bin.part"
  worker_rank_sha=$(worker 1 "sha256sum '$wrank_remote' 2>/dev/null | cut -d' ' -f1 || true")
if [[ "$head_rank_sha" != "$worker_rank_sha" ]]; then
  log "Worker: rank 1's weight file over ConnectX (resumable temporary file)"
  rsync -a --partial --inplace --info=stats2 "$wrank" "$whost:$wrank_part"
  part_sha=$(worker 1 "sha256sum '$wrank_part' | cut -d' ' -f1")
  [[ "$part_sha" == "$head_rank_sha" ]] || die "rank 1's temporary worker copy failed its SHA-256 check"
  worker 1 "mv -f '$wrank_part' '$wrank_remote'"
fi
worker_rank_sha=$(worker 1 "sha256sum '$wrank_remote' | cut -d' ' -f1")
[[ "$head_rank_sha" == "$worker_rank_sha" ]] || die "rank 1's weight file differs after the worker copy"
worker 1 "rm -f -- '$wrank_part'"
log "Worker: rank 1 weight-file SHA-256 matches the head"

# Runtime preflight uses the same host-path mount and image as start.sh, then compares actual content beyond its readability checks.
worker_files_check 1
vision_head=$(sha256sum "$SNAP/$VISION_SHARD" | cut -d' ' -f1)
vision_worker=$(worker 1 "sha256sum '$NFS_VOLUME/$MODEL_REL/$VISION_SHARD' | cut -d' ' -f1")
[[ "$vision_head" == "$vision_worker" ]] || die "the worker's vision shard differs from the head's"
kit_head=$(cd "$KIT_DIR/kit" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
kit_worker=$(worker 1 "cd '$NFS_VOLUME/$KIT_SUB/kit' && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum")
[[ "$kit_head" == "$kit_worker" ]] || die "the worker's kernel kit differs from the head's"
for rel in "$DATA_SUB/dsv41_token_map.json" "$DATA_SUB/model/chat_template.jinja"; do
  head_sha=$(sha256sum "$HF_CACHE/$rel" | cut -d' ' -f1)
  remote_sha=$(worker 1 "sha256sum '$NFS_VOLUME/$rel' | cut -d' ' -f1")
  [[ "$head_sha" == "$remote_sha" ]] || die "worker file differs from the head: $rel"
done
log "Worker: config, kit, token map, served template and vision shard content verified; no NFS required"
