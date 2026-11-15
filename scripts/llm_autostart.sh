#!/bin/bash
# Start the saved GLM or DeepSeek selection ONCE after either node reboots.
# Derived from the owner-requested glm53-autostart.sh; the timer, handled marker and state directory stay unchanged.
# Rules: one attempt per pair of boot IDs; never restart after a crash, failed start or manual stop; both nodes must
# pass memory/pressure/GPU gates; port/firewall/key guards must be active; gates get GIVE_UP_MIN minutes to recover.
# Dry run: DRY_RUN=1 ~/.local/bin/glm53-autostart.sh (checks/logs only; stops/starts/records nothing).
set -u
SELECTED_FILE="${LLM_SELECTED_FILE:-$HOME/.local/state/llm-switch/selected}"
TARGET=$(cat "$SELECTED_FILE" 2>/dev/null || echo glm)
case "$TARGET" in
  glm)
    RECIPE="$HOME/glm53-1104"
    CONTAINER="glm53-spark"
    DEFAULT_MIN=96
    ;;
  dsv41)
    RECIPE="$HOME/dsv41-1017-tf"
    CONTAINER="dsv41-1017-exl3-tf"
    DEFAULT_MIN=112
    ;;
  *) echo "Invalid saved model selection: $TARGET" >&2; exit 1 ;;
esac
WORKER="sakkout@192.168.200.2"
REQUIRE_KEY=1
KEYFILE="$HOME/.config/glm53/api-key"
GUARD_PORTS="29583 29584"
STATE="${GLM_AUTOSTART_STATE:-$HOME/.local/state/glm53-autostart}"
MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-$DEFAULT_MIN}"
MAX_PSI_FULL_AVG60="${MAX_PSI_FULL_AVG60:-1.00}"
GIVE_UP_MIN="${GIVE_UP_MIN:-45}"
DRY="${DRY_RUN:-0}"

mkdir -p "$STATE" && chmod 700 "$STATE"
exec 9>"$STATE/lock"
flock -n 9 || exit 0
LOG="$STATE/autostart.log"
log()  { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
note() { [[ "$(cat "$STATE/last-note" 2>/dev/null)" == "$*" ]] && return 0; printf '%s' "$*" > "$STATE/last-note"; log "$*"; }
(( DRY )) && log "DRY_RUN=1: checks only; saved selection is $TARGET"

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "$WORKER")
on_worker() { "${SSH[@]}" "$@"; }

here_boot=$(cat /proc/sys/kernel/random/boot_id)
there_boot=$(on_worker cat /proc/sys/kernel/random/boot_id 2>/dev/null) || { note "waiting: Spark not reachable over the cable"; exit 0; }
pair="$here_boot $there_boot"
handled=$(cat "$STATE/handled" 2>/dev/null || true)
[[ "$handled" == "$pair" ]] && exit 0
record() { (( DRY )) || echo "$pair" > "$STATE/handled"; }

# An ssh failure is never read as stopped. Only the selected recipe's rank pair is expected after a clean switch/stop.
st_here=$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo absent)
st_there=$(on_worker "docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null || echo absent") ||
  { note "waiting: Spark not reachable over the cable"; exit 0; }
if [[ "$st_here" == true && "$st_there" == true ]]; then
  if (( DRY )); then log "DRY_RUN: $TARGET already running on both nodes"; exit 0; fi
  if "$HOME/.local/bin/llm-use" "$TARGET" >>"$LOG" 2>&1; then
    record; note "$TARGET verified running on both nodes; boots $pair marked handled"
  else
    log "ERROR: $TARGET was running but failed model/auth/correctness verification; boots not marked handled"
  fi
  exit 0
fi
if [[ "$st_here" == true || "$st_there" == true ]]; then
  if (( DRY )); then log "DRY_RUN: would run ./stop.sh to clear a lone $TARGET rank (GB10=$st_here Spark=$st_there)"
  else log "a lone $TARGET rank is left after a reboot (GB10=$st_here Spark=$st_there): running ./stop.sh to clear it"
       (cd "$RECIPE" && ./stop.sh) >> "$LOG" 2>&1; log "stop.sh exit $?"; fi
  exit 0
fi

up_here=$(awk '{print int($1/60)}' /proc/uptime)
up_there=$(on_worker "awk '{print int(\$1/60)}' /proc/uptime" 2>/dev/null || echo 0)
since_boot=$(( up_here < up_there ? up_here : up_there ))
mem_gate() {
  local avail psi gpu
  avail=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)
  psi=$(awk '/^full/{split($3,a,"="); print a[2]}' /proc/pressure/memory)
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || { echo "nvidia-smi failed"; return 1; }
  (( avail >= MIN )) || { echo "MemAvailable ${avail} GiB < ${MIN}"; return 1; }
  awk -v p="$psi" -v m="$MAXPSI" 'BEGIN{exit !(p+0 <= m+0)}' || { echo "memory pressure full avg60=${psi} > ${MAXPSI}"; return 1; }
  [[ -z "$gpu" ]] || { echo "other GPU process(es) running: $(echo $gpu)"; return 1; }
  echo "ok ${avail} GiB"; return 0
}
fail=""
systemctl is-active --quiet glm-port-guard || fail+=" GB10 glm-port-guard not active;"
for p in $GUARD_PORTS; do grep -qw "$p" /etc/glm-port-guard.nft 2>/dev/null || fail+=" glm-port-guard does not cover port $p;"; done
[[ -s "$KEYFILE" && "$(stat -c %a "$KEYFILE")" == 600 ]] || fail+=" API key file missing or not mode 600;"
if [[ "$TARGET" == glm ]]; then
  [[ -s "$RECIPE/.env" && "$(stat -c %a "$RECIPE/.env")" == 600 ]] || fail+=" $RECIPE/.env missing or not mode 600;"
fi
systemctl is-active --quiet wifi-guard     || fail+=" GB10 wifi-guard not active;"
on_worker "systemctl is-active --quiet wifi-guard" 2>/dev/null || fail+=" Spark wifi-guard not active;"
docker info >/dev/null 2>&1                || fail+=" GB10 docker not ready;"
on_worker "docker info >/dev/null 2>&1"    || fail+=" Spark docker not ready;"
here_mem=$(MIN=$MIN_AVAIL_GIB MAXPSI=$MAX_PSI_FULL_AVG60 mem_gate) || fail+=" GB10 ${here_mem};"
there_mem=$(on_worker "$(declare -f mem_gate); MIN=$MIN_AVAIL_GIB MAXPSI=$MAX_PSI_FULL_AVG60 mem_gate" 2>/dev/null) || fail+=" Spark ${there_mem:-unreachable};"
ss -ltn "sport = :8888" 2>/dev/null | grep -q LISTEN && fail+=" port 8888 already in use;"
if [[ -n "$fail" ]]; then
  if (( since_boot >= GIVE_UP_MIN )); then
    record; log "GAVE UP for boots $pair (${since_boot} min after the later boot):$fail $TARGET not started; run llm-use $TARGET by hand once fixed"
  else
    note "waiting (${since_boot} min after boot):$fail"
  fi
  exit 0
fi

if (( DRY )); then log "DRY_RUN: gates pass (GB10 $here_mem, Spark $there_mem); would start $TARGET with preparation disabled for boots $pair"; exit 0; fi
record                                               # recorded BEFORE starting: a failed start is not retried
out="$STATE/start-$(date +%Y%m%d-%H%M%S).log"
log "gates pass (GB10 $here_mem, Spark $there_mem; guards active); starting $TARGET for boots $pair, output in $out"
if [[ "$TARGET" == dsv41 ]]; then
  (cd "$RECIPE" && PREPARE=0 ./start.sh) > "$out" 2>&1
else
  (cd "$RECIPE" && ./start.sh) > "$out" 2>&1
fi
rc=$?
log "start.sh exit $rc ($(grep -aoE 'serving [^ ]+ at [^ ]+|context: [0-9]+|loaded in [0-9.]+s|is now LIVE' "$out" | tr '\n' ' '))"
# Idempotent llm-use performs the exact model-id, auth and correctness checks and leaves client files on the saved target.
if "$HOME/.local/bin/llm-use" "$TARGET" >>"$LOG" 2>&1; then
  log "$TARGET boot start verified"
else
  log "ERROR: $TARGET boot verification failed; stopping its pair (this boot remains handled)"
  (cd "$RECIPE" && ./stop.sh) >>"$LOG" 2>&1; log "stop.sh exit $?"
fi
exit 0
