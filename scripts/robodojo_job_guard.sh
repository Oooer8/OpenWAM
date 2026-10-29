#!/usr/bin/env bash
# The scheduler must own this shell for its full lifetime (no exec of trainer).
openwam_job_record() {
    python - "$1" "${2:-0}" <<'PY'
import datetime, json, os, socket
from pathlib import Path
root = Path(os.environ['OPENWAM_JOB_STATUS_DIR'])
if root.is_dir():
    path = root / ('volcano_job_status_node_' + os.environ.get('NODE_RANK', '0') + '.json')
    value = dict(status=__import__('sys').argv[1], exit_code=int(__import__('sys').argv[2]),
                 stage=os.environ.get('OPENWAM_JOB_STAGE', 'bootstrap'),
                 hostname=socket.gethostname(), wrapper_pid=os.getppid(),
                 updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 failure_hold_seconds=int(os.environ['OPENWAM_FAILURE_HOLD_SECONDS']),
                 preflight_log=os.environ.get('OPENWAM_PREFLIGHT_LOG'),
                 preflight_progress=str(root / ('runtime_progress_node_' + os.environ.get('NODE_RANK', '0') + '.json')))
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
PY
}

openwam_job_exit() {
    local code=$1
    trap - EXIT
    if [[ ${DRY_RUN:-0} == 1 || ${OPENWAM_OPERATOR_STOP:-0} == 1 ]]; then return; fi
    set +e
    if ((code == 0)); then openwam_job_record completed 0; return; fi
    openwam_job_record failed_holding "$code"
    echo "FATAL: stage=${OPENWAM_JOB_STAGE:-bootstrap} exit=$code. Keeping this job alive for ${OPENWAM_FAILURE_HOLD_SECONDS}s for in-place diagnosis; training is NOT running." >&2
    echo 'Inspect this task console and volcano_job_status_node_*.json. Intentional task termination bypasses this hold.' >&2
    local retry="$OPENWAM_JOB_STATUS_DIR/volcano_retry_node_${NODE_RANK:-0}" requested
    echo "After repair, request retry IN THIS allocation by writing wrapper PID $$ to $retry" >&2
    local remaining=$OPENWAM_FAILURE_HOLD_SECONDS pause
    while true; do
        if [[ -f $retry && ! -L $retry && -n ${OPENWAM_JOB_ENTRY:-} ]]; then
            requested=''; read -r requested < "$retry" || true
            if [[ $requested == "$$" ]]; then
                rm -- "$retry"
                openwam_job_record retry_requested "$code"
                exec bash "$OPENWAM_JOB_ENTRY"
            fi
        fi
        ((remaining > 0)) || break
        pause=30; ((remaining < pause)) && pause=$remaining
        sleep "$pause"
        remaining=$((remaining-pause))
    done
    openwam_job_record failed_hold_expired "$code"
    exit "$code"
}

openwam_operator_stop() {
    export OPENWAM_OPERATOR_STOP=1
    if [[ -n ${OPENWAM_TRAIN_PID:-} ]]; then
        kill -TERM "$OPENWAM_TRAIN_PID" 2>/dev/null || true
        wait "$OPENWAM_TRAIN_PID" 2>/dev/null || true
    fi
    exit "$1"
}

openwam_wait_child() {
    local code
    if wait "$OPENWAM_TRAIN_PID"; then code=0; else code=$?; fi
    OPENWAM_TRAIN_PID=''
    return "$code"
}

export OPENWAM_FAILURE_HOLD_SECONDS="${OPENWAM_FAILURE_HOLD_SECONDS:-3600}"
if ! [[ $OPENWAM_FAILURE_HOLD_SECONDS =~ ^[0-9]+$ ]] || ((OPENWAM_FAILURE_HOLD_SECONDS > 86400)); then
    echo 'Invalid OPENWAM_FAILURE_HOLD_SECONDS; using3600.' >&2
    export OPENWAM_FAILURE_HOLD_SECONDS=3600
fi
export OPENWAM_JOB_STATUS_DIR="${OPENWAM_JOB_STATUS_DIR:-/mnt/vepfs01/output/tiecheng.guo/robodojo_runs/20261021_8x16_gc6_verified}"
trap 'openwam_job_exit $?' EXIT
trap 'openwam_operator_stop 143' TERM
trap 'openwam_operator_stop 130' INT
