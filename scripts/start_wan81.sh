#!/bin/bash

set -euo pipefail

ROOT=/home/share/CHUANJUN/building_wan
SUBMIT_SCRIPT="$ROOT/scripts/submit_wan81.sh"
LOG_DIR="$ROOT/logs/slurm"
JOB_NAME=building_wan

cd "$ROOT"
mkdir -p "$LOG_DIR"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [existing-job-id]" >&2
    exit 2
fi

if [[ $# -eq 1 ]]; then
    job_id=$1
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
        echo "Job ID must contain digits only: $job_id" >&2
        exit 2
    fi
    echo "Attaching to existing Slurm job: $job_id"
else
    job_id=$(sbatch --parsable "$SUBMIT_SCRIPT")
    job_id=${job_id%%;*}
    echo "Submitted Slurm job: $job_id"
fi

stdout_log="$LOG_DIR/$JOB_NAME-$job_id.out"
stderr_log="$LOG_DIR/$JOB_NAME-$job_id.err"

echo "Standard output: $stdout_log"
echo "Standard error:  $stderr_log"
echo "Press Ctrl+C to stop displaying logs; the Slurm job will continue."

tail_pid=""

stop_tail() {
    if [[ -n "$tail_pid" ]] && kill -0 "$tail_pid" 2>/dev/null; then
        kill "$tail_pid" 2>/dev/null || true
        wait "$tail_pid" 2>/dev/null || true
    fi
}

stop_display() {
    echo
    echo "Stopped local log display. Slurm job $job_id was not cancelled."
    echo "Resume with: bash $ROOT/scripts/start_wan81.sh $job_id"
    stop_tail
    exit 130
}

trap stop_display INT TERM

while [[ ! -e "$stdout_log" && ! -e "$stderr_log" ]]; do
    if job_state=$(squeue -h -j "$job_id" -o '%T' 2>&1); then
        if [[ -z "$job_state" ]]; then
            echo "[$(date +%H:%M:%S)] Job is not in squeue yet; waiting for logs..."
        else
            echo "[$(date +%H:%M:%S)] Job $job_id state: $job_state; waiting for logs..."
        fi
    else
        echo "[$(date +%H:%M:%S)] Slurm status query failed; continuing to wait: $job_state" >&2
    fi
    sleep 5
done

if [[ -e "$stdout_log" || -e "$stderr_log" ]]; then
    tail -c 16384 -F "$stdout_log" "$stderr_log" &
    tail_pid=$!
else
    echo "No log files found for Slurm job $job_id." >&2
    exit 3
fi

state=""
exit_code=""
while true; do
    if queue_state=$(squeue -h -j "$job_id" -o '%T' 2>&1); then
        if [[ -n "$queue_state" ]]; then
            sleep 5
            continue
        fi
    else
        echo "[$(date +%H:%M:%S)] Slurm status query failed; log display will continue: $queue_state" >&2
        sleep 10
        continue
    fi

    if accounting=$(sacct -X -n -j "$job_id" --format=State,ExitCode -P 2>&1); then
        accounting=$(printf '%s\n' "$accounting" | head -n 1)
        state=${accounting%%|*}
        exit_code=${accounting#*|}
        case "$state" in
            COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*|DEADLINE*|REVOKED*)
                break
                ;;
        esac
    else
        echo "[$(date +%H:%M:%S)] Slurm accounting query failed; log display will continue: $accounting" >&2
    fi
    sleep 10
done

sleep 2
stop_tail
trap - INT TERM

echo
echo "Slurm job $job_id finished: state=${state:-unknown}, exit_code=${exit_code:-unknown}"
echo "Logs remain at:"
echo "  $stdout_log"
echo "  $stderr_log"

if [[ "$state" != COMPLETED* ]]; then
    exit 1
fi
