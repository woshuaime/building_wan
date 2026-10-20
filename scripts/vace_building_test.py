"""Reuse completed P01 and sequentially run only P02-P08, retaining four files each."""

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import buildingfiller_pilot8 as delivery
import vace_p01 as baseline
import vace_p01_81f as orbit
from vace_p01_81f_console import ConsoleState, visible_worker
from vace_p01_81f_results import FINAL_FILENAMES, validate_final_results
from vace_p01_reuse import check_no_symlink_components, copy_verified_p01, find_verified_p01, validate_p01_results
from vace_building_reuse import validate_completed_sample


ROOT = baseline.ROOT
OUTPUT_ROOT = ROOT / "outputs/vace/test"
QUEUE = delivery.SAMPLE_MAPPING[1:]
TEMP_PREFIX = "building_wan_vace_remaining_"


def sample_config(record):
    config = orbit.experiment_config()
    config["source"].update(archive=record["archive"], archive_sha256=record["archive_sha256"],
                            package_id=delivery.PACKAGE_ID, sample_relative_root=record["relative_root"],
                            source_id=record["source_id"], stable_id=record["stable_id"])
    return config


def queue_records(package):
    records = {record["pilot"]: record for record in package["records"]}
    baseline.require(len(records) == len(package["records"]) == 8, "Expected eight unique verified Pilot identities")
    queue = []
    for number, pilot, source, kind, tier in QUEUE:
        record = records[pilot]
        baseline.require((record["output_number"], record["source_id"], record["kind"], record["tier"])
                         == (number, source, kind, tier), "Remaining sample queue identity mismatch")
        queue.append(record)
    baseline.require(all(record["pilot"] != "P01" for record in queue), "P01 is never a GPU queue item")
    return queue


def check_targets(output_root=None, resume=False):
    output_root = check_no_symlink_components(OUTPUT_ROOT if output_root is None else output_root)
    allowed = {entry[0] for entry in delivery.SAMPLE_MAPPING}
    if output_root.exists():
        baseline.require(output_root.is_dir(), "test must be a directory")
        baseline.require({path.name for path in output_root.iterdir()} <= allowed,
                         "test contains unrecognized historical files/directories; nothing will be deleted")
    for number, pilot, source, kind, tier in QUEUE:
        target = output_root / number
        if target.exists() or target.is_symlink():
            baseline.require(resume,
                             f"{pilot} source{source}: destination already exists; refusing overwrite or automatic rerun: {target}; "
                             "use explicit --resume only after verifying completed results")
            check_no_symlink_components(target)
            baseline.require(target.is_dir(), f"Resume destination is not a directory: {target}")
    orbit.check_output_volume()
    return output_root


def plan_samples(package, resume=False):
    records = queue_records(package)
    output_root = check_targets(resume=resume)
    pending, completed = [], {}
    for record in records:
        target = output_root / record["output_number"]
        if target.exists():
            baseline.require(resume and not pending,
                             f"Non-contiguous completed results at {record['pilot']}; refusing to guess or overwrite")
            completed[record["pilot"]] = validate_completed_sample(record, target)["files_sha256"]
        else:
            pending.append(record)
    return pending, completed


def p01_source(output_root=None):
    output_root = OUTPUT_ROOT if output_root is None else output_root
    existing = output_root / "01"
    if existing.exists() or existing.is_symlink():
        validate_p01_results(existing)
        return existing
    return find_verified_p01(ROOT / "outputs/vace")


def preflight(state, resume=False):
    from vace_p01_environment import check_environment

    state.stage("delivery_validation", "Full ZIP SHA/CRC, 6077 payload hashes, sample identities and all original PNGs")
    package = delivery.read_delivery()
    records = queue_records(package)
    pending, completed = plan_samples(package, resume=resume)
    original = p01_source()
    print(f"PASS: P01 completed 81F result verified for reuse only: {original}", flush=True)
    if resume:
        print(f"PASS: explicit resume; verified completed prefix={list(completed)}; "
              f"GPU queue only={[record['pilot'] for record in pending]}", flush=True)
    for record in records:
        print(f"PASS: {record['output_number']} -> {record['pilot']} source{record['source_id']} "
              f"{record['kind']} {record['tier']}; 81 GT/Damaged/Exact; "
              f"original black Exact frames={record['empty_exact_frame_indices']}", flush=True)
    print("PASS: effective human QA accepted; training_ready=false, generation_verified=false, generation_evidence_verified=true; visual inference only", flush=True)
    config = orbit.experiment_config()
    print(f"Unified VACE prompt: {config['inference']['prompt']}", flush=True)
    print(f"Unified negative prompt: {config['inference']['negative_prompt']}", flush=True)
    print("Parameters: original VACE-1.3B; no LoRA; 81x512x512; 16FPS; seed20261018; steps30; CFG5; full BF16", flush=True)
    state.stage("model_environment", "Checking original model hashes, CUDA, memory, other tasks and local dependencies")
    models = baseline.inspect_models(config, verify=True)
    baseline.require(models["ready_for_gpu"], "Original VACE components are missing or not verified")
    environment = check_environment()
    baseline.require(environment["cuda"]["free_bytes"] >= 24 * 1024 ** 3, "Less than 24 GiB available VRAM; no automatic fallback")
    import torch

    baseline.require(torch.backends.cuda.flash_sdp_enabled() or torch.backends.cuda.mem_efficient_sdp_enabled(),
                     "Efficient 81-frame SDPA backend is disabled; refusing math-only inference")
    print(f"PASS: official model hashes; {environment['cuda']['device_name']}; "
          f"free VRAM={environment['cuda']['free_bytes'] / 1024 ** 3:.2f} GiB; no other GPU tasks", flush=True)
    state.stage("cpu_input_contracts", "Checking each remaining sample's own 81-frame adapter and loader in memory")
    for record in records:
        selected = delivery.read_sample(record, include_gt=False)
        baseline.check_loader_contract([item["conditioning"] for item in selected],
                                       [item["model_mask"] for item in selected], num_frames=81)
        print(f"PASS: {record['pilot']} actual CPU input contract, mask distribution and B080; no GT RGB in model inputs", flush=True)
        del selected
    baseline.check_old_training()
    state.stage("preflight", "All seven samples and P01 reuse passed; no model loaded or GPU inference started", completed=True)
    return package, original


def inherited_status(state):
    lock_fd = int(os.environ["WAN_VACE81_LOCK_FD"])
    baseline.require(os.fstat(lock_fd).st_ino == (ROOT / "logs/.wan_a6000_gpu.lock").stat().st_ino,
                     "Missing inherited GPU launcher lock")
    status_fd = int(os.environ["WAN_VACE81_STATUS_FD"])

    def send(event):
        content = (json.dumps(event, ensure_ascii=False) + "\n").encode()
        baseline.require(len(content) < 4096 and os.write(status_fd, content) == len(content), "Anonymous runtime event failed")

    state.event_sink = send
    return send


def gate_worker(state, resume=False):
    send = inherited_status(state)
    package, original = preflight(state, resume=resume)
    pending, completed = plan_samples(package, resume=resume)
    send(dict(preflight_complete=True, archive_sha256=package["archive_sha256"], p01_source=str(original),
              completed_hashes=completed, pending_pilots=[record["pilot"] for record in pending]))
    return 0


def sample_worker(temporary, pilot, state):
    send = inherited_status(state)
    baseline.require(temporary.is_dir() and not temporary.is_symlink()
                     and temporary.parent == Path("/tmp") and temporary.name.startswith(TEMP_PREFIX),
                     "Sample worker requires its private system temporary directory")
    record = json.loads(os.environ["WAN_VACE_BATCH_RECORD"])
    baseline.require(pilot != "P01" and any((record["output_number"], pilot, record["source_id"], record["kind"], record["tier"]) == entry
                                          for entry in QUEUE) and record["pilot"] == pilot,
                     "Only the fixed P02-P08 GPU queue is authorized")
    try:
        state.stage("sample_preflight", f"Rechecking {pilot} source{record['source_id']} without GT conditioning")
        config = sample_config(record)
        selected = delivery.read_sample(record, include_gt=False)
        baseline.check_loader_contract([item["conditioning"] for item in selected],
                                       [item["model_mask"] for item in selected], num_frames=81)
        models = baseline.inspect_models(config, verify=True)
        baseline.require(models["ready_for_gpu"], "Sample model components are not verified")
        from vace_p01_environment import check_environment

        environment = check_environment()
        baseline.require(environment["cuda"]["free_bytes"] >= 24 * 1024 ** 3, "Insufficient available VRAM; stopping batch")
        import torch

        baseline.require(torch.backends.cuda.flash_sdp_enabled() or torch.backends.cuda.mem_efficient_sdp_enabled(),
                         "Efficient 81-frame SDPA backend became unavailable")
        orbit.check_output_volume()
        baseline.check_old_training()
        peaks = orbit.execute_sample(config, selected, models, state, temporary / "deliverables",
                                     lambda: delivery.read_gt_B020(record))
        send(dict(worker_complete=True, pilot=pilot, source_id=record["source_id"], archive_sha256=record["archive_sha256"],
                  **peaks, stage=state.last_stage, last_completed_stage=state.last_completed_stage))
        return 0
    except BaseException as error:
        send(dict(worker_error=f"{type(error).__name__}: {str(error)[:400]}", stage=state.last_stage,
                  last_completed_stage=state.last_completed_stage))
        raise


def worker_environment(lock_fd):
    environment = os.environ.copy()
    environment.update(DIFFSYNTH_SKIP_DOWNLOAD="True", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONDONTWRITEBYTECODE="1", WAN_VACE81_LOCK_FD=str(lock_fd))
    return environment


def supervise(arguments, environment, lock_fd, state, expected):
    events = {}

    def receive(event):
        events.update(event)
        state.last_stage = event.get("stage", state.last_stage)
        state.last_completed_stage = event.get("last_completed_stage", state.last_completed_stage)

    command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), *arguments]
    returncode, interrupted = visible_worker(command, environment, lock_fd, receive)
    if returncode != 0 or interrupted or not events.get(expected):
        code = 128 - returncode if returncode < 0 else returncode or (130 if interrupted else 1)
        raise orbit.WorkerFailure(code, f"worker returncode={returncode}, interrupted={interrupted}; "
                                  f"{events.get('worker_error', 'Missing valid completion event')}")
    return events


def summary(statuses, success, error=None):
    print("\nVACE BUILDING TEST: " + ("SUCCESS" if success else "FAILED"), flush=True)
    for entry in delivery.SAMPLE_MAPPING:
        print(f"{entry[1]}: {statuses[entry[1]]}", flush=True)
    if error:
        print(f"Reason: {error}", flush=True)
    label = "FINAL OUTPUT" if OUTPUT_ROOT.exists() else "OUTPUT TARGET (not created)"
    print(f"\n{label}:\n{OUTPUT_ROOT.resolve()}\n", flush=True)
    print("01–08: each successful sample contains exactly four validated files.", flush=True)
    print("\n".join(FINAL_FILENAMES), flush=True)
    print("先查看各目录的03_补前补后总览.png，再看04_四栏验证.png和两个视频。", flush=True)


def run_batch(state, resume=False):
    statuses = {entry[1]: "NOT RUN" for entry in delivery.SAMPLE_MAPPING}
    statuses["P01"] = "REUSE NOT YET VERIFIED - NEVER QUEUED"
    current = None
    durations = []
    try:
        check_targets(resume=resume)
        package = delivery.read_delivery(verify_images=False)
        records, completed = plan_samples(package, resume=resume)
        with orbit.cleanup_signals(), baseline.gpu_lock() as lock_stream:
            baseline.check_old_training()
            environment = worker_environment(lock_stream.fileno())
            state.stage("batch_preflight", "All-sample read-only gate in a separate process; parent never initializes CUDA")
            gate_arguments = ["--_gate", "--resume"] if resume else ["--_gate"]
            gate = supervise(gate_arguments, environment, lock_stream.fileno(), state, "preflight_complete")
            baseline.require(gate.get("archive_sha256") == package["archive_sha256"], "Delivery changed between batch preflight processes")
            if resume:
                baseline.require(gate.get("completed_hashes") == completed
                                 and gate.get("pending_pilots") == [record["pilot"] for record in records],
                                 "Completed results or resume queue changed between preflight processes")
            original = Path(gate["p01_source"])
            copy_verified_p01(original, OUTPUT_ROOT / "01")
            statuses["P01"] = "REUSED - 81F ALREADY COMPLETED"
            for record in queue_records(package):
                pilot = record["pilot"]
                if pilot in completed:
                    current = pilot
                    baseline.require(validate_completed_sample(record, OUTPUT_ROOT / record["output_number"])["files_sha256"]
                                     == completed[pilot], f"Completed {pilot} changed after preflight; stopping without overwrite")
                    statuses[pilot] = "REUSED - VERIFIED 81F ALREADY COMPLETED"
                    print(f"{pilot}: REUSED; existing four files remain untouched; no GPU inference", flush=True)
            current = None
            for index, record in enumerate(records, 1):
                sample_started = time.monotonic()
                current = record["pilot"]
                target = OUTPUT_ROOT / record["output_number"]
                baseline.require(not target.exists() and not target.is_symlink(), f"Refusing existing destination: {target}")
                print(f"\nRunning sample {current} ({index}/{len(records)} queued; overall {len(completed) + index}/7): "
                      f"output {record['output_number']}/08 source{record['source_id']}\n", flush=True)
                with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX, dir="/tmp") as name:
                    temporary = Path(name)
                    record_environment = dict(environment, WAN_VACE_BATCH_RECORD=json.dumps(record, ensure_ascii=True))
                    baseline.require(len(record_environment["WAN_VACE_BATCH_RECORD"].encode()) < 100000,
                                     "Sample descriptor exceeds the safe in-memory environment limit")
                    events = supervise(["--_worker", str(temporary), "--sample", current], record_environment,
                                       lock_stream.fileno(), state, "worker_complete")
                    baseline.require(events.get("pilot") == current and events.get("source_id") == record["source_id"]
                                     and events.get("archive_sha256") == record["archive_sha256"], "Worker output/sample identity mismatch")
                    allocated, reserved = events.get("peak_allocated_bytes"), events.get("peak_reserved_bytes")
                    baseline.require(isinstance(allocated, int) and isinstance(reserved, int) and 0 < allocated <= reserved,
                                     "Missing valid measured GPU memory peak")
                    check_no_symlink_components(target)
                    state.stage("publishing", f"Publishing four validated files for {current} to {target}")
                    orbit.publish_results(temporary / "deliverables", target)
                statuses[current] = "SUCCESS"
                durations.append(time.monotonic() - sample_started)
                remaining = (len(records) - index) * sum(durations) / len(durations)
                print(f"{current}: SUCCESS | completed {len(completed) + index}/7 | peak {allocated / 1024 ** 3:.2f} GiB allocated / "
                      f"{reserved / 1024 ** 3:.2f} GiB reserved | sample elapsed {durations[-1]:.1f}s | "
                      f"total elapsed {state.elapsed:.1f}s | batch ETA ~{remaining:.1f}s (measured completed samples)", flush=True)
                current = None
            baseline.require({path.name for path in OUTPUT_ROOT.iterdir()} == {entry[0] for entry in delivery.SAMPLE_MAPPING},
                             "Final test root must contain exactly directories 01 through 08")
            for number, pilot, source, kind, tier in delivery.SAMPLE_MAPPING:
                current = pilot
                validate_final_results(OUTPUT_ROOT / number)
            current = None
        summary(statuses, True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if current:
            statuses[current] = "REUSE VALIDATION FAILED" if current == "P01" else "FAILED"
        code = error.returncode if isinstance(error, orbit.WorkerFailure) else 130 if isinstance(error, KeyboardInterrupt) else 1
        summary(statuses, False, error)
        print(f"Stopped stage: {state.last_stage}\nExit code: {code}\nNo automatic retries or continuation.", file=sys.stderr, flush=True)
        return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Read-only seven-sample and P01 reuse checks; no output writes")
    mode.add_argument("--prepare", action="store_true", help="Full read-only checks, then copy/reuse P01 only; no GPU inference")
    mode.add_argument("--run", action="store_true", help="User authorization for only P02-P08, sequentially")
    mode.add_argument("--_gate", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true",
                        help="Explicitly validate and reuse only the contiguous completed prefix; never overwrite")
    parser.add_argument("--sample", choices=[entry[1] for entry in QUEUE], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    state = ConsoleState(total_steps=30)
    try:
        if args._gate:
            return gate_worker(state, resume=args.resume)
        if args._worker is not None:
            baseline.require(args.sample is not None, "Internal worker requires one remaining sample")
            return sample_worker(args._worker, args.sample, state)
        baseline.require(args.sample is None, "A single-sample CLI override is not allowed")
        if args.run:
            return run_batch(state, resume=args.resume)
        package, original = preflight(state, resume=args.resume)
        if args.prepare:
            result = copy_verified_p01(original, OUTPUT_ROOT / "01")
            print(f"P01: REUSED - 81F ALREADY COMPLETED ({result['status']})\n{OUTPUT_ROOT / '01'}", flush=True)
        if args.resume:
            print("PASS: explicit resume ready; only the listed missing suffix can run; completed results remain unchanged.", flush=True)
        else:
            print("PASS: P02-P08 ready; no new GPU inference, downloads, input copies, logs or reports.", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        code = error.returncode if isinstance(error, orbit.WorkerFailure) else 130 if isinstance(error, KeyboardInterrupt) else 1
        print(f"VACE BUILDING TEST: FAILED\nStopped stage: {state.last_stage}\nReason: {error}\nExit code: {code}", file=sys.stderr, flush=True)
        if args._worker is not None or args._gate:
            if state.event_sink is not None:
                try:
                    state.event_sink(dict(worker_error=f"{type(error).__name__}: {str(error)[:400]}", stage=state.last_stage,
                                          last_completed_stage=state.last_completed_stage))
                except (OSError, ValueError) as status_error:
                    print(f"Runtime status pipe unavailable: {status_error}", file=sys.stderr, flush=True)
            traceback.print_exc()
        return code


if __name__ == "__main__":
    sys.exit(main())
