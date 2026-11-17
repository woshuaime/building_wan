"""Create one immutable run description; do not cache, benchmark or train here."""
import argparse
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

from common import ROOT, PYTHON, CACHE, PROMPT, now, write_json, sha, identity, runtime_env


def configure(run):
    if run.exists() and any(run.iterdir()):
        raise ValueError("Choose a new empty run; launch/resume existing runs with run.py")
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoints").mkdir(exist_ok=True)
    (run / "checks").mkdir(exist_ok=True)
    (run / "source").mkdir(exist_ok=True)
    data = ROOT / "data/pointcloud_vace/pointcloud_video_pairs_v2"
    manifest = data / "manifest.csv"
    model_source = json.loads((ROOT / "configs/vace_p01_model_sources_v1.json").read_text())
    paths = dict(checkpoint=str(ROOT / model_source["checkpoint"]["default_local_path"]),
                 **{key: str(ROOT / value["path"]) for key, value in model_source["reused_components"].items()})
    for key, path in paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing existing {key}: {path}")
    source = list(Path(__file__).parent.glob("*.py")) + [ROOT / name for name in (
        "scripts/vace_p01.py", "scripts/vace_p01_environment.py", "scripts/vace_roof_training.py",
        "code/DiffSynth-Studio/diffsynth/pipelines/wan_video.py",
        "code/DiffSynth-Studio/diffsynth/models/wan_video_dit.py",
        "code/DiffSynth-Studio/diffsynth/models/wan_video_vace.py",
        "code/DiffSynth-Studio/diffsynth/models/wan_video_vae.py",
        "code/DiffSynth-Studio/diffsynth/diffusion/base_pipeline.py",
        "code/DiffSynth-Studio/diffsynth/diffusion/flow_match.py",
        "code/DiffSynth-Studio/diffsynth/core/gradient/gradient_checkpoint.py")]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(source)}
    config = dict(created=now(), project_root=str(ROOT), python=PYTHON, seed=20261115,
                  paths=dict(data=str(data), cache=str(CACHE)), cache_root=str(CACHE),
                  manifest_sha256=sha(manifest), render_config_sha256=sha(data / "render_config.json"),
                  models=dict(repository=model_source["model_id"], revision=model_source["huggingface_revision"],
                              paths=paths, identities={k: identity(p) for k, p in paths.items() if k != "tokenizer"},
                              expected_sha256=dict(checkpoint=model_source["checkpoint"]["accepted_files"][0]["sha256"],
                                  **{k: v["sha256"] for k, v in model_source["reused_components"].items() if k != "tokenizer"}),
                              tokenizer_sha256=model_source["reused_components"]["tokenizer"]["files_sha256"]),
                  source_sha256=source_hashes,
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  git_status_at_prepare=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                  training=dict(train_buildings=10747, epochs=3, total_samples=32241, width=832, height=480,
                                frames=17, fps=16, batch_candidates=[1, 2], gradient_accumulation_steps=1,
                                learning_rate=1.e-4, final_learning_rate=1.e-5, warmup_sample_fraction=.02,
                                adam_betas=[.9, .999], adam_eps=1.e-8, weight_decay=0., gradient_clip_norm=1.,
                                scheduler_steps=1000, scheduler_shift=5., vace_scale=1.,
                                lora_rank=8, lora_alpha=8, lora_modules="15 VACE blocks self/cross attention q/k/v/o",
                                backbone_dtype="bfloat16", increment_optimizer_dtype="float32", num_workers=0,
                                loss="GT gray<239 dilated3 pixels; fractional4n+1/8x8 latent areas; foreground/background equal normalized MSE, per-sample scheduler weight",
                                tiled=True, tile_size=[30, 52], tile_stride=[15, 26]),
                  prompt=PROMPT, negative_prompt="", automatic_validation=False,
                  recovery=dict(first_update_checkpoint=True, checkpoint_updates=200, checkpoint_seconds=300,
                                automatic_oom_restarts=2, preserve=["latest", "previous", "epoch1", "epoch2", "epoch3"]),
                  references=["https://github.com/ali-vilab/VACE/blob/main/vace/models/wan/wan_vace.py",
                              "https://docs.pytorch.org/docs/2.11/checkpoint.html",
                              "https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html"])
    import hashlib
    config["config_id"] = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    write_json(run / "config.json", config)
    write_json(run / "source_identity.json", source_hashes)
    for path in source:
        destination = run / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    (run / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    for action in ("stop", "resume"):
        command = [PYTHON, str(Path(__file__).with_name("run.py")), action, str(run)]
        path = run / f"{action}.sh"
        path.write_text("#!/usr/bin/env bash\nexec " + shlex.join(command) + "\n", encoding="utf-8")
        path.chmod(0o755)
    write_json(run / "progress.json", dict(time=now(), stage="configured", status="not_started",
               samples_consumed=0, total_samples=32241, optimizer_updates=0, epochs=3,
               epoch=0, epoch_completed=0, epoch_buildings=10747, validation_started=False))
    print(json.dumps(dict(run=str(run), config_id=config["config_id"], automatic_validation=False), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    configure(args.run.resolve())
