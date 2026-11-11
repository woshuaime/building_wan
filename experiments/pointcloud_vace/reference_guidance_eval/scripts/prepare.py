"""Prepare one physical 17-frame circular clip and one fixed draft reference on CPU."""
import argparse
import numpy as np
from PIL import Image
from runtime import *


def prepare(run, start):
    from geometry import geometry_bundle, grouped_occupancy
    if run.exists():
        raise ValueError("Choose a new run directory")
    indices = circular_indices(start)
    center = indices[8]
    video = decode(CASE / "inputs/vace_video.mp4")
    mask = decode(CASE / "inputs/vace_video_mask.mp4")
    if len(video) != 81 or video.shape != mask.shape:
        raise ValueError("Expected the existing aligned 81-frame source")
    video, mask = video[indices].copy(), mask[indices].copy()
    if not np.isin(mask, [0, 255]).all():
        raise ValueError("Model mask must remain binary")
    observed = decode(CASE / "reference/damaged.mp4")[indices].copy()
    white = mask[..., 0] == 255
    if not white.any() or not (video[white] == 127).all():
        raise ValueError("White editing region must use the existing RGB127 input")
    draft = decode(OLD_PREDICTION)[center].copy()
    reference = np.where(white[8, ..., None], draft, observed[8])
    points, colors, cameras, scale, identity = observed_case()
    geometry = geometry_bundle(points, cameras, scale, indices)
    gate = grouped_occupancy(white[:, None].astype(np.float32), 16).reshape(1, -1, 1)
    for directory in ("cache", "reference", "logs", "comparisons/keyframes", "group_A", "group_B", "group_C"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run / "cache/clip.npz", video=video, mask=mask, observed=observed)
    np.savez_compressed(run / "cache/geometry.npz", **geometry)
    for filename, pixels in (("observed.png", observed[8]), ("draft.png", draft), ("reference.png", reference)):
        Image.fromarray(pixels).save(run / "reference" / filename)
    source = read(ROOT / "configs/vace_p01_model_sources_v1.json")
    models = dict(checkpoint=str(ROOT / source["checkpoint"]["default_local_path"]))
    models.update({key: str(ROOT / source["reused_components"][key]["path"])
                   for key in ("text_encoder", "vae", "tokenizer")})
    weights = {key: str(TRAIN_RUN / "checkpoints" / f"{key}_final.safetensors") for key in ("adapter", "lora")}
    for path in (*models.values(), *weights.values()):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    prompt = (CASE / "inputs/prompt.txt").read_text(encoding="utf-8")
    config = dict(created_at=now(), experiment="reference_guidance_eval_v2", building="wall_13697", case_dir=str(CASE),
        role="development_diagnostic_not_blind_test", model="Wan2.1-VACE-1.3B", models=models, weights=weights,
        checkpoint_updates=3000, lora_rank=8, lora_alpha=8, lora_scale=1,
        start=start, frame_indices=indices, center_frame=center, source_unique_camera_positions=80,
        source_reference_frames=list(range(0, 80, 10)), geometry_identity=identity,
        reference=dict(draft_video=str(OLD_PREDICTION), draft_original_frame=center, extra_draft_gpu_calls=0,
                       B="reference/observed.png", C="reference/reference.png",
                       composite="mask white=draft; mask black=unfilled damaged observation"),
        inference=dict(prompt=prompt, negative_prompt="", seed=20261018, height=480, width=832, num_frames=17,
                       fps=16, num_inference_steps=30, cfg_scale=5, cfg_merge=False, sigma_shift=5,
                       vace_scale=1, tiled=True, tile_size=[30, 52], tile_stride=[15, 26],
                       torch_dtype="bfloat16", rand_device="cpu"),
        tasks=[dict(group=group, seed=seed) for seed in (20261018, 20261019) for group in ("B", "C", "A")],
        inference_budget=6, training_updates=0, memory_mode="full", reference_interface="vace_reference_image",
        initial_noise="native sampler; identical B/C shapes and seeds; A/B video noise not forced identical",
        target_geometry_shape=[1, 7800, 64], reference_geometry_shape=[1, 9360, 64],
        gate_shape=list(gate.shape), mask_pixels_per_frame=white.sum((1, 2)).tolist(),
        same_pose_endpoint_metric=False, gt_mesh_hidden_conditioning=False, raw_output_compositing=False,
        handoff="first formal reference-conditioned sampling step; manual review and report deferred by user")
    write(run / "config.json", config)
    event(run, "prepared", frame_indices=indices, center_frame=center, formal_inference_calls=0, total_tasks=6,
          observed_points=len(points), geometry_frames=len(geometry["target_C"]), expected_gate_shape=list(gate.shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--start", type=int, default=78)
    args = parser.parse_args()
    prepare(args.run.resolve(), args.start)
