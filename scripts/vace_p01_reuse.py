"""Validate and non-destructively reuse the completed P01 four-file result."""

import io
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import vace_p01 as baseline
import vace_p01_81f as experiment
from vace_p01_81f_results import FINAL_FILENAMES, REPRESENTATIVE_FRAMES, validate_final_results
from vace_p01_results import HEADER_HEIGHT, SIZE


DEFAULT_OUTPUT_ROOT = baseline.ROOT / "outputs/vace"
P01_RESULT_PATTERN = "P01_465_81f_*"


def check_no_symlink_components(path):
    path = Path(path).absolute()
    for component in (path, *path.parents):
        baseline.require(not component.is_symlink(), f"Refuse a symlink path component: {component}")
    return path


def file_hashes(directory):
    return {name: baseline.sha256(Path(directory) / name) for name in FINAL_FILENAMES}


def require_crop(image, column, row, expected, description):
    left = column * SIZE[0]
    top = row * (SIZE[1] + HEADER_HEIGHT) + HEADER_HEIGHT
    actual = np.asarray(image.crop((left, top, left + SIZE[0], top + SIZE[1])))
    baseline.require(np.array_equal(actual, np.asarray(expected)),
                     f"P01 result identity mismatch: {description}")


def validate_p01_results(source, project_root=baseline.ROOT):
    source = check_no_symlink_components(source)
    validated = validate_final_results(source)
    config = experiment.experiment_config()
    selected = experiment.memory_inputs(config, project_root)
    with Image.open(source / FINAL_FILENAMES[2]) as overview:
        overview.load()
        for column, frame_index in enumerate(REPRESENTATIVE_FRAMES):
            with Image.open(io.BytesIO(selected[frame_index]["original_bytes"]["damaged"])) as damaged:
                require_crop(overview, column, 0, damaged, f"Damaged B{frame_index:03}")
    with Image.open(source / FINAL_FILENAMES[3]) as four_columns:
        four_columns.load()
        with Image.open(io.BytesIO(selected[20]["original_bytes"]["damaged"])) as damaged:
            require_crop(four_columns, 0, 0, damaged, "Damaged B020 four-column panel")
        require_crop(four_columns, 1, 0, selected[20]["model_mask"], "ModelMask B020")
        require_crop(four_columns, 3, 0, experiment.read_gt_B020(config, project_root), "GT B020")
    validated.update(source_id=465, source=str(source), files_sha256=file_hashes(source),
                     source_identity_verified=True,
                     historical_parameters_independently_recoverable=False)
    return validated


def find_verified_p01(output_root=DEFAULT_OUTPUT_ROOT, source=None, project_root=baseline.ROOT):
    if source is not None:
        source = check_no_symlink_components(source)
        validate_p01_results(source, project_root)
        return source
    output_root = check_no_symlink_components(output_root)
    baseline.require(output_root.is_dir(), f"P01 output root does not exist: {output_root}")
    valid = []
    errors = []
    for candidate in sorted(output_root.glob(P01_RESULT_PATTERN)):
        try:
            validate_p01_results(candidate, project_root)
            valid.append(candidate)
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"{candidate.name}: {error}")
    baseline.require(len(valid) <= 1,
                     "Multiple valid P01 81-frame results; explicitly choose one, never guess the newest: "
                     + ", ".join(str(path) for path in valid))
    baseline.require(bool(valid), "No valid completed P01 81-frame result found; do not rerun P01. "
                     + "; ".join(errors))
    return valid[0]


def find_p01_result(root=DEFAULT_OUTPUT_ROOT, source=None, project_root=baseline.ROOT):
    return find_verified_p01(root, source=source, project_root=project_root)


def validate_p01_result(source, project_root=baseline.ROOT):
    return validate_p01_results(source, project_root=project_root)


def copy_verified_p01(source, target, project_root=baseline.ROOT):
    source = check_no_symlink_components(source)
    target = check_no_symlink_components(target)
    baseline.require(source == target or (not target.is_relative_to(source)
                                          and not source.is_relative_to(target)),
                     "P01 source and target must be independent directories")
    validated_source = validate_p01_results(source, project_root)
    expected_hashes = validated_source["files_sha256"]
    if source == target:
        return dict(status="REUSED_EXISTING", source=str(source), target=str(target), files=list(FINAL_FILENAMES))
    if target.exists():
        validated_target = validate_p01_results(target, project_root)
        baseline.require(validated_target["files_sha256"] == expected_hashes,
                         "Existing test/01 differs from the verified P01 source; refuse to overwrite")
        return dict(status="REUSED_EXISTING", source=str(source), target=str(target), files=list(FINAL_FILENAMES))
    target.mkdir(parents=True, exist_ok=False)
    for name in FINAL_FILENAMES:
        original = source / name
        baseline.require(original.is_file() and not original.is_symlink(), "P01 source file changed during copy")
        with original.open("rb") as reader, (target / name).open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 ** 2)
    copied = validate_p01_results(target, project_root)
    baseline.require(copied["files_sha256"] == expected_hashes, "Copied P01 file hashes do not match the source")
    baseline.require(file_hashes(source) == expected_hashes, "P01 source changed during copy; preserve both directories")
    return dict(status="COPIED", source=str(source), target=str(target), files=list(FINAL_FILENAMES))
