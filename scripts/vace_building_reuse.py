"""Read-only source-identity validation for completed Pilot8 four-file results."""

import io
from pathlib import Path

import numpy as np
from PIL import Image

import buildingfiller_pilot8 as delivery
import vace_p01 as baseline
from vace_p01_81f_results import FINAL_FILENAMES, REPRESENTATIVE_FRAMES, validate_final_results
from vace_p01_results import HEADER_HEIGHT, SIZE
from vace_p01_reuse import check_no_symlink_components, file_hashes


def require_source_crop(image, column, expected, pilot, description):
    left = column * SIZE[0]
    actual = np.asarray(image.crop((left, HEADER_HEIGHT, left + SIZE[0], HEADER_HEIGHT + SIZE[1])))
    baseline.require(np.array_equal(actual, np.asarray(expected)),
                     f"{pilot}: completed result source identity mismatch: {description}")


def validate_completed_sample(record, destination):
    identity = (record["output_number"], record["pilot"], record["source_id"], record["kind"], record["tier"])
    baseline.require(identity in delivery.SAMPLE_MAPPING, "Completed sample identity is not in the fixed Pilot8 mapping")
    destination = check_no_symlink_components(Path(destination))
    baseline.require(destination.name == record["output_number"],
                     f"{record['pilot']}: completed output directory number does not match its source identity")
    validated = validate_final_results(destination)
    original_hashes = file_hashes(destination)
    selected = delivery.read_sample(record, include_gt=False)
    baseline.require(len(selected) == 81, f"{record['pilot']}: completed source must have 81 frames")
    with Image.open(destination / FINAL_FILENAMES[2]) as overview:
        overview.load()
        for column, frame_index in enumerate(REPRESENTATIVE_FRAMES):
            with Image.open(io.BytesIO(selected[frame_index]["original_bytes"]["damaged"])) as damaged:
                require_source_crop(overview, column, damaged, record["pilot"], f"Damaged B{frame_index:03}")
    with Image.open(destination / FINAL_FILENAMES[3]) as four_columns:
        four_columns.load()
        with Image.open(io.BytesIO(selected[20]["original_bytes"]["damaged"])) as damaged:
            require_source_crop(four_columns, 0, damaged, record["pilot"], "Damaged B020 four-column panel")
        require_source_crop(four_columns, 1, selected[20]["model_mask"], record["pilot"], "ModelMask B020")
        require_source_crop(four_columns, 3, delivery.read_gt_B020(record), record["pilot"], "GT B020")
    check_no_symlink_components(destination)
    baseline.require(sorted(path.name for path in destination.iterdir()) == sorted(FINAL_FILENAMES)
                     and all((destination / name).is_file() and not (destination / name).is_symlink()
                             for name in FINAL_FILENAMES),
                     f"{record['pilot']}: completed result files changed during validation")
    baseline.require(file_hashes(destination) == original_hashes,
                     f"{record['pilot']}: completed result hashes changed during validation")
    validated.update(pilot=record["pilot"], source_id=record["source_id"], output_number=record["output_number"],
                     destination=str(destination), archive_sha256=record["archive_sha256"],
                     files_sha256=original_hashes, source_identity_verified=True,
                     historical_parameters_independently_recoverable=False)
    return validated
