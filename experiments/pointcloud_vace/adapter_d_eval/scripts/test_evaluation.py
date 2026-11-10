"""CPU safeguards for observed-only inference, CFG, replay and startup handoff."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from common import CASES, TRAIN_RUN, sha, observed_points, verify_frozen
from binding import FixedGeometryModelFn


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_withheld_original_ids_are_removed_before_geometry(self):
        case = CASES / "eave_3230"
        record = json.loads((case / "case.json").read_text())
        points, colors, cameras, scale, audit = observed_points(case)
        self.assertEqual(len(points), record["cloud_size"] - record["count"])
        self.assertTrue(set(record["hidden_original_point_ids"]).isdisjoint(audit["original_point_ids_used"]))
        self.assertEqual(len(cameras), 81)
        self.assertTrue(np.allclose(cameras[0]["position"], cameras[-1]["position"]))
        self.assertFalse(audit["mesh_used"])
        self.assertFalse(audit["target_video_decoded"])

    def test_both_cfg_branches_receive_same_geometry_and_projection(self):
        tokens, mask, projection = object(), object(), object()
        model = FixedGeometryModelFn(tokens, mask, projection)
        with patch("integration.model_fn_wan_video", return_value=7) as fn:
            self.assertEqual(model(context="positive"), 7)
            self.assertEqual(model(context="negative"), 7)
        self.assertEqual(model.calls, 2)
        for call in fn.call_args_list:
            self.assertIs(call.kwargs["structure_tokens"], tokens)
            self.assertIs(call.kwargs["structure_mask"], mask)
            self.assertIs(call.kwargs["structure_projection"], projection)

    def test_config_and_weight_changes_are_rejected(self):
        import hashlib
        with tempfile.TemporaryDirectory() as t:
            run = Path(t)
            weight = run / "increment"
            weight.write_bytes(b"original")
            config = dict(frozen_files={str(weight): sha(weight)}, method="D")
            config["config_id"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            (run / "config.json").write_text(json.dumps(config))
            self.assertEqual(verify_frozen(run)["method"], "D")
            weight.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Frozen source/input/weight changed"):
                verify_frozen(run)
            config["method"] = "wrong model"
            (run / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                verify_frozen(run)

    def test_reference_decoding_cannot_precede_raw_save(self):
        import postprocess
        with tempfile.TemporaryDirectory() as t:
            run = Path(t)
            case = dict(name="test", output_dir=str(run), case_dir=str(run))
            with patch.object(postprocess, "verify_frozen", return_value={"cases": [case]}), patch.object(postprocess, "decode") as decode:
                with self.assertRaisesRegex(ValueError, "before raw predictions"):
                    postprocess.postprocess(run, "test")
                decode.assert_not_called()

    def test_loaded_adapter_handles_full_81_frame_observed_geometry(self):
        from geometry import geometry_bundle
        from adapter import MultiViewStructureAdapter
        from safetensors.torch import load_file
        points, _, cameras, scale, _ = observed_points(CASES / "eave_3230")
        bundle = geometry_bundle(points, cameras, scale, list(range(81)))
        g = {k: torch.from_numpy(np.asarray(v).copy())[None] for k, v in bundle.items()}
        module = MultiViewStructureAdapter()
        module.load_state_dict(load_file(str(TRAIN_RUN / "checkpoints/adapter_final.safetensors")), strict=True)
        with torch.no_grad():
            tokens = module(g)
        self.assertEqual(tuple(tokens.shape), (1, 32760, 64))
        self.assertTrue(torch.isfinite(tokens).all())

    def test_completed_gpu_case_is_not_replayed(self):
        import run as runner
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "attempt_started.json").write_text("{}")
            with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as call:
                with self.assertRaisesRegex(ValueError, "already been launched"):
                    runner.launch(root, False)
                self.assertEqual(call.call_count, 1)

    def test_hold_shell_has_valid_separate_closing_bracket(self):
        import subprocess
        import run as runner
        with tempfile.TemporaryDirectory(prefix="d eval ") as t:
            root = Path(t)
            captured = []
            def fake_run(args, **kwargs):
                if args[:2] == ["tmux", "new-session"]:
                    captured.append(args[-1])
                    return SimpleNamespace(returncode=0)
                return SimpleNamespace(returncode=1)
            with patch.object(runner.subprocess, "run", side_effect=fake_run), patch.object(runner.subprocess, "check_output", return_value="777\n"):
                runner.launch(root, True)
            condition = captured[0].split("; do sleep 1; done")[0].removeprefix("while ")
            # A valid test returns 0 (flag absent), then 1 (flag present), never 2 (syntax error).
            self.assertEqual(subprocess.call(["bash", "-c", condition]), 0)
            (root / "start.ready").touch()
            self.assertEqual(subprocess.call(["bash", "-c", condition]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
