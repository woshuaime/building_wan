"""CPU contract tests for the new data-version variable and paired delivery."""
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import complex_common as common
import complex_train as training
import complex_evaluate as evaluation
from run_pilot import jobs

class CmixTests(unittest.TestCase):
    def test_sampling_preserves_source_and_global_randomness(self):
        original = common.read_csv(common.REV1_RUN / 'shared_sampling_table.csv')
        state = random.getstate()
        rows = common.mixed_sampling(original)
        self.assertEqual(random.getstate(), state)
        self.assertEqual([{k: v for k, v in r.items() if k != 'data_version'} for r in rows], original)
        hard = common.Counter(r['cs'] for r in rows if r['data_version'] == 'OriginalHard')
        self.assertEqual(hard, {'CS09': 20, 'CS10': 14, 'CS11': 16})
        self.assertEqual(common.Counter(r['data_version'] for r in rows), {'Rev1': 250, 'OriginalHard': 50})
        self.assertEqual(rows, common.sampling())

    def test_validation_and_frame080_cannot_enter_training(self):
        with patch.object(common, 'conditions') as loader:
            for cs in common.VAL:
                for version in common.VERSIONS:
                    with self.assertRaises(ValueError):
                        common.training_sample(cs, 0, version)
            with self.assertRaises(ValueError):
                common.training_sample('CS09', 64, 'Rev1')
            loader.assert_not_called()

    def test_versions_resolve_to_correct_input_roots(self):
        for version in common.VERSIONS:
            path = common.source_path('CS10', 'damaged', 24, version)
            self.assertTrue(path.is_relative_to(common.data_root(version)))
            self.assertTrue(path.is_file())
        self.assertNotEqual(common.source_path('CS10', 'damaged', 24, 'Rev1'), common.source_path('CS10', 'damaged', 24, 'OriginalHard'))

    def test_condition_loader_does_not_read_gt(self):
        roles = []
        def source(cs, role, frame, version):
            roles.append(role)
            self.assertEqual(version, 'OriginalHard')
            return common.Image.new('L' if role == 'exact' else 'RGB', (512, 512), 0)
        with patch.object(common, 'source', source):
            result = common.conditions('CS12', count=2, version='OriginalHard')
        self.assertEqual(roles, ['damaged', 'exact', 'damaged', 'exact'])
        self.assertNotIn('gt', result)

    def test_cache_does_not_cross_versions(self):
        import torch
        from pathlib import Path
        calls = []
        def sample(cs, start, version):
            return {'version': version}, common.np.zeros((5, 64, 64), common.np.float32), common.np.zeros((5, 64, 64), common.np.float32)
        def encode(pipe, sample, *args):
            calls.append(sample['version'])
            return {'input_latents': torch.zeros(1, 16, 5, 64, 64),
                    'vace_context': torch.full((1, 96, 5, 64, 64), 1.0 if sample['version'] == 'Rev1' else 2.0)}
        pipe = SimpleNamespace(device='cpu', vae=SimpleNamespace(to=lambda device: None))
        with tempfile.TemporaryDirectory(dir=common.RUN / 'tmp') as directory:
            root = Path(directory); (root / 'cache').mkdir()
            with patch.object(training, 'RUN', root), patch.object(training, 'config', return_value={'config_id': 'test', 'training': {}}), \
                 patch.object(training, 'training_sample', sample), patch('vace_roof_training.encode_sample', encode):
                values = []
                for version in ('Rev1', 'OriginalHard', 'Rev1'):
                    encoded, _, _ = training.cached_window(pipe, torch.zeros(1), {'cs': 'CS10', 'start': '3', 'data_version': version})
                    values.append(float(encoded['vace_context'].mean()))
        self.assertEqual(calls, ['Rev1', 'OriginalHard'])
        self.assertEqual(values, [1.0, 2.0, 1.0])

    def test_original_training_and_inference_parameters_retained(self):
        old = common.read_json(common.REV1_RUN / 'resolved_config.json')
        new = common.config()
        self.assertEqual(old['model'], new['model'])
        self.assertEqual(old['inference'], new['inference'])
        for key in old['training']:
            if key != 'frozen_encoding_cache':
                self.assertEqual(old['training'][key], new['training'][key], key)

    def test_queue_is_one_training_plus_exactly_sixteen_inferences(self):
        queue = jobs()
        self.assertEqual(len(queue), 18)
        self.assertEqual(queue[0][0], 'train_C_mix')
        self.assertEqual([r[1][1:] for r in queue[1:5]], [['OriginalHard', cs] for cs in common.HARD_IDS])
        self.assertEqual([r[1][1:] for r in queue[5:17]], [['Rev1', cs] for cs in common.IDS])
        self.assertEqual(queue[-1][0], 'evaluate')

    def test_relative_zero_and_psnr_direction(self):
        self.assertEqual(evaluation.paired_change(0, .1, 'masked_mae')['relative_change_percent'], 'N/A')
        self.assertTrue(evaluation.paired_change(20, 19, 'masked_psnr')['regressed'])
        self.assertFalse(evaluation.paired_change(.2, .1, 'masked_mae')['regressed'])
        self.assertEqual(evaluation.paired_change('N/A', .1, 'masked_mae')['absolute_change'], 'N/A')

    def test_panel_preserves_full_content_and_same_frame(self):
        calls = []
        def pixels(version, cs, method, frame):
            calls.append((version, cs, method, frame))
            value = ['Before', 'GT', 'C_old', 'C_mix'].index(method)*40
            return common.np.full((512, 512, 3), value, common.np.uint8)
        with patch.object(evaluation, 'pixels', pixels):
            result = common.np.asarray(evaluation.panel('OriginalHard', 'CS12', 80))
        self.assertEqual(result.shape, (1088, 1024, 3))
        for i in range(4):
            x, y = (i % 2)*512, (i // 2)*544+32
            self.assertTrue((result[y:y+512, x:x+512] == i*40).all())
        self.assertTrue(all(r[0] == 'OriginalHard' and r[1] == 'CS12' and r[3] == 80 for r in calls))

if __name__ == '__main__':
    unittest.main(verbosity=2)
