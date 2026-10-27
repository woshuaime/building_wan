"""CPU behavioral checks for window pairing, isolation, resume and metric conventions."""
import tempfile
import io
import unittest
from unittest.mock import patch
from complex_common import *
from complex_train import remaining_rows, validate_history
from complex_evaluate import paired_change, summed, metrics, group_metrics
from run_pilot import jobs


class Orbit17Tests(unittest.TestCase):
    def setUp(self):
        self.eligible = {m: {cs: list(range(64 if m == 'C_lin17' else 80)) for cs in TRAIN} for m in METHODS}
        self.original = [dict(update=str(i), cs=TRAIN[(i-1) % 8], start=str(i % 64), timestep_index=str(i % 1000),
                              noise_seed=str(i), stochastic_seed=str(1000+i)) for i in range(1, 301)]

    def test_crossing_order_and_excluded_duplicate(self):
        self.assertEqual(window_ids('C_ring17', 72), list(range(72, 80)) + list(range(9)))
        self.assertEqual(window_ids('C_ring17', 79), [79] + list(range(16)))
        for method in METHODS:
            for start in self.eligible[method]['CS01']:
                frames = window_ids(method, start)
                self.assertEqual(len(set(frames)), 17)
                self.assertNotIn(80, frames)
        with self.assertRaises(ValueError):
            window_ids('C_lin17', 64)

    def test_visibility_filter_keeps_individual_black_frames(self):
        visible = [False]*80; visible[0] = True
        self.assertEqual(eligible_starts(visible, 'C_lin17'), [0])
        self.assertEqual(eligible_starts(visible, 'C_ring17'), [0] + list(range(64, 80)))

    def test_original_prefix_and_paired_random_conditions(self):
        state = random.getstate()
        tables = paired_tables(self.original, self.eligible)
        self.assertEqual(random.getstate(), state)
        for i, old in enumerate(self.original):
            self.assertEqual({k: tables['C_lin17'][i][k] for k in old}, old)
        for linear, circular in zip(*[tables[m] for m in METHODS]):
            for key in ('update', 'cs', 'data_version', 'timestep_index', 'noise_seed', 'stochastic_seed'):
                self.assertEqual(linear[key], circular[key])
        self.assertEqual(tables, paired_tables(self.original, self.eligible))

    def test_extension_draw_order_and_separate_circular_rng(self):
        tables = paired_tables(self.original, self.eligible)
        rng = random.Random(20261027)
        cs = rng.choice(TRAIN)
        expected = [cs, str(rng.choice(self.eligible['C_lin17'][cs])), str(rng.randrange(1000)),
                    str(rng.randrange(2**31)), str(rng.randrange(2**31))]
        self.assertEqual([tables['C_lin17'][300][k] for k in ('cs', 'start', 'timestep_index', 'noise_seed', 'stochastic_seed')], expected)
        ring = random.Random(20261028)
        for row in tables['C_ring17']:
            self.assertEqual(int(row['start']), ring.choice(self.eligible['C_ring17'][row['cs']]))

    def test_cache_identity_uses_order_and_version(self):
        row = dict(cs='CS01', start='0', method='C_lin17', data_version='Rev1', frame_ids=json.dumps(list(range(17))))
        with patch('complex_common.config', return_value={'encoding_config_id': 'encoding'}):
            self.assertEqual(cache_path(row), cache_path(dict(row, method='C_ring17')))
            self.assertNotEqual(cache_path(row), cache_path(dict(row, frame_ids=json.dumps(list(range(16, -1, -1))))))
            self.assertNotEqual(cache_path(row), cache_path(dict(row, data_version='OriginalHard')))

    def test_inference_conditions_never_read_gt(self):
        calls = []
        def fake_source(cs, role, frame, version):
            calls.append(role)
            return Image.new('L' if role == 'exact' else 'RGB', (512, 512), 0)
        with patch('complex_common.source', side_effect=fake_source):
            sample = conditions('CS12', [79, 0, 1], 'OriginalHard')
        self.assertEqual(calls, ['damaged', 'exact']*3)
        self.assertNotIn('gt', sample)
        self.assertEqual(sample['frame_ids'], [79, 0, 1])

    def test_validation_and_hard_never_enter_training(self):
        row = dict(cs='CS12', method='C_ring17', start='72', frame_ids=json.dumps(window_ids('C_ring17', 72)), data_version='Rev1')
        with self.assertRaises(ValueError):
            training_sample(row)
        with self.assertRaises(ValueError):
            training_sample(dict(row, cs='CS10', data_version='OriginalHard'))

    def test_resume_uses_absolute_budget_and_checks_history(self):
        rows = paired_tables(self.original, self.eligible)['C_lin17']
        self.assertEqual([int(r['update']) for r in remaining_rows(rows, 300, 600)], list(range(301, 601)))
        self.assertEqual(remaining_rows(rows, 600, 600), [])
        validate_history(rows, rows[:300], 300)
        with self.assertRaises(ValueError):
            validate_history(rows, [dict(rows[0], start='wrong')] + rows[1:300], 300)
        with self.assertRaises(ValueError):
            remaining_rows(rows, 400, 300)

    def test_queue_has_both_groups_exactly_40_predictions(self):
        tasks = jobs()
        inference = [(a[1], int(a[2]), a[3], a[4]) for _, a, _ in tasks if a[0] == 'complex_infer.py']
        self.assertEqual(len(inference), 40)
        self.assertEqual(set(inference), set(trajectory_order()))
        self.assertEqual(len(set(inference)), 40)
        training = [(a[1], a[3]) for _, a, _ in tasks if a[0] == 'complex_train.py']
        self.assertEqual(training, [('C_lin17', '300'), ('C_lin17', '600'), ('C_ring17', '600')])
        names = [r[0] for r in tasks]
        self.assertLess(names.index('early_probe'), names.index('train_C_lin17_600'))
        self.assertLess(names.index('train_C_ring17_600'), names.index('infer_C_ring17_300_Rev1_CS10'))

    def test_metric_zero_empty_and_pixel_weighting(self):
        gt = np.zeros((4, 4, 3)); pred = np.ones_like(gt)
        mask = np.zeros((4, 4), dtype=bool); mask[0, 0] = True
        first = metrics.pixel_sums(pred, gt, gt, mask, mask)
        second = metrics.pixel_sums(gt, gt, gt, np.ones_like(mask), mask)
        aggregate = metrics.reduced(summed([first, second]))
        self.assertAlmostEqual(aggregate['masked_mae'], 1/17)
        empty = metrics.reduced(metrics.pixel_sums(gt, gt, gt, np.zeros_like(mask), np.zeros_like(mask)))
        self.assertEqual(empty['masked_mae'], 'N/A')
        perfect = metrics.reduced(second)
        self.assertEqual(perfect['masked_psnr'], 'Infinity')
        self.assertEqual(paired_change(0, .1, 'masked_mae')['relative_change_percent'], 'N/A')
        self.assertTrue(paired_change(20, 19, 'masked_psnr')['regressed'])

    def test_ordinary_c_uses_all_latents_not_mask_weighting(self):
        import torch
        base = legacy('complex_train')
        pred = torch.tensor([[[[[1., 2.], [3., 4.]]]]], requires_grad=True)
        mask = torch.zeros((1, 1, 2, 2)); edge = torch.zeros_like(mask)
        result = base.weighted_error(pred, torch.zeros_like(pred), mask, edge, 'C')
        self.assertEqual(float(result), 7.5)
        result.backward()
        self.assertTrue(bool((pred.grad != 0).all()))

    def test_loss_grid_follows_local_positions_after_reorder(self):
        pixel = np.zeros((17, 512, 512), dtype=bool)
        pixel[0, 8, 8] = True; pixel[16, 16, 16] = True
        mapped = loss_grid(pixel)
        self.assertEqual(mapped.shape, (5, 64, 64))
        self.assertEqual(mapped[0, 1, 1], 1)
        self.assertEqual(mapped[4, 2, 2], 1)
        self.assertEqual(mapped.sum(), 2)

    def test_video_encode_decode_and_order(self):
        from media import encode_and_check
        with tempfile.TemporaryDirectory(dir=RUN / 'tmp') as folder:
            path = Path(folder) / 'cpu_video.mp4'
            def frame(i):
                return np.full((64, 64, 3), 30 + 2*i, dtype=np.uint8)
            result = encode_and_check(path, frame, (64, 64))
            self.assertEqual(result['frames'], 81)
            self.assertLessEqual(result['max_channel_encoding_mae_255'], 3)

    def test_early_and_final_metrics_cover_all_pairs_without_gpu(self):
        import complex_evaluate as evaluation
        saved = {}
        mask = np.ones((4, 4), dtype=bool)
        rgb = np.full((4, 4, 3), 64/255)
        values = metrics.pixel_sums(rgb, rgb, rgb, mask, mask)
        originals = []
        old_frames = []
        for cs in IDS:
            common = dict(cs=cs, source_id=1, split='train' if cs in TRAIN else 'val', method='C')
            old_frames += [dict(common, frame=f'{i:03}', **metrics.reduced(values), **values) for i in range(80)]
            originals.append(dict(common, group='test', **metrics.reduced(values), **values,
                                  loop_full_mae=0, loop_masked_mae=0,
                                  regular_frames=80, masked_valid_frames=80, structure_valid_frames=80, known_valid_frames=0))
        def fake_read(path):
            if Path(path).is_relative_to(RUN):
                return saved[Path(path).name]
            return old_frames if Path(path).name == 'frame_metrics.csv' else originals
        def capture(path, rows, fields=None):
            rows = list(rows)
            # Exercise real CSV field handling, including empty regression lists.
            writer = csv.DictWriter(io.StringIO(), fieldnames=fields or list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
            saved[Path(path).name] = rows
        with patch.object(evaluation, 'read_csv', side_effect=fake_read), \
             patch.object(evaluation, 'write_csv', side_effect=capture), \
             patch.object(evaluation, 'labels', return_value=(np.stack([mask]*80), np.stack([mask]*80))), \
             patch.object(evaluation, 'source', side_effect=lambda cs, role, f, v: Image.fromarray(
                 np.full((4, 4), 255, dtype=np.uint8)) if role == 'exact' else Image.fromarray(np.full((4, 4, 3), 64, dtype=np.uint8))), \
             patch.object(evaluation, 'read_image', return_value=Image.fromarray(np.full((4, 4, 3), 64, dtype=np.uint8))), \
             patch.object(evaluation, 'valid_prediction', return_value=True):
            early = evaluation.compute_metrics(early=True)
            final = evaluation.compute_metrics()
        self.assertEqual(len(early), 8)
        self.assertEqual(len(final), 56)  # 16 old C + 40 newly predicted trajectories.
        self.assertEqual(len(saved['frame_metrics.csv']), 56*80)
        self.assertEqual(len(saved['segment_metrics.csv']), 56*3)
        self.assertEqual(len(saved['group_summary.csv']), 12)
        self.assertEqual(len([r for r in saved['paired_differences.csv'] if r['comparison'] == 'ring600_vs_lin600']), 16*11)


if __name__ == '__main__':
    unittest.main(verbosity=2)
