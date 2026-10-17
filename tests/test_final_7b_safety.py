"""Final report inference and9B hold checks; synthetic summaries only."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import plot_route_b_final as plots
import finish_7b_safely as finish
import cleanup_7b_evaluation_disk as disk
from experiment_common import AXES, ROOT, sha
from route_b_staged import SEEDS


def fixture():
    scores = {'N-seed42': {a: .5 for a in AXES}}
    for arm, value in (('A1', .6), ('A2', .55), ('U', .65)):
        for seed in SEEDS:
            scores[f'{arm}-seed{seed}'] = {a: value for a in AXES}
    contrasts = {}
    for label, delta, interval in (('A1-A2', .05, [.01, .09]), ('A1-U', -.05, [-.09, -.01]), ('A1-N', .1, [-.01, .2])):
        contrasts[label] = {a: {'difference': delta, 'bonferroni_familywise_ci95': interval} for a in AXES}
    return {'status': 'all16_evaluations_complete', 'scores': scores, 'matched_seeds': SEEDS,
            'family_size': 12, 'bootstrap_replicates': 10000, 'paired_contrasts': contrasts}


class FinalSafetyTests(unittest.TestCase):
    def test_all12_interval_conclusions_and_zero_boundary(self):
        data = fixture()
        with patch.object(plots, 'read_json', return_value=data):
            _, rows = plots.validated_results()
        self.assertEqual(len(rows), 12)
        self.assertEqual([r['direction'] for r in rows], ['left_higher'] * 4 + ['right_higher'] * 4 + ['interval_includes_zero'] * 4)
        data['paired_contrasts']['A1-A2']['pope']['bonferroni_familywise_ci95'] = [0, .09]
        with patch.object(plots, 'read_json', return_value=data):
            _, rows = plots.validated_results()
        self.assertEqual(rows[0]['direction'], 'interval_includes_zero')

    def test_four_seeds_cannot_render_as_final(self):
        data = fixture()
        data['matched_seeds'] = SEEDS[:-1]
        with patch.object(plots, 'read_json', return_value=data):
            with self.assertRaises(AssertionError):
                plots.validated_results()

    def test_miscomputed_delta_cannot_enter_final_figure(self):
        data = fixture()
        data['paired_contrasts']['A1-A2']['mmstar']['difference'] = .3
        with patch.object(plots, 'read_json', return_value=data):
            with self.assertRaises(AssertionError):
                plots.validated_results()

    def test_9b_cannot_resume_before_figures_are_server_verified(self):
        responses = [{'status': '9b_queues_paused_for_7b_delivery'},
                     {'status': 'full_7b_content_and_delivery_verified'}, {'status': 'generated_locally_only'}]
        with patch.object(finish, 'read_json', side_effect=responses), patch.object(finish.subprocess, 'run') as start:
            with self.assertRaises(AssertionError):
                finish.resume_9b('synthetic-token')
            start.assert_not_called()

    def cleanup_guard_fixture(self, protected=False, wrong_hash=False):
        temporary = ROOT / 'tmp'
        temporary.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=temporary) as name:
            root = Path(name)
            self.assertTrue(root.resolve().is_relative_to(ROOT.resolve()))
            path = root / ('protected/current-checkpoint.pt' if protected else '.model_cache/qwen3b/model.safetensors')
            path.parent.mkdir(parents=True)
            path.write_bytes(b'synthetic cache fixture')
            record = {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                      'sha256': 'wrong' if wrong_hash else sha(path)}
            plan = {'expected_free_GiB': 30, 'files': [record]}
            with patch.object(disk, 'ROOT', root), patch.object(disk, 'OLD', root / 'old'), \
                 patch.object(disk, 'CALIBRATION', root / 'calibration'), patch.object(disk, 'DESTINATION', root / 'report'), \
                 patch.object(disk, 'plan', return_value=plan):
                with self.assertRaises(AssertionError):
                    disk.cleanup('synthetic-token', apply=True)
            self.assertTrue(path.is_file())

    def test_cleanup_never_deletes_current_checkpoint_outside_allowed_caches(self):
        self.cleanup_guard_fixture(protected=True)

    def test_cleanup_rechecks_hash_immediately_before_deletion(self):
        self.cleanup_guard_fixture(wrong_hash=True)


if __name__ == '__main__':
    unittest.main()
