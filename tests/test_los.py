"""LOS training/cache/checkpoint integration, not a validation-score experiment."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_los_cache
import evaluate
import run_los
import train
from radiounet import RadioWNet, checkpoint_features


META = {"grid": [256, 256], "resolution_m_per_px": 1.0,
        "heightmap_range_m": [0., 496.], "pathloss_range_db": [20., 228.],
        "simulation": {"tx_height_m_agl": 3., "rx_height_m_agl": 1.}}


class LOSIntegrationTests(unittest.TestCase):
    def test_cache_train_reload_and_full_fixture_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "metadata.json").write_text(json.dumps(META))
            for split in ("train", "val"):
                (data / f"{split}_index.csv").write_text("sample_id,hm_row,tx_row,tx_col\n00000_00,0,32,150\n")
                np.save(data / f"{split}_hm.npy", np.zeros((1, 256, 256), np.float32))
                np.save(data / f"{split}_pl58.npy", np.full((1, 256, 256), 80, np.float32))
                mask = np.ones((1, 256, 256), np.uint8)
                mask[:, 8:12, 20:24] = 0
                np.save(data / f"{split}_mask58.npy", np.packbits(mask, axis=1))
            cache_args = build_los_cache.build_parser().parse_args([
                "--data-root", str(data), "--cache-root", str(root / "cache"), "--device", "cpu"])
            with contextlib.redirect_stdout(io.StringIO()):
                build_los_cache.main(cache_args)
                build_los_cache.main(cache_args)  # complete cache is reused
            torch.manual_seed(0)
            baseline = RadioWNet(phase="secondU")
            with torch.no_grad():
                baseline.Wconv_up000[0].bias.fill_(.2)
            baseline_path = root / "baseline.pt"
            torch.save({"model": baseline.state_dict(), "args": {"band": "58", "phase": "secondU"}}, baseline_path)
            args = train.build_parser().parse_args([
                "--data-root", str(data), "--los-cache", str(root / "cache"),
                "--phase", "secondU", "--second-features", "los", "--init-from", str(baseline_path),
                "--device", "cpu", "--no-amp", "--num-workers", "0", "--batch-size", "1",
                "--epochs", "1", "--panels", "0", "--no-fingerprints", "--no-gradient-diagnostics",
                "--out", str(root / "run"), "--run-name", "los"])
            # The new main path must not calculate these optional diagnostics.
            with mock.patch("experiment_provenance.fingerprint", side_effect=AssertionError("hash called")), \
                 mock.patch("train.first_stage_hash", side_effect=AssertionError("hash called")), \
                 mock.patch("train.masked_gradient_loss", side_effect=AssertionError("gradient diagnostic called")), \
                 contextlib.redirect_stdout(io.StringIO()):
                train.main(args)
            ckpath = root / "run" / "los_best.pt"
            checkpoint = torch.load(ckpath, weights_only=False)
            self.assertEqual(checkpoint_features(checkpoint)[0], "los")
            for name, value in baseline.state_dict().items():
                if not name.startswith("W"):
                    self.assertTrue(torch.equal(value, checkpoint["model"][name]))
            self.assertGreater(checkpoint["model"]["Wconv_up000.0.weight"][:, -3].abs().sum().item(), 0)
            provenance = json.loads((root / "run" / "los_provenance.json").read_text())
            self.assertFalse(provenance["fingerprints_enabled"])
            self.assertNotIn("firstU_state_sha256", provenance)
            scoring = evaluate.build_parser().parse_args([
                "--ckpt", str(ckpath), "--data-root", str(data), "--device", "cpu", "--no-amp",
                "--num-workers", "1", "--batch-size", "1", "--panels", "0", "--out", str(root / "val.json")])
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate.main(scoring)  # restores the cache path saved in the checkpoint
            result = json.loads((root / "val.json").read_text())["summary"]
            self.assertEqual(result["second_features"], "los")
            self.assertEqual(result["samples"], 1)
            self.assertAlmostEqual(result["rmse_db_masked"], checkpoint["metrics"]["val/rmse_db_masked"], places=5)

    def test_entry_runs_one_full_training_and_one_evaluation(self):
        settings = dict(epochs=150, batch_size=16, lr=1e-4, lr_sched="cosine", lr_min=1e-6,
                        lr_step=20, lr_gamma=.1, clip=0, amp=True, no_augment=False,
                        num_workers=8, device="cuda", limit_batches=0)
        args = run_los.build_parser().parse_args([
            "--data-root", "data", "--baseline-ckpt", "baseline.pt",
            "--cache-root", "cache", "--out-root", "new_run"])
        commands = run_los.build_commands(args, settings)
        self.assertEqual([item["kind"] for item in commands], ["cache", "train", "evaluate"])
        parsed = train.build_parser().parse_args(commands[1]["command"][2:])
        self.assertEqual(parsed.epochs, 150)
        self.assertEqual(parsed.limit_batches, 0)
        self.assertEqual(parsed.second_features, "los")
        self.assertEqual(parsed.init_from, str(Path("baseline.pt").resolve()))
        self.assertEqual(parsed.grad_weight, 0)
        self.assertTrue(parsed.no_fingerprints and parsed.no_gradient_diagnostics)
        scoring = evaluate.build_parser().parse_args(commands[2]["command"][2:])
        self.assertEqual(scoring.limit, 0)
        self.assertIsNone(scoring.boundary_out)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
