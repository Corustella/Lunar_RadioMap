"""Geometry correctness and checkpoint/optimizer integration; no RMSE claims."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import ablate_coordinates
import evaluate
import lunar_dataset
import train
from experiment_provenance import first_stage_hash
from radiounet import (RadioWNet, geometry_config, tx_geometry,
                       checkpoint_features, load_model_state, trains_in_phase)


META = {"grid": [256, 256], "resolution_m_per_px": 1.0,
        "heightmap_range_m": [0., 496.], "pathloss_range_db": [20., 228.],
        "simulation": {"tx_height_m_agl": 3., "rx_height_m_agl": 1.}}
CONFIG = geometry_config(META)
LAYERS = ("Wlayer00.0.weight", "Wconv_up00.0.weight", "Wconv_up000.0.weight")


def sample(row=32, col=150):
    height = (torch.arange(256)[:, None] + torch.arange(256)[None, :]) / 1024
    x = torch.zeros(1, 2, 256, 256)
    x[0, 0] = height
    x[0, 1, row, col] = 1
    return x


class GeometryTests(unittest.TestCase):
    def test_signed_coordinates_and_antenna_offsets(self):
        for row, col in ((0, 0), (255, 255), (128, 111)):
            x = sample(row, col)
            f = tx_geometry(x, CONFIG)
            self.assertEqual(f.shape, (1, 3, 256, 256))
            self.assertAlmostEqual(f[0, 0, 17, 29].item(), (29-col)/255, places=7)
            self.assertAlmostEqual(f[0, 1, 17, 29].item(), (17-row)/255, places=7)
            self.assertAlmostEqual(f[0, 2, row, col].item(), -2/496, places=7)
            self.assertAlmostEqual(f[0, 2, 17, 29].item(),
                                   x[0, 0, 17, 29].item()-x[0, 0, row, col].item()-2/496, places=7)
        batched = tx_geometry(torch.cat((sample(0, 0), sample(255, 255))), CONFIG)
        self.assertEqual(batched[0, 0, -1, -1].item(), 1)
        self.assertEqual(batched[1, 0, 0, 0].item(), -1)

    def test_all_eight_augmentations_use_current_axes_and_tx_height(self):
        x = sample()
        for flip in (False, True):
            for rotation in range(4):
                with mock.patch.object(lunar_dataset.random, "random", side_effect=[0 if flip else 1, 1]), \
                        mock.patch.object(lunar_dataset.random, "randint", return_value=rotation):
                    maps, _ = lunar_dataset._augment([v.numpy() for v in x[0]], [])
                transformed = torch.from_numpy(np.stack(maps))[None]
                row, col = torch.nonzero(transformed[0, 1], as_tuple=False)[0].tolist()
                f = tx_geometry(transformed, CONFIG)
                self.assertEqual(f[0, 0, row, col].item(), 0)
                self.assertEqual(f[0, 1, row, col].item(), 0)
                self.assertAlmostEqual(f[0, 0, row, 255].item(), (255-col)/255, places=7)
                self.assertAlmostEqual(f[0, 1, 255, col].item(), (255-row)/255, places=7)
                expected = transformed[0, 0]-transformed[0, 0, row, col]-2/496
                torch.testing.assert_close(f[0, 2], expected)

    def test_reject_ambiguous_tx_and_wrong_height_units(self):
        for bad in ("absent", "multiple", "soft", "meters"):
            x = sample()
            if bad == "absent":
                x[:, 1] = 0
            elif bad == "multiple":
                x[0, 1, 0, 0] = 1
            elif bad == "soft":
                x[:, 1] *= .5
            else:
                x[:, 0] *= 496
            with self.assertRaises(ValueError):
                tx_geometry(x, CONFIG)


class ModelTests(unittest.TestCase):
    def test_padding_outputs_gradients_and_frozen_firstU(self):
        torch.manual_seed(9)
        baseline = RadioWNet(phase="secondU")
        with torch.no_grad():
            baseline.Wconv_up000[0].bias.fill_(.2)
        state = baseline.state_dict()
        checkpoint = {"model": state, "args": {}}
        x = sample()
        with torch.no_grad():
            reference = baseline(x)
        hashes, modes = [], []
        for mode in ("zeros", "tx_xyz"):
            model = RadioWNet(phase="secondU", second_features=mode, feature_config=CONFIG)
            migration = load_model_state(model, checkpoint, initialize=True)
            self.assertEqual(migration["zero_padded_weights"], list(LAYERS))
            self.assertEqual(set(model.state_dict()), set(state))
            self.assertEqual(sum(p.numel() for p in model.parameters())-
                             sum(p.numel() for p in baseline.parameters()), 2115)
            for name, value in model.state_dict().items():
                if name in LAYERS:
                    self.assertTrue(torch.equal(value[:, :-3], state[name]))
                    self.assertEqual(torch.count_nonzero(value[:, -3:]).item(), 0)
                else:
                    self.assertTrue(torch.equal(value, state[name]))
            with torch.no_grad():
                outputs = model(x)
            for old, new in zip(reference, outputs):
                torch.testing.assert_close(new, old, rtol=0, atol=1e-6)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(trains_in_phase(name, "secondU"))
            before = first_stage_hash(model.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            model(x)[1].square().mean().backward()
            added = [dict(model.named_parameters())[name].grad[:, -3:] for name in LAYERS]
            self.assertTrue(all(torch.isfinite(v).all() for v in added))
            if mode == "zeros":
                self.assertTrue(all(torch.count_nonzero(v) == 0 for v in added))
            else:
                self.assertGreater(sum(torch.count_nonzero(v).item() for v in added), 0)
            optimizer.step()
            self.assertEqual(first_stage_hash(model.state_dict()), before)
            hashes.append(before)
            modes.append(model)
        self.assertEqual(hashes[0], hashes[1])
        checkpoint = {"model": modes[1].state_dict(), "args": {"second_features": "tx_xyz"},
                      "feature_config": CONFIG}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"geo.pt"
            torch.save(checkpoint, path)
            restored = torch.load(path, weights_only=False)
        load_model_state(modes[1], restored)
        with self.assertRaises(ValueError):
            load_model_state(modes[0], restored)
        with self.assertRaises(ValueError):
            load_model_state(modes[0], {"model": state, "args": {}})
        wrong = copy.deepcopy(restored)
        wrong["feature_config"]["tx_height_m_agl"] = 4
        with self.assertRaises(ValueError):
            load_model_state(modes[1], wrong)
        with self.assertRaises(ValueError):
            checkpoint_features({"args": {"second_features": "tx_xyz"}})
        self.assertEqual(checkpoint_features({"args": {}}), ("none", None))


class IntegrationTests(unittest.TestCase):
    def test_train_resume_and_evaluate_restore_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/"data"
            data.mkdir()
            (data/"metadata.json").write_text(json.dumps(META))
            for split in ("train", "val"):
                (data/f"{split}_index.csv").write_text("sample_id,hm_row,tx_row,tx_col\n00000_00,0,32,150\n")
                np.save(data/f"{split}_hm.npy", sample()[0, 0].numpy()[None]*496)
                np.save(data/f"{split}_pl58.npy", np.full((1, 256, 256), 80, np.float32))
                np.save(data/f"{split}_mask58.npy", np.packbits(np.ones((1, 256, 256), np.uint8), axis=1))
            baseline = RadioWNet(phase="secondU")
            with torch.no_grad():
                baseline.Wconv_up000[0].bias.fill_(.2)
            path = root/"baseline.pt"
            torch.save({"model": baseline.state_dict(), "args": {"band": "58", "phase": "secondU"}}, path)
            args = train.build_parser().parse_args([
                "--data-root", str(data), "--phase", "secondU", "--second-features", "tx_xyz",
                "--init-from", str(path), "--device", "cpu", "--no-amp", "--num-workers", "0",
                "--batch-size", "1", "--epochs", "1", "--limit-batches", "1", "--panels", "0",
                "--out", str(root/"run"), "--run-name", "geo"])
            with contextlib.redirect_stdout(io.StringIO()):
                train.main(args)
                args.resume = "auto"
                train.main(args)  # restores model/optimizer/scheduler, no extra epoch
            ckpath = root/"run"/"geo_best.pt"
            checkpoint = torch.load(ckpath, weights_only=False)
            self.assertEqual(checkpoint_features(checkpoint), ("tx_xyz", CONFIG))
            self.assertEqual(first_stage_hash(checkpoint["model"]), first_stage_hash(baseline.state_dict()))
            scoring = evaluate.build_parser().parse_args([
                "--ckpt", str(ckpath), "--data-root", str(data), "--device", "cpu", "--no-amp",
                "--num-workers", "0", "--batch-size", "1", "--panels", "0", "--out", str(root/"val.json")])
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate.main(scoring)
            result = json.loads((root/"val.json").read_text())
            self.assertEqual(result["summary"]["second_features"], "tx_xyz")
            self.assertEqual(result["summary"]["samples"], 1)
            args.second_features = "zeros"
            with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                train.main(args)

    def test_paired_commands_preserve_baseline_settings_and_initialization(self):
        saved = dict(phase="secondU", band="58", loss="masked", grad_weight=0,
                     epochs=150, batch_size=16, lr=1e-4, lr_sched="cosine", lr_min=1e-6,
                     lr_step=20, lr_gamma=.1, clip=0, amp=True, no_augment=False,
                     num_workers=8, device="cuda", limit_batches=0)
        settings = ablate_coordinates.baseline_settings({"args": saved})
        args = ablate_coordinates.build_parser().parse_args([
            "--data-root", "data", "--baseline-ckpt", "baseline.pt", "--out-root", "new_run"])
        commands = ablate_coordinates.build_commands(args, settings)
        paired = [v["command"] for v in commands if v["kind"] == "train"]
        for cmd in paired:
            parsed = train.build_parser().parse_args(cmd[2:])
            self.assertEqual(parsed.init_from, str(Path("baseline.pt").resolve()))
            self.assertEqual(parsed.grad_weight, 0)
            self.assertEqual(parsed.epochs, 150)
            self.assertEqual(parsed.seed, 0)
        candidate = paired[1].copy()
        for option in ("--out", "--second-features"):
            candidate[candidate.index(option)+1] = paired[0][paired[0].index(option)+1]
        self.assertEqual(candidate, paired[0])
        parsed.init_from = "auto"
        parsed.run_name = "radiownet_58_masked_secondU_tx_xyz"
        with mock.patch.object(train.os.path, "exists", side_effect=lambda p: p.endswith("radiownet_58_masked_firstU_best.pt")):
            self.assertTrue(train.resolve_init_from(parsed).endswith("radiownet_58_masked_firstU_best.pt"))
        saved["grad_weight"] = .03
        with self.assertRaises(ValueError):
            ablate_coordinates.baseline_settings({"args": saved})


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
