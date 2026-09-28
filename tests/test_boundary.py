"""CPU unit/integration checks; synthetic maps are NOT baseline measurements."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

import ablate_boundary
import boundary
import evaluate
import train
from experiment_provenance import first_stage_hash
from radiounet import RadioWNet


class LossTests(unittest.TestCase):
    def test_perfect_and_constant_offset(self):
        y = torch.arange(16).reshape(1, 1, 4, 4).float() / 16
        m = torch.ones_like(y)
        total, mse, gradient = train.compute_loss(y, y, m, "masked", .03, True)
        self.assertEqual((total.item(), mse.item(), gradient.item()), (0, 0, 0))
        total, mse, gradient = train.compute_loss(y + .25, y, m, "masked", .03, True)
        self.assertEqual(mse.item(), .25 ** 2)
        self.assertEqual(gradient.item(), 0)
        self.assertEqual(total.item(), mse.item())

    def test_signed_gradient_and_joint_pair_denominator(self):
        y = torch.tensor([[[[0., 1.], [0., 1.]]]])
        # Two reversed horizontal pairs each have error 2, two vertical pairs 0.
        self.assertEqual(boundary.masked_gradient_loss(1-y, y, torch.ones_like(y)).item(), 1)
        self.assertEqual(boundary.masked_gradient_loss(y*.5, y, torch.ones_like(y)).item(), .25)

    def test_invalid_values_do_not_affect_valid_loss_or_boundaries(self):
        torch.manual_seed(4)
        y, p = torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8)
        m = (torch.rand_like(y) > .3).float()
        y2, p2 = y.clone(), p.clone()
        y2[m == 0], p2[m == 0] = -1000, 900
        self.assertTrue(torch.equal(train.compute_loss(p, y, m, "masked", .03),
                                    train.compute_loss(p2, y2, m, "masked", .03)))
        for name, region in boundary.boundary_regions(y*208, m).items():
            self.assertTrue(torch.equal(region, boundary.boundary_regions(y2*208, m)[name]))
        # Even NaN fill is excluded by the gradient term (legacy MSE assumes finite fill).
        y2[m == 0], p2[m == 0] = float("nan"), float("nan")
        self.assertTrue(torch.equal(boundary.masked_gradient_loss(p, y, m),
                                    boundary.masked_gradient_loss(p2, y2, m)))

    def test_empty_isolated_and_single_pixel_masks_backpropagate_zero(self):
        for shape, isolated in (((1, 1, 3, 3), False), ((1, 1, 3, 3), True), ((1, 1, 1, 1), True)):
            p = torch.ones(shape, requires_grad=True)
            y, m = torch.zeros(shape), torch.zeros(shape)
            if isolated:
                m[..., 0, 0] = 1
            gradient = boundary.masked_gradient_loss(p, y, m)
            self.assertEqual(gradient.dtype, torch.float32)
            self.assertEqual(gradient.item(), 0)
            gradient.backward()
            self.assertTrue(torch.equal(p.grad, torch.zeros_like(p)))
            self.assertTrue(torch.isfinite(train.compute_loss(p, y, m, "masked", .03)))

    def test_auxiliary_computes_in_fp32(self):
        p = torch.ones(1, 1, 2, 2, dtype=torch.float16, requires_grad=True)
        self.assertEqual(boundary.masked_gradient_loss(p, torch.zeros_like(p), p.detach()).dtype, torch.float32)

    def test_zero_weight_matches_legacy_value_and_parameter_gradient(self):
        torch.manual_seed(2)
        x, y = torch.rand(2, 2, 7, 7), torch.rand(2, 1, 7, 7)
        m = (torch.rand_like(y) > .2).float()
        net = torch.nn.Conv2d(2, 1, 1)
        p = net(x)
        legacy = (((p-y)**2)*m).sum()/m.sum()
        legacy_grad = torch.autograd.grad(legacy, tuple(net.parameters()), retain_graph=True)
        current, _, _ = train.compute_loss(p, y, m, "masked", 0, True)
        current_grad = torch.autograd.grad(current, tuple(net.parameters()))
        self.assertTrue(torch.equal(legacy, current))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(legacy_grad, current_grad)))

    def test_resume_legacy_and_mismatch(self):
        train.check_resume_grad_weight({"args": {}}, 0)
        train.check_resume_grad_weight({"args": {"grad_weight": .01}}, .01)
        with self.assertRaises(ValueError):
            train.check_resume_grad_weight({"args": {}}, .01)


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.y = torch.zeros(1, 1, 32, 32)
        self.y[..., 16:] = 40
        self.mask = torch.ones_like(self.y)

    def test_valid_endpoints_and_square_dilation(self):
        points = boundary.high_change_points(self.y, self.mask)
        self.assertEqual(points.sum().item(), 64)
        regions = boundary.boundary_regions(self.y, self.mask)
        self.assertEqual(regions["edge_t10_r3"].sum().item(), 8*32)
        self.assertEqual(regions["near_missing_r5"].sum().item(), 0)
        m = self.mask.clone()
        m[..., 15] = 0
        self.assertEqual(boundary.high_change_points(self.y, m).sum().item(), 0)
        # Missing column plus <=5 neighbors; the column itself is excluded.
        self.assertEqual(boundary.boundary_regions(self.y, m)["near_missing_r5"].sum().item(), 10*32)

    def test_threshold_roundoff_and_sensitivity(self):
        y = self.y / 4  # exact physical jump 10 dB through normalized FP32
        y = (y / 208).double() * 208
        self.assertEqual(boundary.high_change_points(y, self.mask, 10).sum().item(), 64)
        self.assertEqual(boundary.high_change_points(y, self.mask, 20).sum().item(), 0)

    def test_shift_blur_and_amplitude_have_distinct_error_budgets(self):
        shift = torch.zeros_like(self.y)
        shift[..., 18:] = 40
        blur = F.avg_pool2d(F.pad(self.y, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
        predictions = {"shift": shift, "blur": blur, "amplitude": self.y*.8}
        with tempfile.TemporaryDirectory() as temp:
            report = boundary.BoundaryReport(Path(temp)/"report")
            report.update(predictions, self.y, self.mask, ["synthetic_58"], ["58"], 1)
            result = report.finish({}, {}, [])
        rows = {(r["output"], r["region"]): r for r in result["regions"]}
        self.assertEqual(rows["shift", "all_valid"]["sse_db2"], 2*32*40**2)
        self.assertLess(rows["shift", "all_valid"]["bias_db"], 0)
        self.assertAlmostEqual(rows["blur", "all_valid"]["bias_db"], 0)
        for kind in ("shift", "blur"):
            self.assertEqual(rows[kind, "nonedge_t10_r3"]["sse_db2"], 0)
            self.assertGreater(rows[kind, "edge_t10_r3"]["sse_db2"], 0)
        self.assertGreater(rows["amplitude", "nonedge_t10_r3"]["sse_db2"], 0)

    def test_partitions_reconstruct_counts_sse_and_rmse(self):
        torch.manual_seed(8)
        m = (torch.rand_like(self.y) > .2).float()
        p = self.y + torch.randn_like(self.y)*3
        with tempfile.TemporaryDirectory() as temp:
            report = boundary.BoundaryReport(Path(temp)/"report")
            report.update({"model": p}, self.y, m, ["map_58"], ["58"], 1)
            result = report.finish({}, {}, [])
        rows = {r["region"]: r for r in result["regions"]}
        whole = rows["all_valid"]
        for threshold in (5, 10, 20):
            for radius in (1, 3, 5):
                a, b = (rows[f"{part}_t{threshold}_r{radius}"] for part in ("edge", "nonedge"))
                self.assertEqual(a["valid_pixels"]+b["valid_pixels"], whole["valid_pixels"])
                self.assertAlmostEqual(a["sse_db2"]+b["sse_db2"], whole["sse_db2"], places=8)
                rebuilt = ((a["sse_db2"]+b["sse_db2"])/(a["valid_pixels"]+b["valid_pixels"]))**.5
                self.assertAlmostEqual(rebuilt, whole["rmse_db"])
        self.assertAlmostEqual(rows["near_missing_r5"]["sse_db2"]+rows["away_missing_r5"]["sse_db2"], whole["sse_db2"])

    def test_empty_regions_and_invalid_fills_are_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            report = boundary.BoundaryReport(Path(temp)/"report")
            report.update({"model": self.y*float("nan")}, self.y*float("nan"),
                          torch.zeros_like(self.mask), ["empty_58"], ["58"], 1)
            result = report.finish({}, {}, [])
            self.assertTrue(all(r["rmse_db"] is None and r["sse_db2"] == 0 for r in result["regions"]))
            with self.assertRaises(FileExistsError):
                boundary.BoundaryReport(Path(temp)/"report")

    def test_profiles_use_only_target_and_valid_support(self):
        target = self.y[0, 0].numpy()
        mask = self.mask[0, 0].numpy() > 0
        profiles = boundary.select_profiles(target, mask)
        self.assertEqual(len(profiles), 2)
        self.assertEqual(profiles, boundary.select_profiles(target.copy(), mask.copy()))
        self.assertEqual(profiles[0]["col"], 15)
        mask[:, 15] = False
        self.assertEqual(boundary.select_profiles(target, mask), [])


class ProtocolTests(unittest.TestCase):
    def checkpoints(self):
        args = vars(train.build_parser().parse_args(["--phase", "secondU", "--epochs", "150"]))
        state = {"layer.weight": torch.ones(1), "Wlayer.weight": torch.ones(1)}
        return ({"args": {**args, "phase": "firstU"}, "model": state},
                {"args": args, "model": {k: v.clone() for k, v in state.items()}})

    def test_protocol_rejects_wrong_firstU_and_unknown_history(self):
        first, second = self.checkpoints()
        del second["args"]["grad_weight"]  # legacy checkpoint
        settings = ablate_boundary.historical_settings(first, second)
        self.assertEqual(settings["epochs"], 150)
        second["model"]["layer.weight"] += 1
        with self.assertRaisesRegex(ValueError, "firstU weights"):
            ablate_boundary.historical_settings(first, second)
        first, second = self.checkpoints()
        del second["args"]["amp"]
        with self.assertRaisesRegex(ValueError, "lacks settings"):
            ablate_boundary.historical_settings(first, second)

    def test_commands_share_settings_and_explicit_initialization(self):
        first, second = self.checkpoints()
        args = ablate_boundary.build_parser().parse_args([
            "--data-root", "data", "--first-ckpt", "first.pt",
            "--reference-second-ckpt", "second.pt", "--out-root", "new-run"])
        commands = ablate_boundary.build_commands(args, ablate_boundary.historical_settings(first, second))
        training = [r["command"] for r in commands if r["kind"] == "train"]
        self.assertEqual(len(training), 3)
        for cmd in training:
            self.assertEqual(cmd[cmd.index("--init-from")+1], str(Path("first.pt").resolve()))
            self.assertEqual(cmd[cmd.index("--epochs")+1], "150")
            self.assertEqual(cmd[cmd.index("--seed")+1], "0")
            self.assertNotIn("--resume", cmd)
            self.assertNotIn("--limit-batches", cmd)
        self.assertEqual(len({cmd[cmd.index("--out")+1] for cmd in training}), 3)
        common = []
        for cmd in training:
            copy = cmd.copy()
            for flag in ("--out", "--grad-weight"):
                i = copy.index(flag)
                del copy[i:i+2]
            common.append(copy)
        self.assertTrue(all(cmd == common[0] for cmd in common))

    def test_read_only_plan_and_execution_record(self):
        first, second = self.checkpoints()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            torch.save(first, root/"first.pt")
            torch.save(second, root/"second.pt")
            args = ablate_boundary.build_parser().parse_args([
                "--data-root", str(root/"data"), "--first-ckpt", str(root/"first.pt"),
                "--reference-second-ckpt", str(root/"second.pt"), "--out-root", str(root/"out")])
            with mock.patch.object(ablate_boundary.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                plan = ablate_boundary.main(args)
            run.assert_not_called()
            self.assertFalse((root/"out").exists())
            self.assertEqual(plan["status"], "planned")

            def fake_execute(command, **kwargs):
                if "--boundary-out" not in command:
                    return
                directory = Path(command[command.index("--boundary-out")+1])
                report = boundary.BoundaryReport(directory)
                y = torch.zeros(1, 1, 32, 32)
                y[..., 16:] = 40
                report.update({"secondU": y+2}, y, torch.ones_like(y), ["synthetic_58"], ["58"], 1)
                report.finish({}, {"rmse_db_masked": 2}, [])

            args.execute = True
            with mock.patch.object(ablate_boundary.subprocess, "run", side_effect=fake_execute) as run, contextlib.redirect_stdout(io.StringIO()):
                result = ablate_boundary.main(args)
            self.assertEqual(run.call_count, 7)
            self.assertEqual(result["status"], "complete")
            self.assertTrue(all(row["delta_rmse_vs_grad0"] == 0 for row in result["comparison"]))
            self.assertTrue((root/"out"/"comparison.csv").exists())
            with self.assertRaises(FileExistsError):
                ablate_boundary.main(args)


def synthetic_dataset(root):
    root.mkdir()
    (root/"metadata.json").write_text(json.dumps({"heightmap_range_m": [0, 496], "pathloss_range_db": [20, 228]}))
    for split, count in (("train", 2), ("val", 3)):
        with (root/f"{split}_index.csv").open("w", newline="") as handle:
            fields = ["sample_id", "hm_row", "tx_row", "tx_col"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for i in range(count):
                writer.writerow(dict(sample_id=f"synthetic_{split}_{i}", hm_row=i, tx_row=128, tx_col=128))
        hm = np.broadcast_to(np.linspace(0, 300, 256, dtype=np.float32), (count, 256, 256)).copy()
        y = np.full((count, 256, 256), 80, dtype=np.float16)
        y[:, :, 128:] = 130
        m = np.ones((count, 256, 256), dtype=np.uint8)
        m[:, :10, :10] = 0
        y[m == 0] = 145
        np.save(root/f"{split}_hm.npy", hm)
        np.save(root/f"{split}_pl58.npy", y)
        np.save(root/f"{split}_mask58.npy", np.packbits(m.reshape(count, -1), axis=1))


class IntegrationTests(unittest.TestCase):
    def test_train_freezes_firstU_and_evaluate_reuses_exact_predictions(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root/"data"
            synthetic_dataset(data)
            initial = RadioWNet(inputs=2, phase="firstU")
            # Ensure a live final ReLU in this synthetic optimization check.
            with torch.no_grad():
                initial.Wconv_up000[0].bias.fill_(.1)
            state = {k: v.clone() for k, v in initial.state_dict().items()}
            first_path = root/"first.pt"
            torch.save({"model": state, "args": {"phase": "firstU", "band": "58"}}, first_path)
            args = train.build_parser().parse_args([
                "--data-root", str(data), "--phase", "secondU", "--grad-weight", ".03",
                "--init-from", str(first_path), "--device", "cpu", "--no-amp",
                "--num-workers", "0", "--batch-size", "2", "--epochs", "1",
                "--limit-batches", "1", "--panels", "0", "--out", str(root/"run"), "--run-name", "synthetic"])
            with contextlib.redirect_stdout(io.StringIO()):
                train.main(args)
            ckpath = root/"run"/"synthetic_best.pt"
            ck = torch.load(ckpath, map_location="cpu", weights_only=False)
            self.assertEqual(first_stage_hash(state), first_stage_hash(ck["model"]))
            self.assertTrue(any(not torch.equal(state[k], v) for k, v in ck["model"].items() if k.startswith("W")))
            self.assertEqual(ck["args"]["grad_weight"], .03)
            with (root/"run"/"synthetic_loss_components.csv").open() as handle:
                components = next(csv.DictReader(handle))
            self.assertAlmostEqual(float(components["train_total"]), float(components["train_mse"])+.03*float(components["train_gradient"]), places=7)

            model = RadioWNet(inputs=2, phase="secondU")
            calls = []
            model.register_forward_hook(lambda *unused: calls.append(1))
            args = evaluate.build_parser().parse_args([
                "--ckpt", str(ckpath), "--data-root", str(data), "--device", "cpu", "--no-amp",
                "--num-workers", "0", "--batch-size", "2", "--out", str(root/"val.json"),
                "--boundary-out", str(root/"boundary"), "--panels", "8"])
            with mock.patch.object(evaluate, "RadioWNet", return_value=model), contextlib.redirect_stdout(io.StringIO()):
                evaluate.main(args)
            self.assertEqual(len(calls), 2)  # exactly ceil(3/2), never re-infer for plots
            result = json.loads((root/"val.json").read_text())
            report = json.loads((root/"boundary"/"summary.json").read_text())
            self.assertEqual(report["samples"], 3)
            self.assertEqual(len(report["examples"]), 3)
            total_sse, total_n = 0, 0
            for row, example in zip(result["per_sample"], report["examples"]):
                with np.load(root/"boundary"/example["npz"]) as arrays:
                    valid = arrays["mask"] > 0
                    error = (arrays["prediction"]-arrays["target"])[valid].astype(np.float64)*float(arrays["pl_scale_db"])
                    n, sse = valid.sum(), np.square(error).sum()
                    total_sse += sse
                    total_n += n
                self.assertAlmostEqual((sse/n)**.5, row["rmse_db_masked"], delta=2e-5)
                self.assertAlmostEqual((sse/n)**.5, example["rmse_db"], places=10)
                self.assertTrue((root/"boundary"/example["png"]).exists())
            self.assertAlmostEqual((total_sse/total_n)**.5, result["summary"]["rmse_db_masked"], delta=2e-5)
            self.assertEqual(report["provenance"]["firstU_state_sha256"], first_stage_hash(state))
            self.assertTrue(report["provenance"]["data_metadata_and_actual_loader"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
