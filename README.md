# The First Lunar Pathloss Radio Map Prediction Challenge

Baseline code and reference tooling for the **ICASSP 2027 Grand Challenge** on
lunar radio map prediction.

Given a 256×256 lunar heightmap and a one-hot transmitter position, predict the
256×256 pathloss map in dB, at **415 MHz** and **5.8 GHz**. Ground truth comes
from Sionna RT ray-tracing over synthetic lunar topography at 1 m/px.

- **Data and leaderboard:** [Kaggle competition](https://www.kaggle.com/t/1e9cf1fa5c774010a6922e98abb60449)
- **Challenge site:** https://lunarradiomapchallenge.github.io

---

## Metric

**RMSE in dB over valid (ray-traced) pixels only**, computed separately per band
and weighted 50/50.

The ray tracer could not resolve every cell, for previous work the unresolved cells were
gap-filled. At 5.8 GHz with a static placeholder of exactly 145.0 dB, covering
about 11.4% of pixels. Those values are placeholders, not physics, and they are
excluded from scoring. Every training and validation map ships with a **validity
mask**: 1 = ray-traced, 0 = filled.

Train with a masked loss. Training on every pixel measurably improves the
all-pixel diagnostic while making the ranked metric _worse_ - the voids are
geometrically structured, so a model can score by learning where the tracer
failed instead of learning propagation.

You submit full 256×256 maps. The organizers apply the mask at scoring time.

---

## Quick start

Open `notebooks/lunar_starter.ipynb` (on Kaggle, in Google Colab, locally...), attach the competition
data, run it top to bottom. It loads the arrays, trains a small baseline, and
writes a correctly formatted `submission.csv`.

**Locally:**

```bash
python -m venv .venv && .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python train.py    --data-root <data> --band 415 --loss masked --phase firstU
python evaluate.py --ckpt runs/<run>_best.pt --data-root <data> --split val
```

`<data>` is wherever you unpacked the Kaggle download.

---

## What's here

| file                            | what it is                                                                |
| ------------------------------- | ------------------------------------------------------------------------- |
| `lunar_dataset.py`              | reference PyTorch dataloader                                              |
| `train.py`                      | trains the RadioUNet baseline, both stages                                |
| `evaluate.py`                   | scores a checkpoint on `val`, in dB, masked and unmasked                  |
| `score_submission.py`           | validates a submission directory; `--validate-only` needs no ground truth |
| `metrics.py`                    | RMSE / NMSE / PSNR / SSIM, with masked variants                           |
| `radiounet.py`                  | seam onto the vendored upstream model                                     |
| `lunar_maxvit.py`              | independent lunar-input MaxViT encoder and convolution decoder              |
| `model_factory.py`             | architecture selection and strict, backward-compatible checkpoint loading  |
| `benchmark_inference.py`       | batch-one timing including live physical feature construction               |
| `los_features.py`              | input-only terrain clearance, log-distance and cached features              |
| `build_los_cache.py`           | builds or reuses train/val/test LOS caches                                  |
| `run_los.py`                   | runs two seeds, each with firstU then secondU LOS training                   |
| `experiment_provenance.py`     | code version, data/loader paths, settings and environment records            |
| `RadioUNet/`                    | RadioUNet model code                                                      |
| `notebooks/lunar_starter.ipynb` | end-to-end starter, Kaggle-ready                                          |

---

## The data

Download from Kaggle and unpack at one root. It is **consolidated arrays**, one
file per split per field not one file per sample:

```
train_index.csv   train_hm.npy   train_pl415.npy   train_mask415.npy
val_index.csv     val_hm.npy     val_pl415.npy     val_mask415.npy
                                 ..._pl58.npy      ..._mask58.npy
metadata.json     README.md      lunar_dataset.py
```

**Row _i_ of `train_pl415.npy` is row _i_ of `train_index.csv`.** That ordering
is the contract. Three things are compressed out:

```python
# heightmaps are deduplicated -- 51 transmitters on a terrain share one
hm = np.load("train_hm.npy", mmap_mode="r")[int(row["hm_row"])]

# TX maps are not stored; a one-hot 256x256 is two integers
tx = np.zeros((256, 256), np.float32)
tx[int(row["tx_row"]), int(row["tx_col"])] = 1.0

# masks are bit-packed
mask = np.unpackbits(np.load("train_mask415.npy", mmap_mode="r")[i]).reshape(256, 256)
```

`lunar_dataset.py` does all of this for you. Always use `mmap_mode="r"` -
`train_pl415.npy` is 3.4 GB.

| split | terrains | samples/band | targets  | masks    |
| ----- | -------- | ------------ | -------- | -------- |
| train | 6,120    | 25,720       | ✅       | ✅       |
| val   | 680      | 2,330        | ✅       | ✅       |
| test  | 1,200    | 4,950        | withheld | withheld |

Splitting is at **terrain** level: a terrain with 51 transmitters contributes 51
samples sharing one heightmap, so a per-sample split would leak.

---

## Baseline

The current structural study compares `lunar_conv_tx_refine_unet` (U) with
`lunar_conv_tx_refine_pyramid` (P), seed0 each. Both keep T fixed and use the
same full second U down to 8x8. U predicts one correction; P reconstructs a
32x32 regional correction with zero-mean details at 64/128/256. The original
inputs, loss and 150-epoch settings are retained. See
[LUNAR_T_FULL_RECONSTRUCTION.md](LUNAR_T_FULL_RECONSTRUCTION.md) for the math,
checkpoint rules, and training/evaluation commands.

The preceding refinement control is `lunar_conv_tx_refine_frozen`, seed0 only.
It keeps the trained T parameters, BatchNorm buffers and stochastic layers fixed
in evaluation mode, and optimizes only the unchanged shallow refiner. Inputs,
masked MSE and the refiner's 150-epoch schedule stay unchanged. See
[LUNAR_T_REFINEMENT_FROZEN.md](LUNAR_T_REFINEMENT_FROZEN.md) for the single-seed
initialization, resume and server commands.

The preceding joint refinement study starts from trained T (`lunar_conv_tx_content`)
checkpoints and compares continued T with `lunar_conv_tx_refine`, seeds 0/1 each.
The latter adds a shallow 256/128/64 U with cross-stage decoder features and a
zero-initialized correction head. Both stages train jointly on the final masked
MSE. Initial validation is retained in best selection; both groups restart the
same 150-epoch optimizer/schedule. See [LUNAR_T_REFINEMENT.md](LUNAR_T_REFINEMENT.md)
for initialization rules, controls and server commands.

The preceding terrain-interaction study keeps C (`lunar_conv_aligned`) and compares
T (`lunar_conv_tx_content`) with TG (`lunar_conv_tx_geometry`), seeds 0/1 each.
Both add one source-conditioned interaction at 32x32; TG adds input-only
TX-ground-RX distance/direction/normal bias. Training and the decoder stay fixed.
See [LUNAR_TERRAIN_INTERACTION.md](LUNAR_TERRAIN_INTERACTION.md) for the definition,
physical limits, checkpoint compatibility and server commands.

The preceding spatial-aggregation study keeps C (`lunar_conv_aligned`) and adds
L (`lunar_conv_local3`, dilations 1/1/1) and M (`lunar_conv_multiscale`, 1/2/3).
Only one depthwise convolution at 32x32 is replaced; both variants keep C's
initial function and training recipe. Run seeds 0/1 for each, reusing C's results.
See [LUNAR_CONV_SPATIAL.md](LUNAR_CONV_SPATIAL.md) for controls and commands.
The completed A/B ablation has been retired from the active code; its historical
code and checkpoint support remain recoverable at Git commit `9a40414`.

The preceding alignment experiment uses `train.py --arch lunar_conv_aligned`:
a single U-shaped network matched to the MaxViT+AdamW baseline, retaining its
convolutions and pointwise MLPs while removing window/grid attention and SE.
It jointly trains on the same H/TX/F/log-distance inputs from scratch; its initial
run used **seed 0 only**. See [LUNAR_CONV_ALIGNED.md](LUNAR_CONV_ALIGNED.md) for the exact
comparison boundary and server commands. Existing architectures remain available.

The previous LOS+distance experiment uses `train.py --arch radiownet_biscale --phase secondU`
for two shared 64/128 decoder feedback rounds. It runs **seed 0 only**, with a
30-epoch review point and the original 150-epoch LR horizon. See
[BISCALE_RECONSTRUCTION.md](BISCALE_RECONSTRUCTION.md) for the model and commands.

The LOS+distance path-profile experiment is available as
`train.py --arch radiownet_path_profile --phase secondU`. It adds a shared GRU
over input-only terrain clearance profiles in one 64x64 refinement block.
The current experiment runs **seed 0 only**; see [PATH_PROFILE.md](PATH_PROFILE.md)
for the fixed initialization, cost definition and server commands.

An experimental, jointly trained encoder/decoder is available with
`train.py --arch lunar_maxvit --los-cache <existing-F-cache>`. It keeps the
LOS/distance definitions and adds a full-resolution terrain/TX decoding route.
It is a separate model; its default phase is `single`, and evaluation restores
the architecture from the checkpoint. See [design, validation and server
commands](LUNAR_MAXVIT.md). No lunar accuracy improvement is established yet.

RadioUNet, trained with a masked loss. Reference numbers on the
withheld test set, over valid pixels:

| model                    | 415 MHz | 5.8 GHz | combined (50/50) |
| ------------------------ | ------- | ------- | ---------------- |
| two band-specific models | 4.66 dB | 6.28 dB | **5.47 dB**      |
| one conditional model    | 4.76 dB | 6.42 dB | **5.59 dB**      |

Two stages: `--phase firstU` trains the coarse U-Net, `--phase secondU` the
refinement on top of it (`--init-from auto` picks up the matching firstU run).

### LOS inputs in both stages

The baseline architecture and MSE training follow reference commit `bf60377`.
Without feature flags, baseline checkpoints and the original two-stage workflow
remain supported. The upstream `RadioUNet/modules.py` is unchanged.

LOS uses the minimum terrain clearance along the TX-to-RX line, encoded with a
fixed asinh scale from training metadata. It is a continuous geometric proxy,
not a simulator LOS label. Feature construction reads only height and TX;
height, TX, F, labels and masks receive the same augmentation.

| mode | firstU input | secondU input |
| --- | --- | --- |
| baseline | height, TX | out1, height, TX |
| historical secondU LOS | height, TX | out1, height, TX, F, 0, 0 |
| both-stage LOS | height, TX, F | out1, height, TX, F, 0, 0 |
| LOS + distance refinement | height, TX, F | out1, height, TX, F, rho, 0 |

`--first-features los --second-features los` selects the new mode. Initializing
from a historical LOS checkpoint adds 579 zero weights in firstU's three input
convolutions and preserves secondU weights and hidden widths. Checkpoints record
the feature placement; evaluation restores it automatically. Existing LOS
caches use the same definition and can be reused.

One command runs **seeds 0 and 1 together**, each initialized from its existing
secondU LOS best checkpoint. It trains firstU for 150 epochs, freezes its best
weights, trains secondU for 150 epochs, then scores the full official val. The
optimizer and schedule start fresh for each stage; batch size, learning rate,
AMP and augmentation settings are inherited from the source checkpoints.

```bash
python run_los.py --data-root <data> --cache-root results/los_cache_v1 \
  --init-ckpts <seed0_LOS_best.pt> <seed1_LOS_best.pt> \
  --seeds 0 1 --out-root runs/los_both_v1 --device cuda --num-workers 8
```

Use a new output directory. The runner writes commands, paths, settings, timing
and per-seed RMSE comparisons to `manifest.json` and `comparison.json`; the mean
is a run average, not a prediction ensemble. Adding `--dry-run` prints the plan
without constructing caches or starting training. The four training stages add
600 epochs; firstU timing must be measured rather than inferred from secondU.

Gradient loss, boundary reports, coordinate ablations and hash audits are removed
from the active code. Archived experiments and temporary checks stay outside
the repository. Research records in `docs/` also stay local and Git-ignored;
do not force-add them.

### LOS + log-distance in secondU

`--first-features los --second-features los_distance --phase secondU` keeps
firstU frozen and fills an existing zero channel in secondU with
`rho = log1p(d3/1m) / log1p(d_max/1m)`. The grid distance `d3` uses height plus
the TX/RX antenna heights, all in metres; `d_max` comes from training metadata
height bounds, pixel resolution and grid extent. Distance is computed in FP32
after the joint augmentation, using only height and TX. It takes O(BHW) time
and extra memory and adds no model parameters relative to both-stage LOS.

Distance encoding and the final secondU layout are saved separately from the
original LOS cache definition, so existing caches remain usable. Initializing
with `--init-from` a LOS checkpoint copies the same model weights; evaluation
restores the distance mode automatically. Resume requires the same input mode.

Run seeds 0 and 1 from their corresponding **both-stage LOS firstU best**
checkpoints, with fresh Adam and cosine state. Use the existing secondU LOS
results as the reference; no firstU retraining or repeated LOS control is
needed. This is an input representation hypothesis whose benefit must be
measured on full-val masked RMSE.

```bash
set -e
out_root=runs/los_distance_v1
mkdir "$out_root"
for seed in 0 1; do
  python train.py --data-root <data> --los-cache results/los_cache_v1 \
    --band 58 --phase secondU --loss masked \
    --first-features los --second-features los_distance \
    --init-from "runs/los_both_v1/seed${seed}/firstU/radiownet_58_masked_firstU_los_both_best.pt" \
    --seed "$seed" --epochs 150 --batch-size 16 \
    --lr 1e-4 --lr-sched cosine --lr-min 1e-6 --clip 0 \
    --amp --device cuda --num-workers 8 --panels 0 \
    --out "$out_root/seed${seed}/secondU" \
    --run-name radiownet_58_masked_secondU_los_distance_both
  python evaluate.py --data-root <data> --los-cache results/los_cache_v1 \
    --ckpt "$out_root/seed${seed}/secondU/radiownet_58_masked_secondU_los_distance_both_best.pt" \
    --band 58 --phase secondU --split val --batch-size 16 \
    --amp --device cuda --num-workers 8 --panels 0 \
    --out "$out_root/seed${seed}/secondU/val.json"
done
```

---

## Submitting

**Kaggle leaderboard.** One row per pixel:

```
ID,PL
pl415_00047_00_3,88.19
```

`ID` is `pl<band>_<sample_id>_<flat_pixel_index>`, row-major over 256×256
(`arr.reshape(-1)` order). Start from `sample_submission.csv` and overwrite `PL`.

> This leaderboard is a **format checker and plays no part in the final
> ranking.** It scores a subset of the _validation_ split, whose ground truth
> ships with the data, anyone can submit the true values and score 0.00. Use it
> to prove your pipeline works.

**Final Submission in December.** Not a CSV. One `.npy` per test sample per
band (`pl415_<sample_id>.npy`, `pl58_<sample_id>.npy`), each 256×256 in dB, plus
your trained model and inference code. Inference must stay under **500 ms** per
map. Check your directory before sending:

```bash
python score_submission.py --submission <dir> --data-root <data> --band both --validate-only
```

---

## Licences

- **This code:** MIT - see `LICENSE`.
- **`RadioUNet/`:** MIT, Copyright (c) 2019 Ron Levie. Unmodified upstream model
  code; see `RadioUNet/README.md` for the source and citation.
- **The dataset:** CC BY 4.0, distributed via Kaggle.

If you modify or redistribute that model code, keep its MIT notice intact.

## Citing

If you use this dataset or build on this challenge, cite **RadioLunaDiff**, the
work this dataset comes from:

```bibtex
@inproceedings{radiolunadiff2026,
  title     = {{RadioLunaDiff}: Estimation of Wireless Network Signal Strength
               in Lunar Terrain},
  author    = {Torrado, P. and Pearson, A. and Klein, J. and
               Moscibroda, A. and Smith, J.},
  booktitle = {ICASSP 2026 -- 2026 IEEE International Conference on Acoustics,
               Speech and Signal Processing (ICASSP)},
  pages     = {21231--21235},
  year      = {2026}
}
```

P. Torrado, A. Pearson, J. Klein, A. Moscibroda, and J. Smith, "RadioLunaDiff:
Estimation of wireless network signal strength in lunar terrain," in \_ICASSP 2026

- 2026 IEEE International Conference on Acoustics, Speech and Signal Processing
  (ICASSP)\_, 2026, pp. 21231–21235.

Two secondary references, if relevant to what you are reporting:

- **The baseline model** is RadioUNet (Levie et al., 2021) - citation in
  `RadioUNet/README.md`.
- **The ground truth** was generated with Sionna RT: J. Hoydis, S. Cammerer,
  F. Ait Aoudia, M. Nimier-David, L. Maggi, G. Marcus, A. Vem, and A. Keller,
  "Sionna," 2022.
