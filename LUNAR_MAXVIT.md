# Lunar MaxViT: independent encoder/decoder

An experimental, single-stage replacement for the two-U-Net model. It is inspired
by [RMTransformer](https://arxiv.org/html/2501.05190v2), rather than a reproduction
of its urban-data experiment. The completed seed-0 Adam run achieved **5.11214 dB**
on the fixed validation set, versus 5.41621 dB for LOS + distance seed 0. This
compares complete architectures/training budgets, not attention in isolation. The original `RadioUNet/modules.py` and LOS-distance baseline remain
available.

## Selective reuse and lunar adaptation

| Keep from MaxViT | Adapt for lunar maps |
| --- | --- |
| MBConv, local-window and cross-region grid attention, relative position bias | Four physical input channels rather than RGB |
| Tiny-scale widths 64/128/256/512 and depths 2/2/5/2 | Native 256x256 input, partition size 8 rather than the classifier's 7 |
| Official encoder GELU, BatchNorm (eps=0.001, momentum=0.01), stochastic depth to 0.2 | Remove the classification head; reconstruct a complete spatial regression map |
| Hierarchical features | Five convolution decoder stages plus a full-resolution terrain/TX branch |

External input is `[H, TX, F]`; the model appends the existing log-distance rho
after joint augmentation, giving `[H, TX, F, rho]`. H is the normalized continuous
terrain, not a binary building mask. F remains signed; no ImageNet standardization
or clipping is applied. The TX is one-hot. F and rho use the existing definitions,
training-metadata scales and terrain-following TX/RX heights. Both features are
available from test inputs alone. No target or validity mask enters the model.

The full-resolution 4-to-16-channel convolution branch and final raw-input skip
provide a short route for exact TX location and terrain/clearance contours. Their
benefit is a **research hypothesis**, not a demonstrated physical mechanism.
Attention weights are not physical rays. LOS is not used as a hard attention mask:
an obstructed direct path does not rule out reflected or other simulated paths.
No urban wall/material/antenna channels, extra diffraction model, FSPL residual,
auxiliary edge loss or old firstU prediction are introduced.

```text
H, TX, cached F                         3 x 256 x 256
       + rho from augmented H/TX       4 x 256 x 256
       |-- full-resolution CNN        16 x 256 x 256
       `-- MaxViT stem                 64 x 128 x 128
           stages                     64 x 64 x 64
                                     128 x 32 x 32
                                     256 x 16 x 16
                                     512 x 8 x 8
           CNN decoder + skips        256/128/64/32/16 channels
           raw physical-input skip
           Conv + ReLU                1 x 256 x 256
```

All **35,318,189** parameters train jointly, from scratch, in `phase=single`.
There is no frozen firstU. The output still directly predicts normalized
pathloss with ReLU; masked MSE, metadata normalization and D4 augmentation are
unchanged. The output head starts with small nonzero weights (normal std=0.001)
and fixed bias 0.5 to keep the ReLU active while allowing first-step upstream
gradients. This is an initialization choice, not a new loss or a data-derived
physical prior. No external pretrained weights are downloaded.

For each stage with N tokens, C channels, h heads and q=64 tokens per attention
group, attention/projections cost O(NqC + NC^2), with working memory roughly
O(Nqh + NC); convolution, decoder and saved training activations add cost.
Fixed groups avoid full-image O(N^2) attention. Actual latency and training memory
must be measured; parameter counts cannot establish the 500 ms budget.

## Existing interfaces

Use the existing `train.py` and `evaluate.py`. `--arch` defaults to `radiownet`;
existing baseline commands retain their stage behavior. For `lunar_maxvit`, the
default phase is `single`, F/rho are automatic, and the existing F cache is reused.
Do not pass RadioWNet's `--first-features` or `--second-features` flags. The first
version supports one band per model (`58` or `415`), not `band=both`.

Checkpoints record the complete model configuration and physical scales.
Evaluation selects the architecture automatically. Old checkpoints without an
architecture field default to RadioWNet. Cross-architecture loading, including
trying to initialize this model from an old LoS U-Net, is rejected. New-model
resume requires matching architecture/configuration/scales; `--init-from` only
loads weights, whereas `--resume` restores the existing training state.
Resume follows the repository's existing behavior and does not promise exact RNG
trajectory reproduction. AMP GradScaler state is saved/restored when available;
older checkpoints without it remain supported. The best score is now updated before writing `last`,
so resuming cannot lose a new best solely because that field lagged one epoch.

No extra package is needed beyond the existing torch, torchvision and NumPy.
Record the actual torch/torchvision/CUDA versions. The known torchvision
`meshgrid` future-interface warning is not a training error; do not change model
logic just to suppress it.

## Controlled AdamW experiment (2026-10-17)

The current decision is **one seed, seed 0**, from scratch, training the complete
encoder and decoder. Keep the model, H/TX/F/rho, D4 augmentation, masked MSE,
batch 16, 150 epochs, lr=1e-4, cosine floor 1e-6, AMP and stochastic depth unchanged.
Only replace Adam with AdamW and weight decay 0.05. No parameter scan is scheduled.

The existing run's best was epoch 45 (zero-based), 5.11214 dB; last was 5.50939 dB.
Training loss fell another 34.07% after best. This motivates a regularization test,
but does not prove that missing weight decay caused the validation degradation.
[AdamW](https://arxiv.org/abs/1711.05101) supports decoupled weight decay in adaptive
optimization; [MaxViT](https://arxiv.org/abs/2204.01697) used AdamW/0.05 in its
ImageNet classification recipe. Neither establishes lunar regression efficacy.
The intended improvement is a **research hypothesis**.

`--optimizer adam` and `--weight-decay 0` remain the defaults for old commands.
For AdamW, only Conv2d/ConvTranspose2d/Linear weights decay (35,146,228 parameters).
Biases, normalization parameters and 22 relative-position bias tables do not
(171,961 parameters). Betas remain (0.9, 0.999), eps 1e-8. Optimizer configuration
and ordered parameter names are saved in checkpoint, provenance and summary.
Resume checks the optimizer and decay groups before restoring state. Legacy Adam
checkpoints can resume as Adam, but cannot be silently resumed as AdamW. New
runs reject existing outputs unless resume is explicit. `--resume auto` rejects
partial output directories with no last checkpoint; use a new directory instead.

AdamW implements the additional update `w <- (1 - lr * weight_decay) * w` on the
selected parameters. Update time and optimizer-state memory remain O(P), where
P is parameter count. The inference graph and parameter count are unchanged.
The user's earlier random-weight server benchmark was mean 22.61 ms, p95 23.51 ms
for 16 samples, including live F and transfers. This is prior measured evidence,
not a new benchmark or official 500 ms certification.

## Automatically saved diagnostic artifacts

- `<run>_history.csv`: retains the existing train loss; adds accumulated online
  masked SSE/count, normalized MSE and dB RMSE. These use augmented training data,
  train mode and changing weights, so they are not a fixed-checkpoint train score.
  Old history rows leave the newly introduced measurements blank on resume.
- `<run>_curves.png/.pdf/.json`: objective, train-online/val pooled RMSE, learning
  rate; refreshed every five epochs and at completion (`--plots-every 0` disables).
  No smoothing or interpolation of missing measurements.
- `<run>_best_epoch<N>_panels.png`: separately loads best at completion, with the
  same AMP setting as validation. Periodic panels still show the latest snapshot.
  Invalid target/error pixels are gray; the model still outputs the full map.
- `val.json`: per-sample terrain_id, valid pixel count and masked SSE in dB^2,
  allowing later terrain aggregation without averaging per-image RMSE.
- `val_rmse_distribution.png/.pdf/.json`: equal-weight sample histogram and ECDF,
  with pooled pixel score labeled separately (`evaluate.py --no-plots` disables).
  Samples sharing terrain are correlated; no independence or significance claims.

Plots use the existing Matplotlib dependency; no additional inference is needed
for curves/distributions. Plotting costs O(E) for E epochs and O(N log N) for N
validation samples, with O(E) or O(N) auxiliary memory. The best panel export adds
one forward pass on the selected few samples, not a second full validation.

## Local commit and push

The changes are limited to optimizer configuration, logging/figures and this
run document. Ignored `docs/` and repository-external diagnostics stay local.
Run in local PowerShell after reviewing the diff:

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git diff --stat
git add train.py evaluate.py analysis_plots.py LUNAR_MAXVIT.md
git diff --cached --stat
git commit -m "Add controlled AdamW training and diagnostic figures"
git push origin main
```

## Server sync and formal seed-0 training

Use the already activated training environment (the previous prompt was `gc1`).
These commands are provided for the user; Codex has not run them on the server.
Reuse the existing F cache; the architecture/input definition has not changed.
No extra server smoke run is required for this proposal. Previous downloaded
training epoch times totaled about 3.42 hours; allow roughly 3-4 hours plus I/O
and load-dependent overhead, rather than treating this as a duration guarantee.

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch main
git pull --ff-only

python train.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --arch lunar_maxvit --band 58 --loss masked \
  --optimizer adamw --weight-decay 0.05 \
  --epochs 150 --batch-size 16 --lr 1e-4 \
  --lr-sched cosine --lr-min 1e-6 \
  --num-workers 8 --device cuda --amp --seed 0 \
  --init-from none --resume auto \
  --panels 6 --panels-every 5 --plots-every 5 \
  --out runs/lunar_maxvit_adamw_v1/seed0 \
  --run-name lunar_maxvit_58_masked_single_los_distance_adamw

python evaluate.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --ckpt runs/lunar_maxvit_adamw_v1/seed0/lunar_maxvit_58_masked_single_los_distance_adamw_best.pt \
  --split val --batch-size 16 --num-workers 8 --device cuda --amp \
  --out results/lunar_maxvit_adamw_v1/seed0/val.json
```

The new directory starts from scratch. Re-running the same training command
resumes only this AdamW run's last checkpoint; it does not resume the original
Adam experiment. Training retains the existing non-exact-RNG resume behavior.

Compare **new best with old best 5.11214 dB**, not with old last 5.50939 dB.
The practical next-step gate is at least 0.10 dB global improvement together with
improvement in the existing high-change regions (10 dB adjacent valid-pixel
change, Chebyshev radius 3, intersected with the official mask). This is a
budget decision threshold, not statistical significance. Only seed 0 is scheduled;
no cross-seed stability or hidden-test generalization conclusion is available.
If best does not materially improve, do not automatically scan weight decay or
launch further seeds. Existing best checkpoints are retained.

Sources: [RMTransformer](https://arxiv.org/html/2501.05190v2),
[MaxViT](https://arxiv.org/abs/2204.01697),
[author code](https://github.com/google-research/maxvit),
[torchvision implementation](https://docs.pytorch.org/vision/main/_modules/torchvision/models/maxvit.html),
[AdamW](https://arxiv.org/abs/1711.05101).
