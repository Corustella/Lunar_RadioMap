# Lunar MaxViT: independent encoder/decoder

An experimental, single-stage replacement for the two-U-Net model. It is inspired
by [RMTransformer](https://arxiv.org/html/2501.05190v2), rather than a reproduction
of its urban-data experiment. A lower lunar validation RMSE has **not** yet been
demonstrated. The original `RadioUNet/modules.py` and LOS-distance baseline remain
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

## Commit the local implementation

The current worktree includes the previously authorized LOS-baseline restoration
and this new model. Review both before committing. The following PowerShell
commands stage tracked changes and the four new implementation files; ignored
`docs/` and repository-external experiments stay out of the commit.

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git diff --stat
git add -u
git add lunar_maxvit.py model_factory.py benchmark_inference.py LUNAR_MAXVIT.md
git diff --cached --stat
git commit -m "Restore LOS baseline and add lunar MaxViT encoder-decoder"
git push origin main
```

## Server sync, timing and smoke check

Run these on the SSH GPU server after confirming/activating its training
environment. These commands are provided for the user; Codex has not run them
on the server. Use new result/run paths. The repository/data paths below follow
the existing project convention.

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch main
git pull --ff-only

python benchmark_inference.py \
  --data-root ~/liuanda/LunarRM --band 58 --device cuda \
  --samples 16 --warmup 3 \
  --out results/lunar_maxvit_v1/inference_random.json
```

This benchmark needs no F cache: each distinct sample constructs F live, then
runs the whole network including rho and transfers a complete dB prediction to
CPU. It reports feature/network/output and total batch-one latency; disk input
reads are separately reported, including an input-read-plus-total column. Model
startup and warm-up are excluded. The JSON contains timing boundaries, metadata,
actual loader, environment and per-sample results. Random weights establish cost,
not accuracy. These measurements are not official certification or a worst-case
guarantee. The script rejects existing output paths.

Reuse the existing complete F cache. If the path below is not the existing cache
location, point both commands to its actual directory. The cache builder verifies
metadata/index and reuses complete splits; it never silently overwrites an
incomplete or mismatched cache. A new full cache costs substantially more than a
small smoke check.

```bash
python build_los_cache.py \
  --data-root ~/liuanda/LunarRM --cache-root results/los_cache_v1 \
  --splits train val --device cuda

python train.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --arch lunar_maxvit --band 58 --loss masked \
  --epochs 1 --batch-size 16 --limit-batches 1 \
  --num-workers 8 --device cuda --panels 0 --seed 0 \
  --out runs/lunar_maxvit_v1_smoke --run-name lunar_maxvit_smoke
```

Timing and the smoke check should take seconds to minutes depending on hardware
and storage. The smoke test verifies one real batch and one validation batch,
not an accuracy result. Do not resume its weights for formal training. If memory
or AMP fails, diagnose it before altering batch size or optimizer settings.
The full-model training duration is unknown until measured; do not infer it
from the old secondU timing.

## Formal paired run, after the checks pass

Only one architecture is proposed. Seeds 0/1 are trained and analyzed together,
each for a single 150-epoch phase, using the existing Adam, lr=1e-4, cosine to
1e-6, batch 16, AMP and D4 protocol. This is a new complete training configuration,
not a claim that these settings are optimal for all Transformers. The old model
used two stages, so this comparison does not isolate attention from capacity or
training cost. Do not add another grid of decoders/losses before reading results.

```bash
for seed in 0 1; do
  python train.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --arch lunar_maxvit --band 58 --loss masked \
    --epochs 150 --batch-size 16 --lr 1e-4 \
    --lr-sched cosine --lr-min 1e-6 \
    --num-workers 8 --device cuda --seed "$seed" \
    --out "runs/lunar_maxvit_v1/seed${seed}" \
    --run-name lunar_maxvit_58_masked_single_los_distance || break

  python evaluate.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --ckpt "runs/lunar_maxvit_v1/seed${seed}/lunar_maxvit_58_masked_single_los_distance_best.pt" \
    --split val --batch-size 16 --num-workers 8 --device cuda \
    --out "results/lunar_maxvit_v1/seed${seed}/val.json" || break
done
```

After training, repeat the batch-one benchmark with `--ckpt` and a new JSON path.
Primary comparison is the full official validation pooled valid-pixel RMSE in
dB. Both seeds, per-terrain SSE/count and the previously defined high-change
regions should be analyzed together. Small-sample interface tests and visual
sharpness cannot substitute for that score. Do not promise <4 dB from the urban
paper's reported relative improvement.

Sources: [RMTransformer](https://arxiv.org/html/2501.05190v2),
[MaxViT](https://arxiv.org/abs/2204.01697),
[author code](https://github.com/google-research/maxvit),
[torchvision implementation](https://docs.pytorch.org/vision/main/_modules/torchvision/models/maxvit.html).
