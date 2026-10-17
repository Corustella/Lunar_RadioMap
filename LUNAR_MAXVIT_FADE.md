# LunarMaxViT + FADE: seed-0 experiment

This candidate replaces the final three upsampling/fusion units with FADE L2H.
It is a controlled lunar adaptation, not a reproduction of the FADE paper's
whole network. The encoder, decoder, guidance branch and output head all train
together from scratch. Select it with `--arch lunar_maxvit_fade`.

## Evidence and intended mechanism

The completed AdamW run achieved 5.097016 dB best validation RMSE, only 0.015122 dB
better than Adam. High-change regions still account for 83.41% of remaining SSE.
Fixed profiles include wide transitions, missed spikes and regional amplitude
errors. These observations motivate testing joint coarse/fine reconstruction;
they do not establish that upsampling is the causal bottleneck.

[FADE](https://arxiv.org/html/2407.13500v1) studies joint encoder/decoder-guided
upsampling in visual dense prediction, including BTS on NYU Depth V2. Its
continuous depth-regression results support the mechanism as a candidate.
Lunar pathloss has a different input and physical generation process, so a
reduction in lunar dB RMSE remains a research hypothesis. The learned gate and
kernel weights are feature weights, not reflection or transmission coefficients.

## Selected lunar adaptation

| Decoder output resolution | Guidance | Operation |
|---|---|---|
| 16, 32 | Existing encoder skips | Existing transposed convolution + concat + double convolution |
| 64 | Encoder block 1 | 1x1 projections to 64 channels, FADE, double convolution |
| 128 | Encoder stem | 1x1 projections to 32 channels, FADE, double convolution |
| 256 | Full-resolution physical-input branch | 1x1 projections to 16 channels, FADE, double convolution |

FADE uses the standard 64-dimensional embedding, shared 3x3 content encoder,
5x5 softmax reassembly kernels, nearest interpolation and coarse-feature gate.
The shared content encoder is applied separately to both branches, retaining
both applications of its bias as in the author implementation. The fusion is
`gate * guide + (1 - gate) * reassembled_coarse`. A second concatenation of the
same guide is not added. The channel projections are needed to make the two
branches compatible for gating. Projections and FADE convolutions use Xavier
uniform weights and zero biases; following double convolutions retain the
project's original initialization.

H/TX/F/rho, 256x256 geometry, full-map direct regression and the ReLU output
remain as before. Input contours are not imposed as pathloss edges. A target
or validity mask never enters inference. Both old RadioWNet and the old
`lunar_maxvit` architecture remain available.

The new model has **35,163,739** trainable parameters (old: 35,318,189).
AdamW decay applies to 34,991,396 Conv/Linear weight parameters; the remaining
172,343 bias/normalization/relative-position parameters have zero decay.

## Equivalent reassembly with bounded workspace

`fade.py` follows the author's L2H operator and includes its MIT notice in
`FADE_LICENSE`. Reassembly uses PyTorch reshape/einsum/unfold/fold rather than
adding mmcv or einops. Low-resolution rows are tiled; the four output subpixels
are contracted directly with their kernels, avoiding a full fourfold repeated
patch tensor. Backward saves the input and kernels and recomputes local patches.
The implementation supports first-order gradients, including ordinary AMP
training; double backward is unsupported.

For output HxW, C channels, batch B and kernel size K, reassembly time is
O(B*C*K^2*H*W). Saved input/kernel activations are O(B*(C+K^2)*H*W). Each unfolded
patch tensor has a 32 MiB target budget (at least one row); temporary contractions,
folds, output/kernel tensors and the rest of the model also consume memory.
This is not a claim that total workspace is only 32 MiB.

No new dependency is required. Sources:
[author L2H implementation](https://github.com/poppinace/fade/blob/main/FADE_L2H.py),
[author PyTorch implementation](https://github.com/poppinace/fade/blob/main/pytorch_implementation/FADE_L2H.py),
[license](https://github.com/poppinace/fade/blob/main/LICENSE).

## Compatibility and diagnostics

Checkpoints record the separate architecture identifier and exact model/input
configuration. Cross-architecture initialization/resume and modified FADE
configuration are rejected. Old MaxViT seeded weights, config and same-batch
forward have been checked against the previous code and are bitwise identical.
Existing `evaluate.py` selects the model automatically from its checkpoint.

The existing CSV history, periodic training curves (PNG/PDF/JSON), fixed panels,
best-epoch panels and sample RMSE distribution exports are reused. No training
metric, loss, normalization, dataset loader or split has been changed.

## Local validation

Independent CPU references verified neighborhood indexing, zero padding,
row tiling and feature/kernel gradients, including numerical first-order
gradcheck. Literal double-precision reference maximum errors were 2.67e-15
(forward) and 7.11e-15 (gradients).

On a local RTX 4060 Laptop, CUDA/AMP forward and backward reached the encoder,
full-resolution branch, new gate and output head. A repeated real-input batch
of 16 completed two training steps with finite gradients for every parameter;
peak allocated memory was 5,379,279,872 bytes (about 5.01 GiB). This is a memory
and gradient check, not a precision/generalization experiment.

The random-weight batch-one benchmark used 16 distinct terrain/TX samples and
3 distinct warmups, with live F, input transfer, all network operations/rho and
CPU dB output. Local total latency: mean **156.23 ms**, p95 **197.97 ms**, maximum
**212.78 ms**. Input disk reads are separately reported. This is local observed
cost, not server execution, a worst-case guarantee or official certification.

## Local commit and push

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git diff --stat
git add fade.py FADE_LICENSE lunar_maxvit.py model_factory.py train.py benchmark_inference.py LUNAR_MAXVIT.md LUNAR_MAXVIT_FADE.md
git diff --cached --stat
git commit -m "Add jointly trained FADE decoder for lunar MaxViT"
git push origin main
```

Ignored docs and repository-external local experiments are not committed.

## Server sync, timing and formal training: only seed 0

Use the already activated training environment. The following are commands for
the user, not actions executed by Codex on the server. The existing F cache is
reused. The old AdamW run's epoch times totaled about 3.45 hours; FADE training
duration has not been measured on that server. Allow several hours and use the
first few recorded epoch times to estimate the actual total.

Run the timing command once at its new output path. The local model test has
passed; it is not necessary to wait for another conversation before starting
the formal run. Inspect the server timing output against the same 500 ms scope.

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch main
git pull --ff-only

python benchmark_inference.py \
  --data-root ~/liuanda/LunarRM --arch lunar_maxvit_fade \
  --band 58 --device cuda --samples 16 --warmup 3 \
  --out results/lunar_maxvit_fade_v1/inference_random.json

python train.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --arch lunar_maxvit_fade --band 58 --loss masked \
  --optimizer adamw --weight-decay 0.05 \
  --epochs 150 --batch-size 16 --lr 1e-4 \
  --lr-sched cosine --lr-min 1e-6 \
  --num-workers 8 --device cuda --amp --seed 0 \
  --init-from none --resume auto \
  --panels 6 --panels-every 5 --plots-every 5 \
  --out runs/lunar_maxvit_fade_v1/seed0 \
  --run-name lunar_maxvit_fade_58_masked_single_los_distance_adamw

python evaluate.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --ckpt runs/lunar_maxvit_fade_v1/seed0/lunar_maxvit_fade_58_masked_single_los_distance_adamw_best.pt \
  --split val --batch-size 16 --num-workers 8 --device cuda --amp \
  --out results/lunar_maxvit_fade_v1/seed0/val.json
```

The new directory starts from scratch; repeat only the train command after an
interruption to resume this FADE run's last state. Do not initialize from the
old Adam/AdamW best. After completion the same benchmark can use the new best
checkpoint and a distinct output JSON path for trained-weight timing.

Compare new best against **5.097016 dB**, with the fixed official validation set
and existing high-change-region protocol. The practical gate remains at least
0.10 dB global improvement plus high-change-region improvement and the complete
inference budget. It is a research budget threshold, not statistical significance.
Only seed 0 is scheduled; no grid of alternative upsamplers, losses or extra
features is included. The previous best is retained regardless of the outcome.
