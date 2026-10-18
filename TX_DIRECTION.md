# TX direction sampling on the fixed LOS+distance RadioWNet

Use `--arch radiownet_tx_direction --phase secondU --first-features los
--second-features los_distance`. This candidate changes one convolution unit
in the existing U-Net refinement. It does not use the MaxViT computational graph.

The complete MaxViT implementation is preserved on `research/maxvit` at
`5afd09c5e7eb27ee3ca8010f97774b9b3969f246`. Development is on
`research/tx-direction-unet`; it retains the common training/evaluation tools
and old architecture loaders. `main` is unchanged by this branch work.

## Evidence and hypothesis

Project evidence: the existing LOS+distance two-stage runs achieved valid-pixel
RMSE 5.416208 / 5.450843 dB for seeds 0/1. The fixed high-change region accounts
for about 83% of residual SSE in previous audits. Standard spatial convolutions
do not explicitly orient their sampling frame towards TX. These facts do not
establish direction sampling as the cause of the current errors.

[Deformable ConvNets, ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.html)
provides an image detection/segmentation precedent for adapting sampling
geometry. This operator is not the paper's learned per-pixel offset network;
TX-oriented sampling, tied tangential pairs and its lunar application are
research hypotheses. No lunar accuracy benefit or algorithmic novelty is
established by that paper or by the local checks below.

The hypothesis is that one source-oriented local interaction can improve
prediction over the exact existing LOS+distance protocol. The weights are
feature coefficients, not reflection coefficients, paths or complex fields.
There is no imposed radial monotonicity, hard LOS/NLOS gate or diffraction loss.

## One modified unit

`Wlayer20.0` is the original 3x3, padding-1 convolution at 32x32, mapping
60 channels to 70. Its weights/bias, original ReLU and the upstream forward
are preserved. A directional residual is added before that ReLU:

```text
Y = Conv3x3(X) + P A(X, TX)

e_r(p) = (p - t) / max(||p - t||, 1e-6)
e_t(p) = (-e_r,y(p), e_r,x(p))
r_r, r_t = 2 + 1.5 tanh(theta_r, theta_t)

S_a,0(p) = X(p + a r_r e_r(p))
S_a,pair(p) = [X(p + a r_r e_r(p) + r_t e_t(p))
                + X(p + a r_r e_r(p) - r_t e_t(p))] / 2
A_c(p) = sum_{a in {-1,0,1}} [w_a,0,c S_a,0,c(p)
                              + w_a,pair,c S_a,pair,c(p)]
```

P is a bias-free 1x1 projection from 60 to 70 channels. The nine samples use
bilinear interpolation and border extension. There are six independent
depthwise coefficients per channel: inward/center/outward radial rows each
have a center coefficient and a shared tangential-pair coefficient.

Two global learned scales are initialized to 2 feature cells (16 input pixels,
16 m for this data) and bounded to (0.5, 3.5) cells. This is a constrained
anisotropic sampling operator; it intentionally has no independent offset
prediction at every pixel, multiple-radius bank, attention or extra loss.

The nominal feature centers after three stride-2 max pools are input pixels
3.5, 11.5, ..., 251.5. The jointly augmented TX location is converted as
`t_xy = (tx_xy - 3.5) / 8`, retaining fractional coordinates. It is not snapped
to the nearest feature or rescaled by matching image endpoints. At an exactly
coincident TX/feature center both directions are zero and all samples coincide.
The TX context is scoped to one forward call and cleared even on exceptions;
no source tensor is checkpointed or reused across batches.

Tangential pairs are averaged before applying their shared coefficient.
Rotations transport the radial frame; a reflection swaps the two tangential
samples. Consequently the sampling branch respects D4 spatial transforms
(up to numerical interpolation error) when its input feature maps transform
as scalar fields. The original learned convolutions do not guarantee that
property: this is not a claim that the entire network is D4 equivariant.

Depthwise coefficients initialize with normal std 1/sqrt(6), scale parameters
with zero, and P with zero. The shared model state and subsequent CPU RNG state
are preserved. Both initial network outputs match the historical baseline.
At initialization only P receives a task gradient through this residual;
the depthwise coefficients and scales start learning after P changes.

The new branch computes in FP32 under AMP and casts its update to the base
convolution's dtype. No shared convolution, optimizer or AMP setting changes.
All new parameter names begin with W, so the existing phase rule freezes
firstU and trains secondU plus the operator. Legacy tensor keys stay intact;
initialization adds only the three new parameter tensors. Resume/evaluation
require the exact new architecture, physical features and model config.

## Fixed protocol and complexity

The baseline is the existing two-stage RadioWNet:
`out1 = U1(H, TX, F)` and `out2 = U2(out1, H, TX, F, rho, 0)`.
Output 2 remains a direct prediction with ReLU. The historical initialization
checkpoint contains both trained firstU and inherited secondU weights.

Formal runs use the same per-seed `los_both_v1/.../firstU/*_best.pt` starting
weights as the original LOS+distance experiment. FirstU stays frozen and
secondU trains for 150 epochs with Adam (weight decay 0), LR1e-4/cosine to
1e-6, batch16, AMP, D4 and masked MSE. Data-directory loader precedence,
training metadata scaling, official split and valid-pixel metric are unchanged.
Do not initialize this formal comparison from the distance best and add another
150 epochs: that would change the training budget. `--resume auto` resumes
only this candidate's own last state; `--init-from` initializes weights only.

The operator adds 4,562 parameters: 360 depthwise coefficients, two radii and
4,200 projection weights. Total parameters are 13,281,287; secondU trainable
parameters are 2,359,303 (the baseline has 2,354,741).

For K=9, the added time is O(BHW(K*Cin + Cin*Cout)); sampled activations use
O(BHW*K*Cin) memory. It uses PyTorch only and needs no new cache or dependency.
Full timing includes live F, rho, both U-Nets, transfers and CPU dB output.

Analyze seeds 0/1 together against the corresponding existing U baselines.
Continue only for consistent paired gains, a practically useful mean RMSE
reduction (the project's 0.10 dB gate), high-change error improvement and
acceptable complete cost. This is a resource criterion, not significance.
If successful, a same-budget Cartesian sampling control can test whether
TX orientation matters beyond added capacity; it is not included in this run.
Keep MaxViT+AdamW as a separate stronger performance reference. Reused val is
development evidence, not an independent generalization test.

## Local verification

Independent explicit bilinear interpolation and parameter/input gradients
agree with the sampling implementation (FP64 max forward error 1.50e-15,
max checked gradient error 3.91e-14). All eight D4 transforms, fractional TX,
outside-feature-grid TX and coincident TX checks passed; the branch's maximum
D4 discrepancy was 2.78e-15 with nonzero projection weights.

Both outputs on a real CUDA AMP input were bitwise identical to the pre-edit
LOS+distance baseline after historical checkpoint initialization. A bounded
batch16 AMP check used one repeated real sample: projection gradients were
nonzero immediately, and directional weights/scales received finite nonzero
gradients after the first update. FirstU weights remained unchanged. Peak
allocated memory was about 1.87 GiB. This is implementation QA, not accuracy
or a representative training epoch. Raw checks stay outside Git in
`../local_experiments/tx_direction_20261018/`.

The existing train, resume and evaluate CLIs passed a 2-train/2-val fixture
check, including optimizer/scaler restoration, strict checkpoint configuration,
frozen firstU, pooled valid-pixel metrics and plot exports. The 1-to-2-epoch
schedule replay is interface QA only. Both documented seed commands were
parsed and checked against the fixed protocol.

On the local RTX 4060 Laptop GPU, random-weight batch-one inference over 16
distinct real inputs with 3 disjoint warmups took mean 81.18 ms, p95 102.18 ms
and max 108.18 ms. This includes live F, rho, both stages, transfers and CPU dB
output; input reads are reported separately. This measures the complete new
path, not its incremental cost versus the baseline. It excludes model loading
and is neither a server timing result nor an official performance guarantee.

## Local development and branch publication

The archive branch records the complete prior MaxViT implementation. Keep
subsequent TX sampling changes on the U-Net branch. To commit and publish a
local revision, use the existing workflow below; do not add ignored docs or
repository-external QA. Skip the commit command if the revision is committed.

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git switch research/tx-direction-unet
git diff --check
git add tx_direction.py radiounet.py model_factory.py train.py benchmark_inference.py TX_DIRECTION.md
git commit -m "Add TX-oriented sampling to LOS-distance RadioWNet"
git push -u origin research/maxvit
git push -u origin research/tx-direction-unet
```

## Server synchronization and runs

Activate your confirmed training environment first. These use the established
repository/data/cache/checkpoint paths. Codex has not executed them on the
server. The original secondU runs took roughly 2.8-3.7 hours in recorded epoch
time per seed; allow several hours each and use the new run's early epoch
timings for its estimate. The two seeds run sequentially.

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git fetch origin
git switch research/tx-direction-unet
git pull --ff-only

python benchmark_inference.py \
  --data-root ~/liuanda/LunarRM --arch radiownet_tx_direction \
  --band 58 --device cuda --samples 16 --warmup 3 \
  --out results/tx_direction_v1/inference_random.json
```

Inspect complete timing before the long runs; output JSON must use a new path.

```bash
for seed in 0 1; do
  python train.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --arch radiownet_tx_direction --phase secondU --band 58 --loss masked \
    --first-features los --second-features los_distance \
    --optimizer adam --weight-decay 0 \
    --epochs 150 --batch-size 16 --lr 1e-4 \
    --lr-sched cosine --lr-min 1e-6 \
    --num-workers 8 --device cuda --amp --seed "$seed" \
    --init-from "runs/los_both_v1/seed$seed/firstU/radiownet_58_masked_firstU_los_both_best.pt" \
    --resume auto --panels 6 --panels-every 5 --plots-every 5 \
    --out "runs/tx_direction_v1/seed$seed/secondU" \
    --run-name radiownet_58_masked_secondU_tx_direction || break

  python evaluate.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --ckpt "runs/tx_direction_v1/seed$seed/secondU/radiownet_58_masked_secondU_tx_direction_best.pt" \
    --split val --batch-size 16 --num-workers 8 --device cuda --amp \
    --out "results/tx_direction_v1/seed$seed/val.json" || break
done
```

Download both seed directories, full-val JSON/CSV and timing together. Analyze
their best checkpoints using pooled valid-pixel RMSE and the unchanged region
protocol. Smoke checkpoints are not formal experiment initializations.
