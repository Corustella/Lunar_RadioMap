# C: TX-conditioned scene attention on the fixed MaxViT baseline

Select `--arch lunar_maxvit_c`. This is experiment C from the B/C/G decision:
ordinary source-conditioned nonlocal feature aggregation, **without** the
TX-anchor-RX geometric attention bias proposed for G. G is not implemented.
The current B control is `lunar_maxvit` with AdamW (seed0 best 5.097016 dB).

## Evidence and research hypothesis

- Project evidence: B's high-change regions contain 30.89% of valid pixels but
  83.41% of remaining SSE. Fixed profiles include broad transitions, missed
  peaks and regional amplitude errors. FADE best was worse, at 5.128667 dB.
  These findings do not prove that nonlocal interactions are the bottleneck.
- Literature context: [Point2Radio](https://arxiv.org/html/2607.28994v1) uses
  TX-conditioned scene encoding and RX queries on indoor material point clouds
  at 3.5 GHz. [WiSER](https://arxiv.org/html/2606.04770v1) combines TX-conditioned
  scene memory with RX queries and link-corridor tokens on indoor scenes.
  Both are 2026 preprints; their inputs, protocols and metrics differ from
  lunar terrain regression. This module is not a reproduction of either paper,
  and no paper establishes its lunar RMSE benefit or originality.
- Hypothesis tested by C: one additional TX-conditioned content interaction
  improves prediction over the fixed MaxViT baseline. C does not test whether
  an explicit geometric bias provides additional benefit. MaxViT already has
  local and cross-region attention; it does not lack a global receptive field.

## Fixed implementation

After encoder block index 1, U has shape Bx128x32x32. A 2x2 average pool with
stride2 creates 256 scene anchors A (16x16), spanning the full scene. Read the
source feature s from U using the jointly augmented TX one-hot map:

```text
q_i = Wq [U_i, s] + bq
k_j = Wk [A_j, s] + bk
v_j = Wv [A_j, s] + bv
alpha_ij^h = softmax_j(q_i^h . k_j^h / sqrt(16))
U'_i = U_i + Wo concat_h(sum_j alpha_ij^h v_j^h) + bo
```

Four heads use 16 dimensions each (64 total). PyTorch scaled dot-product
attention has dropout0, no causal mask and no geometry bias. Q/K/V use default
Linear initialization; Wo and bo start at zero. No extra normalization, FFN,
position embedding, propagation gate, auxiliary loss or FADE is added.

The new aggregation branch (Q/K/V, attention and output projection) computes in
FP32 under AMP, then casts its update to the incoming feature dtype. A bounded
local batch16 check found Q/K task gradients still zero after 64 steps with FP16
aggregation; FP32 preserves these small gradients. The shared backbone retains
its original AMP behavior. This numerical implementation is recorded in the
model config and must also be matched by any future G control. It is not an
optimizer, loss, initialization or training-budget change.

The linear K source term is the same for every anchor and cancels in softmax;
source-dependent selection enters through Q, while V carries source context.
The concatenations retain the proposed C parameterization. Attention weights
are feature weights, not physical rays or reflection/power coefficients.

U' enters both the remaining encoder blocks and the existing 32x32 skip.
The five original convolution decoder stages, full-resolution branch, direct
normalized pathloss head and ReLU are preserved. All parameters train jointly
from scratch; no firstU or pretrained B checkpoint is frozen or loaded.

### TX sampling and initialization compatibility

The padded stride2 stem and first two downsampling blocks give a stride8
feature lattice centered at input pixel coordinates 0,8,...,248. The source is
bilinearly sampled at (tx_col/8, tx_row/8), with border extension for coordinates
beyond the last feature center. Grid construction and interpolation use FP32
under AMP; the result is cast to the feature dtype. A 255-pixel endpoint is not
rescaled to 31 by treating the feature map as a resized image. No target or
validity mask is read. The existing input contract is a single one-hot TX.

The added module is initialized after all shared parameters inside an isolated
CPU RNG context. Common weights and the subsequent CPU RNG state therefore
match B for the same seed. With zero Wo/bo, C initially returns the same feature
map and prediction. Q/K/V task gradients initially vanish until Wo changes;
this is the expected behavior of the zero residual initialization, not freezing.
CUDA stochastic execution and resume are not guaranteed bitwise reproducible.

Model config records the source sampling convention, head count, anchor pool,
initialization, insertion point and absence of geometry bias. Training, resume,
evaluation and inference timing use the existing factory and scripts. Strict
loading rejects B/FADE weights as a C initialization, or an altered C config.
Old B/FADE configs and state dictionaries remain compatible.

### Cost

C adds 57,664 parameters, for 35,375,853 total. With N=1024, M=256, d=64 and C=128,
attention costs O(B*N*M*d), with O(B*(N+M)*C*d) projections. Materialized scores
take O(B*4*N*M) memory; fused SDPA kernels may use less. There is no new
geometric cache, pairwise visibility tracing or external dependency. Actual
training memory and complete batch1 latency still need measurement on the
execution hardware, including live F, transfers, the whole model and dB output.

### Completed local checks (2026-10-18)

On Windows, RTX 4060 Laptop 8 GB, PyTorch 2.5.1+cu121 and torchvision
0.20.1+cu121:

- 48 TX coordinate checks passed. Independent explicit attention forward and
  backward matched the implementation (maximum parameter-gradient difference
  8.53e-14 in the double-precision reference check).
- B's shared initial weights, subsequent CPU RNG state and a real-input CUDA
  AMP initial prediction were bitwise identical before and after this change.
  C's zero residual initially preserved that prediction. Old B/FADE configs
  were unchanged; incompatible checkpoint/config loading was rejected.
- A bounded batch16 CUDA AMP check used one repeated real sample, not a
  representative training epoch. In 27 updates all gradients stayed finite;
  Q/K/V, output projection and selected encoder parameters had nonzero task
  gradients in the final two updates. Peak allocated memory was 4.54 GiB.
  Q/K gradients were still very small; this proves connectivity, not useful
  learning or removal of all backbone AMP underflow.
- Existing CLI training, resume, evaluation and diagnostic exports passed on
  the existing 2-train/2-val fixture. Pooled SSE/count reproduced its metric.
  These smoke checkpoints and scores are not formal experiment results.
- Random-weight batch1 inference on 16 distinct real val inputs, after three
  warmups, averaged 44.08 ms (p95 47.12 ms; max 47.59 ms), including live F,
  transfers, the full model/rho and CPU dB output. Including input reads gave
  44.23 ms mean; model startup and warmup were excluded. All observed samples
  were below 500 ms. This is neither server timing nor official certification.

Scripts, raw results, the earlier FP16 failure and its diagnosis remain in the
repository-external `../local_experiments/maxvit_c_20261018/`. No formal C
training or full-val accuracy comparison has been performed.

## Experiment protocol

Keep H/TX/F/rho, metadata scaling, official terrain-separated train/val,
masked MSE, D4, batch16, 150 epochs, AdamW0.05, LR1e-4/cosine to1e-6, AMP and
stochastic depth unchanged. No optimizer/loss/width grid is scheduled.
The data-directory `lunar_dataset.py` remains the preferred actual loader.

Run C seeds0/1 and analyze them together. B seed0 is already available; a
paired-seed conclusion also requires B seed1 under the same protocol. C1 must
not be compared only against B0 as if they were paired. Keep the current best.
Select by full-val pooled valid-pixel RMSE, not mean per-map RMSE or last epoch.
Also report the unchanged high-change region, terrain pairs, single/multi
groups and tails. A practical continuation gate is improvement in both paired
seeds, mean RMSE reduction >=0.10 dB, improved high-change error and acceptable
complete inference cost. This is a resource decision, not significance or a
promise of improvement. Existing val has repeatedly been used for development.

## Local commit and push

Only the following tracked implementation/run-document files belong in this
commit. Ignored docs and repository-external local QA stay local.

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git diff --stat
git add lunar_maxvit.py model_factory.py train.py LUNAR_MAXVIT.md LUNAR_MAXVIT_C.md
git diff --cached --stat
git commit -m "Add C control with TX-conditioned MaxViT scene attention"
git push origin main
```

## Server synchronization, timing and C seeds0/1

Run in the training environment you have confirmed and activated. These use
the previously established repository/data/cache paths; Codex has not executed
them on the server. B's recorded epoch times totaled about 3.45 hours per run;
C's server training duration is not yet measured. Allow several hours per seed
and estimate from its actual initial epoch timings, plus initialization/I/O.

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch main
git pull --ff-only

python benchmark_inference.py \
  --data-root ~/liuanda/LunarRM --arch lunar_maxvit_c \
  --band 58 --device cuda --samples 16 --warmup 3 \
  --out results/lunar_maxvit_c_v1/inference_random.json

for seed in 0 1; do
  python train.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --arch lunar_maxvit_c --band 58 --loss masked \
    --optimizer adamw --weight-decay 0.05 \
    --epochs 150 --batch-size 16 --lr 1e-4 \
    --lr-sched cosine --lr-min 1e-6 \
    --num-workers 8 --device cuda --amp --seed "$seed" \
    --init-from none --resume auto \
    --panels 6 --panels-every 5 --plots-every 5 \
    --out "runs/lunar_maxvit_c_v1/seed$seed" \
    --run-name lunar_maxvit_c_58_masked_single_los_distance_adamw || break

  python evaluate.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --ckpt "runs/lunar_maxvit_c_v1/seed$seed/lunar_maxvit_c_58_masked_single_los_distance_adamw_best.pt" \
    --split val --batch-size 16 --num-workers 8 --device cuda --amp \
    --out "results/lunar_maxvit_c_v1/seed$seed/val.json" || break
done
```

Inspect timing against the project's complete 500 ms budget before long runs.
Timing JSON paths must be new. A new training directory starts from scratch;
repeat the train command after interruption to resume its own last state.
Automatic curve, fixed-panel, best-panel and distribution exports are reused.
Smoke checkpoints must never initialize the formal experiment.

If the matching B-seed1 result does not already exist, its unchanged control is:

```bash
python train.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --arch lunar_maxvit --band 58 --loss masked \
  --optimizer adamw --weight-decay 0.05 \
  --epochs 150 --batch-size 16 --lr 1e-4 \
  --lr-sched cosine --lr-min 1e-6 \
  --num-workers 8 --device cuda --amp --seed 1 \
  --init-from none --resume auto \
  --panels 6 --panels-every 5 --plots-every 5 \
  --out runs/lunar_maxvit_adamw_v1/seed1 \
  --run-name lunar_maxvit_58_masked_single_los_distance_adamw

python evaluate.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --ckpt runs/lunar_maxvit_adamw_v1/seed1/lunar_maxvit_58_masked_single_los_distance_adamw_best.pt \
  --split val --batch-size 16 --num-workers 8 --device cuda --amp \
  --out results/lunar_maxvit_adamw_v1/seed1/val.json
```
