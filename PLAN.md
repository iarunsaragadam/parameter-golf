# Parameter Golf: Execution Plan

## The One Insight That Matters

The 4-hour non-record run reveals everything:
- Pre-quantization BPB: **1.1749**
- Post-quantization BPB: **1.2074**
- **0.0325 BPB is destroyed by naive int8 quantization**

The baseline we need to beat is 1.2244. We only need 0.005 improvement (→ 1.2194).
That 0.0325 quantization gap is **6.5× larger** than what we need to win.

This tells us the #1 priority: **don't lose so much to quantization.**

---

## What We're Actually Going to Build

A single modified `train_gpt.py` with three targeted changes. Not 138 scatter-shot
experiments — three changes, each validated, then combined.

---

## Step 1: Quantization-Aware Training (QAT)

**The problem:** The model trains in bf16, then gets brutally rounded to int8 at the end.
Weights that land between quantization levels get randomly shoved to the nearest one.
The model never learned to be robust to this.

**What we code:**
```python
# In the forward pass of each Linear layer, after step N:
def fake_quantize(weight):
    scale = weight.abs().amax(dim=-1, keepdim=True) / 127.0
    w_q = (weight / scale).round().clamp(-127, 127)
    # Straight-through estimator: forward uses quantized, backward uses original
    return weight + (w_q * scale - weight).detach()
```

We activate this in the **last 20% of training** (~last 2 minutes). The model learns
to arrange its weights so they quantize cleanly.

**Where in code:** Wrap the `fc`, `proj`, `qkv`, and `out_proj` weight matrices in
`CausalSelfAttention` and `MLP` classes. Add a global step counter that flips QAT on.

**Lines of code:** ~30
**Expected recovery:** 0.01–0.02 of the 0.0325 gap (conservative estimate)
**Runs to validate:** 3 (QAT at 80%, 70%, 60% of training)

---

## Step 2: Find the Right Model Shape

**The problem:** The baseline is 9 layers × 512 dim. Is that optimal for 16MB? Nobody
checked. The 16MB budget creates a tradeoff: wider models have more capacity per layer
but fewer layers. Deeper models compose better but each layer is weaker.

**What we try (no code changes — all env vars):**

| Config | Layers | Dim | Heads | KV Heads | MLP | Est. Size |
|--------|--------|-----|-------|----------|-----|-----------|
| Baseline | 9 | 512 | 8 | 4 | 2× | 15.8 MB |
| Wide | 7 | 640 | 8 | 4 | 2× | ~15.5 MB |
| Deep | 12 | 448 | 8 | 4 | 2× | ~15.6 MB |
| MQA-Wide | 8 | 576 | 8 | 1 | 2× | ~15.4 MB |
| Big-MLP | 9 | 512 | 8 | 4 | 3× | ~15.9 MB |
| Vocab-2k | 9 | 480 | 8 | 4 | 2× | ~15.7 MB |

MQA (multi-query attention, KV_HEADS=1) is the most interesting: it saves ~200K params
on KV projections, which we reinvest into width or an extra layer.

**Lines of code:** 0 (env var changes only)
**Runs to validate:** 6 configs × 1 run each = 6 runs
**What we learn:** The shape that gives the lowest BPB before quantization

---

## Step 3: Tune the Training Recipe

**The problem:** The baseline hyperparameters were hand-picked, not optimized. The Muon
optimizer's learning rate and the warmdown schedule are the two biggest levers.

**What we sweep (env vars only):**

Round A — Learning rate (4 runs):
```
MATRIX_LR ∈ {0.03, 0.05, 0.06, 0.08}   (baseline: 0.04)
```

Round B — Take best LR, sweep warmdown (3 runs):
```
WARMDOWN_ITERS ∈ {800, 1600, 2000}       (baseline: 1200)
```

Round C — Take best LR+warmdown, sweep batch (3 runs):
```
TRAIN_BATCH_TOKENS ∈ {262144, 786432}    (baseline: 524288)
TRAIN_SEQ_LEN ∈ {2048}                   (baseline: 1024)
```

**Lines of code:** 0
**Runs:** 10 (sequential, each informs the next)
**Why sequential:** LR interacts with batch size. Sweeping them jointly wastes runs.

---

## Step 4: Combine and Submit

Take the best from each step:
1. Best model shape (Step 2)
2. Best hyperparameters (Step 3)
3. Add QAT (Step 1)

Run this combination **5 times with different seeds** to get the mean and std needed
for statistical significance (p < 0.01).

**Runs:** 5 + 2 buffer for final tweaks = 7 runs

---

## Total Execution: 26 Runs

| Step | Runs | What | Code Changes |
|------|------|------|-------------|
| 1. QAT validation | 3 | Validate QAT at different activation points | ~30 lines |
| 2. Shape search | 6 | Find optimal depth/width/heads for 16MB | 0 lines |
| 3. HP tuning | 10 | LR → warmdown → batch size (sequential) | 0 lines |
| 4. Final combo + seeds | 7 | Combine winners, prove significance | 0 lines |
| **Total** | **26** | | **~30 lines** |

---

## The Execution Order (Day-by-Day)

### Day 1: Shape + QAT in parallel
- **Morning:** Launch 6 shape-search runs (can run in parallel if you have the nodes,
  or sequentially in 1 hour). While waiting, write the QAT code.
- **Afternoon:** Run 3 QAT validation runs on baseline shape. Analyze shape results,
  pick the winner.

### Day 2: HP tuning on best shape
- Run 10 HP sweeps sequentially on the winning shape (100 min of compute).
- Each round uses the winner from the previous round.

### Day 3: Combine + submit
- Apply QAT to the best shape + best HPs.
- Run 5 seeds for reproducibility.
- If BPB < 1.2194: submit.
- If not: use remaining 112 runs of budget for deeper exploration (ALBERT weight
  sharing, progressive training, vocab size changes).

---

## Why This Works (And Why Not Less Compute)

**Why 26 runs minimum (~4.3 hours of 8xH100, ~$95):**

The competition requires **5 reproducibility runs** just for the submission — that's
non-negotiable (50 min of compute). The remaining 21 runs are the minimum to avoid
flying blind:

- **Shape search can't be skipped.** The baseline shape was chosen arbitrarily.
  A wrong shape wastes everything else. 6 runs × 10 min = 1 hour to find the right
  foundation. Cutting this means gambling on the baseline shape being optimal.

- **HP tuning can't be skipped.** Learning rate is the single most impactful
  hyperparameter in deep learning. The wrong LR wastes the entire training budget.
  10 sequential runs (100 min) finds the right ballpark. Cutting this means gambling
  on the baseline LR being optimal for a different architecture.

- **QAT validation can't be skipped.** If QAT hurts instead of helps (wrong activation
  point, too aggressive), we need to know before combining. 3 runs (30 min) to validate
  the core thesis.

**Why we keep $400 in reserve:**

The 26-run plan is the *minimum viable submission*. If any step produces surprising
results (e.g., the wide model is way better, suggesting even wider might work), the
reserve lets us dig deeper. The reserve also covers:
- ALBERT weight sharing (if shape search shows depth matters more than width)
- Progressive training (if we're step-count limited, not parameter limited)
- Vocab size experiments (requires retokenization — expensive to get wrong)
- Second-order HP interactions we missed

**The $500 isn't "needed" — $95 is needed. The rest is insurance against surprises.**
