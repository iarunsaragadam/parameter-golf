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

## What We're Building

A single modified `train_gpt.py` with three targeted changes, each validated
independently, then combined into a final submission.

---

## Step 1: Add Quantization-Aware Training (QAT)

**The problem:** The model trains in bf16, then gets brutally rounded to int8 at the
end. Weights that land between quantization levels get shoved to the nearest one.
The model never learned to be robust to this.

**What we do:**

Add a `fake_quantize` function that simulates int8 quantization during the forward
pass using a straight-through estimator (STE). The forward pass sees quantized weights,
but gradients flow through as if the weights were unquantized:

```python
def fake_quantize(weight):
    scale = weight.abs().amax(dim=-1, keepdim=True) / 127.0
    w_q = (weight / scale).round().clamp(-127, 127)
    return weight + (w_q * scale - weight).detach()
```

We activate this in the **last 20% of training** (~last 2 minutes). The model learns
to arrange its weights at values that quantize cleanly.

**Where in code:** Wrap the weight matrices in `CausalSelfAttention.forward()` and
`MLP.forward()` — specifically `qkv`, `out_proj`, `fc`, and `proj`. Add a global
training progress tracker that flips QAT on at the right step.

**What we validate:** Run 3 times with QAT activating at 60%, 70%, and 80% of training
to find the sweet spot. Compare post-quantization BPB against baseline.

**~30 lines of code.**

---

## Step 2: Find the Right Model Shape

**The problem:** The baseline uses 9 layers × 512 dim. Is that optimal for 16MB?
The 16MB budget forces a tradeoff: wider models have more capacity per layer but
fewer layers. Deeper models compose better but each layer is weaker.

**What we do:** Run 6 architecture configs, all via environment variables (zero code
changes). Each must fit under 16MB after int8+zlib compression:

| Config | Layers | Dim | Heads | KV Heads | MLP | Why Try It |
|--------|--------|-----|-------|----------|-----|------------|
| Baseline | 9 | 512 | 8 | 4 | 2× | Control |
| Wide | 7 | 640 | 8 | 4 | 2× | More capacity per layer |
| Deep | 12 | 448 | 8 | 4 | 2× | More compositional depth |
| MQA-Wide | 8 | 576 | 8 | 1 | 2× | Save KV params → reinvest in width |
| Big-MLP | 9 | 512 | 8 | 4 | 3× | More feedforward capacity |
| Vocab-2k | 9 | 480 | 8 | 4 | 2× | Fewer tokens/byte → faster training |

MQA (multi-query attention, KV_HEADS=1) is the most interesting — it saves ~200K
params on KV projections that we reinvest into model width.

**What we learn:** Which shape gives the lowest BPB. This becomes the foundation for
everything else.

---

## Step 3: Tune the Training Hyperparameters

**The problem:** The baseline hyperparameters were hand-picked, not optimized. The
Muon optimizer's learning rate and the warmdown schedule are the two biggest levers.

**What we do:** Three sequential rounds of sweeps on the winning shape from Step 2.
Each round uses the winner from the previous round:

**Round A — Learning rate:**
```
MATRIX_LR ∈ {0.03, 0.05, 0.06, 0.08}   (baseline: 0.04)
```
Pick the best. This is the single most impactful hyperparameter.

**Round B — Warmdown schedule:**
```
WARMDOWN_ITERS ∈ {800, 1600, 2000}       (baseline: 1200)
```
Controls how long the LR decays at the end. Longer warmdown = more time at peak LR.

**Round C — Batch size and sequence length:**
```
TRAIN_BATCH_TOKENS ∈ {262144, 786432}    (baseline: 524288)
TRAIN_SEQ_LEN ∈ {2048}                   (baseline: 1024)
```
Smaller batch = more gradient updates in 10 min. Longer sequences = better context.

**Why sequential:** LR interacts with batch size. Sweeping them jointly wastes runs.
Each round narrows the search space for the next.

---

## Step 4: Combine and Submit

Take the winners from each step:
1. Best model shape (Step 2)
2. Best hyperparameters (Step 3)
3. QAT from Step 1 (tuned activation point)

Run this final config **5 times with different seeds**. The competition requires
statistical significance (p < 0.01), so 5 seeds are mandatory.

If BPB ≤ 1.2194: package `submission.json`, `README.md`, `train.log`, and submit PR.

---

## Execution Order

### Day 1: Shape search + write QAT code
- Launch 6 shape-search runs (parallel if multi-node, or 1 hour sequential).
- While runs execute, write and test the QAT code locally.
- Analyze shape results. Pick the winning architecture.
- Run 3 QAT validation runs on the winning shape.

### Day 2: HP tuning
- Run 10 HP sweeps sequentially on winning shape (~100 min of compute).
- Round A (LR) → Round B (warmdown) → Round C (batch).
- Each round takes the best from the previous.

### Day 3: Combine + submit
- Apply QAT to best shape + best HPs. Verify improvement.
- Run 5 seeds for reproducibility stats.
- If target met: submit. If not: explore fallback ideas below.

---

## Fallback Ideas (If Target Not Met)

If the three main changes don't reach 1.2194, these are the next things to try:

**ALBERT-style weight sharing:** Share transformer block weights across layers.
E.g., 3 unique blocks repeated 4× = 12 effective layers at the parameter cost of 3.
Freed params go toward a wider model (dim 768+). ~15 lines of code.

**Progressive training:** Train a smaller model fast for the first 6 minutes (more
gradient steps), then expand to the full model via net2net-style weight copying for
the final 4 minutes. ~60 lines of code.

**Data curriculum:** Order training shards by difficulty (easy → hard). Score shards
by average loss from a baseline run, then train in ascending order. ~20 lines of code.

---

## Summary

| Step | What We Do | Code Changes |
|------|-----------|-------------|
| 1. QAT | Add fake int8 quantization with STE in last 20% of training | ~30 lines |
| 2. Shape | Try 6 architecture configs via env vars | 0 lines |
| 3. HPs | Sweep LR → warmdown → batch size sequentially | 0 lines |
| 4. Submit | Combine winners, run 5 seeds, submit PR | 0 lines |

**Core thesis:** The quantization gap (0.0325 BPB) is the single biggest opportunity.
QAT addresses it directly. Shape and HP tuning squeeze out the rest. Combined, these
three orthogonal improvements should clear the 0.005 BPB bar with margin to spare.
