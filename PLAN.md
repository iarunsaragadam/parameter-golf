# Parameter Golf: Concrete Attack Plan ($500 Compute Budget)

## Goal
Beat the naive baseline BPB of **1.2244** by at least **0.005 nats** → target **≤1.2194 BPB**
(Non-record reference: 4-hour training achieves 1.2074, showing significant headroom exists)

## Budget Math
- 8xH100 node ≈ $20-25/hour (cloud pricing)
- $500 ÷ ~$22/hr ≈ **~23 hours** of 8xH100 time
- Each run = 10 minutes → **~138 full runs** for experimentation
- Strategy: allocate runs across 4 phases

---

## Phase 1: Hyperparameter Sweep (40 runs, ~$145)
**Why:** The baseline uses hand-picked hyperparameters. The Muon optimizer has many knobs
(6 learning rates, momentum, warmup, warmdown, grad clip, batch size) and small changes
can yield 0.01+ BPB improvement for free.

### Experiments:
1. **Learning rate sweep** (12 runs)
   - Sweep `MATRIX_LR` in {0.02, 0.03, 0.05, 0.06} (baseline: 0.04)
   - Sweep `EMBED_LR` / `TIED_EMBED_LR` in {0.03, 0.07, 0.10} (baseline: 0.05)
   - These are the highest-leverage knobs since matrix weights dominate the parameter count

2. **Schedule tuning** (8 runs)
   - Sweep `WARMDOWN_ITERS` in {800, 1000, 1400, 1600} (baseline: 1200)
   - Sweep `WARMUP_STEPS` in {10, 30, 50, 100} (baseline: 20)
   - Longer warmdown = more time at peak LR = more training signal

3. **Batch size & sequence length** (8 runs)
   - Try `TRAIN_BATCH_TOKENS` in {262144, 786432, 1048576} (baseline: 524288)
   - Try `TRAIN_SEQ_LEN` in {512, 2048} (baseline: 1024)
   - Smaller batch = more gradient updates in 10 min; longer seq = better long-range learning

4. **Muon-specific** (8 runs)
   - Sweep `MUON_MOMENTUM` in {0.90, 0.93, 0.97} (baseline: 0.95)
   - Sweep `BETA1`/`BETA2` for Adam on embeddings
   - Sweep `GRAD_CLIP_NORM` in {0.5, 0.8, 1.5} (baseline: 1.0)

5. **Miscellaneous** (4 runs)
   - `LOGIT_SOFTCAP` in {20, 50} (baseline: 30)
   - `ROPE_BASE` in {5000, 50000} (baseline: 10000)

**Expected gain: 0.005–0.015 BPB from best HP combo**

---

## Phase 2: Architecture Search (35 runs, ~$125)
**Why:** The baseline uses 9×512 (9 layers, dim 512). The optimal depth/width tradeoff
for 16MB is unknown. Also, attention head configuration and MLP ratio matter significantly.

### Experiments:
1. **Depth vs Width** (12 runs)
   - 6×640: fewer layers, wider → better per-layer capacity
   - 12×448: more layers, narrower → more compositional
   - 8×576, 10×480, 16×384, 7×600
   - All must fit ≤16MB after int8+zlib compression

2. **Attention configuration** (8 runs)
   - `NUM_KV_HEADS` in {1, 2, 8} (baseline: 4) — trade KV params for more layers/width
   - `NUM_HEADS` in {4, 16} (baseline: 8)
   - Multi-query attention (KV=1) saves params → can increase model dim

3. **MLP ratio** (6 runs)
   - `MLP_MULT` in {1.5, 2.5, 3, 4} (baseline: 2)
   - Higher MLP ratio = more capacity in feedforward, fewer attention params

4. **Vocabulary size** (6 runs)
   - `VOCAB_SIZE` in {512, 2048, 4096} (baseline: 1024)
   - Larger vocab = fewer tokens to process = faster training per byte
   - But: embedding table eats into 16MB budget
   - Requires retraining tokenizer via `download_hf_docs_and_tokenize.py`

5. **Skip connection patterns** (3 runs)
   - Modify `skip_weights` initialization and connectivity
   - Try DenseNet-style connections vs current encoder-decoder style

**Expected gain: 0.01–0.03 BPB from best architecture**

---

## Phase 3: Novel Techniques — The Unexplored Edge (30 runs, ~$110)
**Why:** These are techniques unlikely to be tried by most competitors, giving us a
unique advantage. This is where the plan diverges from obvious approaches.

### 3A. ALBERT-Style Cross-Layer Weight Sharing (10 runs)
**What:** Share transformer block weights across multiple layers. E.g., train 3 unique
blocks but repeat them 3× each → 9 effective layers with 1/3 the parameters.
**Why it works here:** Freed parameters can go toward wider layers or larger vocab.
A 3-block model repeated 4× = 12 effective layers with the parameter cost of 3.
- Try: 2 unique blocks × 6 repeats, 3×4, 4×3, 6×2
- Combine with increased `MODEL_DIM` (e.g., 768 or 1024) since we save params

**Code change:** Modify the `GPT` model class to reuse `Block` instances in `self.blocks`.
This is a ~10-line change in `train_gpt.py`.

### 3B. Progressive/Staged Training (8 runs)
**What:** Train a smaller model for the first 60% of wall-clock time, then expand to the
full model and continue training (net2net-style width/depth expansion).
**Why:** Smaller models train faster per step → more gradient updates → better early
feature learning. Then expand to capture fine-grained patterns.
- Stage 1 (0–360s): Train 6×384 model
- Stage 2 (360–600s): Expand to 9×512, initialize from stage 1 weights
- Variant: progressive depth — start with 4 layers, add layers every 2 minutes

**Code change:** Add a training stage scheduler that reinitializes the model mid-training.
~50-80 lines of new code.

### 3C. Quantization-Aware Training (6 runs)
**What:** The baseline trains in bf16 then quantizes to int8 post-hoc. The non-record
submission shows a gap: 1.1749 pre-quant → 1.2074 post-quant (**0.0325 BPB lost to
quantization!**). QAT can recover most of this.
**Why:** Straight-through estimator (STE) during the last 20% of training to make weights
robust to int8 rounding. This is pure free BPB.
- Simulate int8 quantization in forward pass during last 2 minutes
- Use STE for gradients through the quantization step
- Also try GPTQ-style post-training quantization (smarter than naive round-to-nearest)

**Code change:** Add fake-quantization wrapper in forward pass, activated after step N.
~30 lines of new code.

### 3D. Data Curriculum (6 runs)
**What:** Instead of random shard sampling, order training data by difficulty (shorter/
simpler documents first, complex ones later).
**Why:** Curriculum learning consistently helps in low-compute regimes. Sort shards by
average loss from the baseline model, train easy→hard.
- Run 1: Use baseline to score all 80 shards by avg loss
- Run 2-6: Train with various curriculum orderings
- Also try: repeat high-quality shards more often (importance sampling)

**Code change:** Modify data loading order in the training loop. ~20 lines.

---

## Phase 4: Combination & Final Submission (33 runs, ~$120)
**Why:** Best gains come from combining orthogonal improvements.

### Steps:
1. **Combine best HP + best architecture** (8 runs)
   - Take the winning HP config from Phase 1
   - Apply to the winning architecture from Phase 2
   - Fine-tune any interactions

2. **Add novel techniques one at a time** (10 runs)
   - Layer best Phase 3 techniques onto the Phase 1+2 winner
   - Measure marginal gain of each technique
   - Keep only those with statistically significant improvement

3. **Final hyperparameter polish** (8 runs)
   - Narrow sweep around the best configuration
   - Focus on learning rate and schedule fine-tuning

4. **Reproducibility & variance estimation** (5 runs)
   - Run the final config 5 times with different seeds
   - Compute mean and std for the submission
   - Verify p < 0.01 vs baseline (required for submission)

5. **Prepare submission** (2 runs)
   - Final validation run with full logging
   - Package submission.json, README.md, train.log

---

## Implementation Priority (What to Code First)

| Priority | Change | Effort | Expected BPB Gain |
|----------|--------|--------|-------------------|
| 1 | Hyperparameter sweep harness (env vars already supported) | 0 lines | 0.005–0.015 |
| 2 | Quantization-aware training (STE in last 20% of steps) | ~30 lines | 0.01–0.03 |
| 3 | ALBERT-style weight sharing + wider model | ~15 lines | 0.01–0.02 |
| 4 | Architecture shape search (depth/width/heads) | 0 lines (env vars) | 0.01–0.03 |
| 5 | Progressive training (staged expansion) | ~60 lines | 0.005–0.015 |
| 6 | Data curriculum ordering | ~20 lines | 0.003–0.010 |

## Risk Assessment
- **Phase 1** is zero-risk (no code changes, only env vars)
- **Phase 2** is low-risk (env var changes + minor arch exploration)
- **Phase 3** techniques are medium-risk but high-reward; QAT is safest
- **Quantization gap** (0.0325 BPB) is the single biggest "free lunch" — this alone could beat the baseline if recovered via QAT

## What Makes This Plan Novel
1. **QAT for parameter golf** — nobody else will likely address the quantization gap explicitly
2. **ALBERT-style weight sharing** — counterintuitive (fewer unique params) but enables much wider models within 16MB
3. **Staged progressive training** — exploits the wall-clock constraint by training faster early
4. **Systematic budget allocation** — most competitors will ad-hoc experiment; we methodically sweep then combine
