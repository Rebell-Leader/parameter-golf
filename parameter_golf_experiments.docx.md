  
**PARAMETER GOLF**

Experiment Plan & Research Roadmap

*OpenAI Model Craft Challenge — 16MB Artifact, ≤10 min on 8×H100*

| Target: beat baseline val\_bpb 1.2244 Tier 1: $25 (\~8 H100-hours) | Tier 2: $500 | Tier 3: $1000 Challenge runs March 18 – April 30, 2026 |
| :---: |

# **Track Legend & Execution Strategy**

Experiments are divided into three tracks. Execute in order: complete Track A baselines first to establish measurement infrastructure, then run Track B innovative approaches in parallel, reserving Track C for Tier 2/3 compute.

| Track | Description | Compute Tier |
| :---- | :---- | :---- |
| **A1–A5  BASELINE / STANDARD** | Must-do optimizations that competitors will saturate. Run first to establish measurement baseline and eliminate obvious wins. | Tier 1 ($25) |
| **B1–B7  INNOVATIVE** | Non-standard approaches with real differentiation. Less certain but higher upside. Run top-3 from A-track results to inform B-track prioritization. | Tier 1–2 |
| **C1–C3  NOVEL / HIGH RISK** | Architecturally unconventional ideas. Requires more compute to validate; submit as non-record runs first. | Tier 2–3 |

| Tier 1 Execution Order ($25 budget) 1\. A1 (baseline reproduction, \~1h)  →  2\. A2 arch grid (smoke tests, \~1h)  →  3\. A3+A4 combined (QAT \+ modernization, \~2h)  →  4\. B1 recurrence (smoke test, \~1h)  →  5\. B4 Mamba (smoke test, \~1h)  →  6\. Best 2 configs full 10-min runs (2×\~0.5h on 8×H100)  →  7\. Apply for Tier 2 with results. |
| :---- |

# **Track A — Standard Baselines**

These experiments represent the floor — every serious competitor will implement them. Run these first to establish your measurement infrastructure and capture easy wins. The goal is to have a well-characterized baseline from which B and C track innovations can be cleanly measured.

| A1  Baseline Reproduction \+ Profiling    BASELINE  |  |
| :---- | :---- |
| **Hypothesis** | Reproduce the official naive baseline (val\_bpb \~1.2244) on 1×H100 and instrument every component to measure where bytes and loss are spent. |
| **Expected bpb gain** | 0.000 (reference point) — establishes measurement harness |
| **Compute (Tier 1\)** | \~1 H100-hour. Run 1× at full wallclock, 1× shortened for fast ablations. |
| **Key risk** | None — if this fails, environment is broken. |
| **Implementation steps** Clone repo, run train\_gpt.py with default config on 1×H100 (sp1024 tokenizer, 9 layers, dim=512). Log: val\_bpb pre/post quant, compressed artifact size breakdown (code bytes vs. weight bytes), peak VRAM, step throughput. Profile with torch.profiler: identify top-3 compute bottlenecks and top-3 memory bottlenecks. Record zlib compression ratio per tensor type (embedding, attention weights, MLP weights) to understand which layers compress best. Establish a 2-minute smoke-test config (200 steps) that correlates well with final val\_bpb for fast iteration. **Success criteria** val\_bpb ≤ 1.226 (within noise of official 1.2244). Smoke-test Pearson r ≥ 0.95 with full run bpb across 5 ablation configs. |  |

| A2  Architecture Ablation Grid    STANDARD  |  |
| :---- | :---- |
| **Hypothesis** | The baseline's depth/width/vocab tradeoff is not optimal. A systematic grid over (layers, dim, heads, vocab\_size) will find a better Pareto point under the compressed-size constraint. |
| **Expected bpb gain** | \~0.01–0.03 improvement over baseline |
| **Compute (Tier 1\)** | \~3 H100-hours. Run 20–30 smoke-test configs (200 steps each), top-3 at full 10-min. |
| **Key risk** | Large search space; smoke-test may not perfectly predict full-run ranking. |
| **Implementation steps** Fix compressed size budget ≤ 15.8MB. Sweep: layers ∈ {6,9,12,16}, dim ∈ {384,512,640}, heads ∈ {4,8}, KV heads ∈ {2,4}, MLP\_mult ∈ {1.5,2,3}. For each config, compute exact int8+zlib compressed size analytically before training to filter over-budget runs. Run all valid configs for 200 steps; rank by val\_bpb. Select top-5 for full 10-min runs. Control: keep tokenizer fixed (sp1024) across all runs to isolate architecture effect. Record and plot: compressed\_size vs. val\_bpb scatter. Identify the efficiency frontier. **Success criteria** At least one config achieves val\_bpb ≤ 1.21 within 15.8MB compressed. |  |

| A3  Quantization-Aware Training (QAT)    STANDARD  |  |
| :---- | :---- |
| **Hypothesis** | The baseline loses \~0.012 bpb from post-hoc int8 quantization. QAT — inserting fake quantization nodes during training — will recover most of this gap for free. |
| **Expected bpb gain** | \~0.008–0.015 improvement (closing the pre/post quant gap) |
| **Compute (Tier 1\)** | \~2 H100-hours. 3 configs: QAT on all weights, QAT on attention only, QAT on MLP only. |
| **Key risk** | QAT adds training instability; learning rate schedule may need retuning. |
| **Implementation steps** Implement fake-quant: insert torch.fake\_quantize\_per\_tensor\_affine with int8 scale/zero computed per-tensor after each linear layer forward pass. Warm up for 500 steps without fake-quant, then enable for the remainder of the 10-min budget. Ablate: fake-quant all layers vs. attention-only vs. MLP-only to find best coverage. Tune QAT learning rate multiplier ∈ {0.3, 0.5, 1.0} relative to base LR. Measure: val\_bpb gap between pre-quant eval and post-quant roundtrip eval. Goal is gap \< 0.002. **Success criteria** Post-quant val\_bpb ≤ pre-quant \+ 0.003 (vs. baseline gap of \~0.012). Net bpb improvement ≥ 0.008. |  |

| A4  SwiGLU \+ RMSNorm \+ RoPE Modernization    STANDARD  |  |
| :---- | :---- |
| **Hypothesis** | Swapping baseline components (GELU→SwiGLU, LayerNorm→RMSNorm, learned pos embeddings→RoPE) improves parameter efficiency and training speed with no architecture restructuring. |
| **Expected bpb gain** | \~0.005–0.015 improvement, primarily from SwiGLU and RoPE long-context extrapolation at eval |
| **Compute (Tier 1\)** | \~1.5 H100-hours. Test each swap independently, then combined. |
| **Key risk** | SwiGLU increases MLP param count by 50% (3 matrices instead of 2); must rebalance dim/layers. |
| **Implementation steps** Swap 1: GELU → SwiGLU. Reduce MLP\_mult from 2 to 1.33 to keep parameter count constant. Swap 2: LayerNorm → RMSNorm (removes bias parameters, \~5% fewer params in norm layers). Swap 3: Learned positional embeddings → RoPE (removes vocab-sized pos embedding table entirely). Swap 4: Combine all three. Redistribute freed parameters into depth (add 1–2 layers). Eval at both training seq\_len (1024) and extended seq\_len (4096, 8192\) with RoPE to test long-context bpb gain. **Success criteria** Combined swap achieves val\_bpb ≤ 1.21. Extended context eval (4096 tokens) shows ≥ 0.01 additional bpb improvement. |  |

| A5  Optimizer & LR Schedule Tuning    STANDARD  |  |
| :---- | :---- |
| **Hypothesis** | The baseline uses AdamW with a cosine schedule. Muon optimizer (as used in modded-nanogpt) or schedule-free Adam may improve convergence within the 10-minute wall-clock budget. |
| **Expected bpb gain** | \~0.005–0.020 improvement from better optimization trajectory |
| **Compute (Tier 1\)** | \~2 H100-hours. Ablate 6 optimizer configs. |
| **Key risk** | Muon requires careful hyperparameter tuning; may destabilize training on very small models. |
| **Implementation steps** Baseline: AdamW, lr=1e-3, cosine decay to 1e-4, 600s wallclock. Config B: Muon for non-embedding params \+ AdamW for embeddings (as in modded-nanogpt). Config C: Schedule-free AdamW (no LR decay needed, adapts automatically). Config D: AdamW with warmup-stable-decay (WSD) schedule — flat middle section preserves peak LR longer. Config E: Gradient clipping sweep (1.0, 0.5, 0.1) on best optimizer. Log loss curves at 30s intervals to detect if optimizer differences compound over training time. **Success criteria** Best optimizer config reaches val\_bpb ≤ 1.21 within 10-minute budget. |  |

# **Track B — Innovative Approaches**

These are the differentiating experiments. Most competitors will not attempt all of these — they require deeper implementation work, architectural knowledge, or creativity beyond standard ML engineering. Running even 2–3 of these successfully will put you in contention for the leaderboard.

| B1  Depth-Recurrent Transformer (Weight Tying Across Layers)    INNOVATIVE  |  |
| :---- | :---- |
| **Hypothesis** | Instead of N unique transformer layers, use 1–2 unique blocks applied recurrently K times. This gives effective depth N=K at cost of 1 layer's parameters, freeing the parameter budget for higher model dim or more recurrent steps. |
| **Expected bpb gain** | \~0.02–0.05 improvement via higher effective depth per byte |
| **Compute (Tier 1\)** | \~3 H100-hours. Sweep recurrence configs: (1 block × 12 steps), (2 blocks × 6 steps), (1 entry \+ 1 shared × 8 steps). |
| **Key risk** | Gradient vanishing through recurrent depth; may need learned per-step scalars or gating. |
| **Implementation steps** Implement RecurrentBlock: a single TransformerBlock with weights shared across K forward passes. Each pass gets a learned scalar gate α\_k ∈ (0,1) to allow depth-varying behavior. Add per-step learned LayerScale (tiny scalar per residual) to let different recurrent steps specialize. Sweep: K ∈ {8, 12, 16} recurrent steps. Use freed params to increase dim: if 1 layer saves 8× params, scale dim from 512 → 1024\. Test: pure recurrence (1 block) vs. staged recurrence (1 unique entry layer \+ 1 shared body block × K). Monitor: does gradient magnitude decay over recurrent depth? Add gradient norm logging per recurrent step. **Success criteria** Recurrent model with effective depth ≥ 12 achieves val\_bpb ≤ 1.19 within 15.8MB compressed. |  |

| B2  FineWeb-Optimized Micro-Vocabulary Tokenizer    INNOVATIVE  |  |
| :---- | :---- |
| **Hypothesis** | The challenge is tokenizer-agnostic (bpb). A BPE tokenizer trained specifically on FineWeb's byte distribution — rather than general English — will achieve higher compression per token, reducing sequence length and improving the model's effective context per byte. |
| **Expected bpb gain** | \~0.01–0.04 improvement from better tokenization of FineWeb's specific text distribution |
| **Compute (Tier 1\)** | \~0.5 H100-hours for tokenizer training. \~2 H100-hours for LM ablations. |
| **Key risk** | Very small vocab (256–512) increases sequence length and slows training. Must rebalance seq\_len vs. batch size. |
| **Implementation steps** Train 4 tokenizers on FineWeb training shards using SentencePiece BPE: vocab sizes 512, 1024, 2048, 4096\. Measure intrinsic tokenizer quality: bytes-per-token on FineWeb val. Plot vocab\_size vs. bytes-per-token. For each tokenizer, compute the int8 embedding table size (vocab × dim × 1 byte) to understand parameter budget impact. Train baseline LM with each tokenizer, same architecture, same 10-min budget. Measure val\_bpb. Find optimal vocab size: the sweet spot where better tokenization outweighs longer sequences and larger embeddings. **Success criteria** Custom FineWeb tokenizer achieves ≥ 5% better bytes-per-token than generic sp1024. LM with optimal vocab achieves val\_bpb ≤ 1.20. |  |

| B3  zlib Entropy Regularization    INNOVATIVE    NOVEL  |  |
| :---- | :---- |
| **Hypothesis** | zlib compresses weight tensors based on their entropy and repeated patterns. Adding a differentiable proxy loss that penalizes weight entropy during training will produce more compressible weights — effectively fitting a larger effective model in the 16MB compressed budget. |
| **Expected bpb gain** | \~0.01–0.03 improvement by fitting \~20% more effective parameters post-compression |
| **Compute (Tier 1\)** | \~2 H100-hours. Test 3 regularization strengths; combine with QAT. |
| **Key risk** | Entropy regularization may conflict with expressiveness; requires careful λ tuning to avoid underfitting. |
| **Implementation steps** Implement proxy entropy loss: for each weight tensor W, compute H(W) ≈ \-Σ p(w) log p(w) where p(w) is estimated via differentiable histogram (softmax over bins). Add to training loss: L\_total \= L\_xent \+ λ \* Σ\_layers H(W\_layer). Sweep λ ∈ {1e-5, 1e-4, 1e-3}. Alternative proxy: penalize variance of weight distribution directly (low-variance weights → lower entropy → better zlib compression). Cheaper to compute. After training, measure: actual zlib compression ratio per tensor. Verify regularized weights compress better. Combine with QAT: both push toward low-entropy, quantization-friendly weights and should synergize. **Success criteria** zlib compression ratio on model weights improves ≥ 15% vs. unregularized baseline. val\_bpb improves ≥ 0.010. |  |

| B4  Mamba / SSM Architecture    INNOVATIVE    HIGH REWARD  |  |
| :---- | :---- |
| **Hypothesis** | State Space Models (Mamba-2) achieve transformer-quality perplexity at 3–5× fewer parameters for the same effective context length. Under a compressed-size constraint, an SSM can fit a much higher-capacity model in 16MB. |
| **Expected bpb gain** | \~0.03–0.07 improvement from fundamentally better parameter efficiency |
| **Compute (Tier 1\)** | \~3 H100-hours. Requires implementing Mamba-2 selective scan; use mamba-ssm library. |
| **Key risk** | Mamba-2 selective scan requires custom CUDA kernels (mamba-ssm package). Integration complexity is high. May not train stably at this scale. |
| **Implementation steps** Install mamba-ssm. Implement a minimal Mamba-2 block: selective SSM \+ SwiGLU, no attention. Drop-in replacement for TransformerBlock. Design hybrid: 75% Mamba blocks \+ 25% attention blocks (attention every 4th layer for global context, Mamba for local). This is the approach used in Jamba/Zamba. Scale up dim and layers using the freed parameter budget. Pure Mamba at same quality uses \~3× fewer params → fit 3× more depth. Benchmark token throughput: Mamba has O(L) complexity vs O(L²) for attention — can process longer sequences in same wall-clock. Train with longer sequences (2048, 4096\) enabled by SSM's linear complexity. More context → lower bpb. **Success criteria** Mamba hybrid model achieves val\_bpb ≤ 1.17 within 15.8MB compressed, beating all transformer configs. |  |

| B5  Knowledge Distillation from Offline Teacher    INNOVATIVE  |  |
| :---- | :---- |
| **Hypothesis** | Train a large teacher model (1B params) offline, then use soft-target distillation during the 10-minute student training run. The teacher never enters the 16MB artifact — it only guides the student's learning signal during training. |
| **Expected bpb gain** | \~0.01–0.03 improvement from richer training signal vs. hard token targets |
| **Compute (Tier 1\)** | \~1 H100-hour for distillation experiments (teacher loaded from HuggingFace; student trains 10 min). |
| **Key risk** | Teacher must be accessible during training (network call during training, or pre-computed soft targets stored locally). Pre-computing soft targets for full FineWeb requires significant storage. |
| **Implementation steps** Load a small but capable teacher (e.g., GPT-2 medium or OLMo-1B) from HuggingFace at training start. Teacher is inference-only; not in artifact. For each training batch, run teacher forward pass to get soft probability distributions over sp1024 vocab. Student loss: L \= α \* KL(student || teacher) \+ (1-α) \* CrossEntropy(student, hard\_targets). Sweep α ∈ {0.3, 0.5, 0.7}. Challenge: teacher vocab ≠ student vocab (sp1024). Solution: project teacher logits through a learned vocab alignment matrix, or use intermediate layer distillation (match hidden states). Alternative: pre-compute teacher logits for first 2 shards offline, store compressed as npz. Load during training. Avoids network calls during the timed run. **Success criteria** Distilled student achieves val\_bpb ≤ 1.20, beating same-architecture student trained without distillation by ≥ 0.01. |  |

| B6  Test-Time Training (TTT) Layers    INNOVATIVE    HIGH REWARD  |  |
| :---- | :---- |
| **Hypothesis** | TTT layers replace the static KV-cache in attention with a mini-model that updates its weights via gradient descent on the current context. This gives each document a dynamically adapting internal state — effectively infinite context within a fixed parameter count. |
| **Expected bpb gain** | \~0.03–0.08 improvement on long documents by adapting to document-specific patterns |
| **Compute (Tier 1\)** | \~3 H100-hours. TTT layers require inner-loop gradient computation; throughput is lower. |
| **Key risk** | High implementation complexity. Inner loop adds significant compute overhead per forward pass. May not converge stably within 10-minute training budget. |
| **Implementation steps** Implement TTT-Linear layer (simplest TTT variant): replace V-projection with a tiny linear model W that is updated via 1 gradient step on each new key-value pair. Hybrid architecture: 50% standard attention layers \+ 50% TTT layers. TTT layers handle long-range context; attention layers handle local patterns. Train the outer model (including TTT layer initialization weights) end-to-end. The inner TTT update rule itself has learnable parameters (learning rate, momentum). At eval time: feed full documents (up to 8192 tokens) to maximize TTT adaptation benefit. TTT state updates as the model reads through the document. Measure: bpb on first 512 tokens of each doc (no TTT benefit) vs. last 512 tokens (full TTT adaptation). Gap shows TTT's contribution. **Success criteria** TTT model shows ≥ 0.02 bpb improvement in second half of long documents vs. first half. Overall val\_bpb ≤ 1.18. |  |

| B7  Ensemble of Micro-Models    INNOVATIVE  |  |
| :---- | :---- |
| **Hypothesis** | Five independently trained 3MB models, each with a different random seed and data order, can be ensembled at inference time. Ensemble bpb is theoretically guaranteed to be ≤ mean individual bpb. Five 3MB models fit in 15MB. |
| **Expected bpb gain** | \~0.01–0.025 improvement from ensemble averaging (theoretical lower bound guarantee) |
| **Compute (Tier 1\)** | \~2 H100-hours. Train 5 × 3MB models in parallel (1 GPU each, 5 GPUs simultaneously). |
| **Key risk** | 5 × 3MB models must each be competitive. Diversity of errors is required for ensemble gain; correlated failures reduce benefit. |
| **Implementation steps** Design a 3MB model: reduce baseline to \~3.1MB compressed. Estimate: 6 layers, dim=384, vocab=1024, tied embeddings → \~3.0MB int8+zlib. Train 5 instances with different seeds (seed ∈ {1,2,3,4,5}) and different data shard orderings to maximize weight space diversity. Ensemble prediction: at each token position, compute geometric mean of 5 models' probability distributions (equivalent to averaging log-probs, then renormalizing). Pack ensemble into artifact: serialize 5 weight files \+ ensemble inference code. Total ≤ 15.8MB. Ablate: does ensemble of diverse architectures (e.g., 2 transformers \+ 1 Mamba \+ 2 recurrent) beat same-architecture ensemble? **Success criteria** 5-model ensemble achieves val\_bpb ≤ individual model bpb − 0.010. Ensemble fits in ≤ 15.8MB compressed. |  |

# **Track C — Novel / High-Risk Approaches**

These experiments are architecturally unconventional and require Tier 2–3 compute to fully validate. Submit these as non-record runs first to de-risk before a full leaderboard submission. Even a negative result is worth submitting per the challenge rules.

| C1  Hypernetwork Weight Generation    NOVEL    HIGH RISK  |  |
| :---- | :---- |
| **Hypothesis** | A tiny hypernetwork (3M params) generates the weights of a larger LM (30M params) on-the-fly at inference. The compressed hypernetwork is much smaller than the compressed weights it generates, because hypernetwork weights encode structure rather than random noise. |
| **Expected bpb gain** | \~0.02–0.06 improvement from fitting 3× more effective parameters in 16MB |
| **Compute (Tier 1\)** | \~4 H100-hours. Novel architecture; requires end-to-end differentiable weight generation. |
| **Key risk** | Training hypernetworks is notoriously unstable. Weight generation may produce degenerate solutions. Long training time to reach competitive bpb. |
| **Implementation steps** Implement HyperTransformer: a small MLP (input: layer index embedding \+ head index embedding → output: weight matrix) that generates all Q/K/V/O projections. The generated LM has 8 layers, dim=512 — but ALL weight matrices are generated by the hypernetwork. Only the hypernetwork weights are stored. Key: hypernetwork must generate diverse weights per layer (not collapse to same output). Add layer/head index inputs with sinusoidal embeddings. Train end-to-end: backprop through weight generation into hypernetwork. Use gradient checkpointing through generation step. Measure: how much does the hypernetwork compress vs. storing weights directly? Compare zlib(hypernetwork\_weights) vs. zlib(generated\_weights). **Success criteria** Hypernetwork artifact ≤ 8MB compressed. Generated LM achieves val\_bpb ≤ 1.20. Net effective parameter efficiency \> 2× baseline. |  |

| C2  Structured Weight Serialization for zlib    NOVEL  |  |
| :---- | :---- |
| **Hypothesis** | zlib uses LZ77 (finds repeated byte sequences). Reordering weight tensors before serialization — sorting by magnitude, interleaving similar layers, or neuron permutation to align repeated sub-patterns — can dramatically improve compression without changing model quality. |
| **Expected bpb gain** | \~0.005–0.02 improvement from fitting more parameters in same byte budget |
| **Compute (Tier 1\)** | \~0.5 H100-hours. Post-processing experiment; no retraining needed. |
| **Key risk** | Compression gains may be model-architecture-specific and hard to generalize. Permutation search is combinatorial. |
| **Implementation steps** Baseline: measure zlib compression ratio on int8 weights serialized in default PyTorch order (layer-by-layer, tensor-by-tensor). Strategy 1: Sort neurons within each layer by L2 norm before serialization. Similar-magnitude weights cluster together → better LZ77 matching. Strategy 2: Serialize all attention Q-matrices together, then all K-matrices, then V-matrices (group by tensor type across layers). Tests if same-type tensors compress better together. Strategy 3: Apply neuron permutation to align layer L and layer L+1 weight sub-blocks. Residual stream alignment \= more repeated 8-byte patterns \= better LZ77. Measure compression ratio improvement for each strategy. Can stack strategies. Implement best as a post-training serialization step. **Success criteria** Best serialization strategy achieves ≥ 10% better compression ratio vs. default. Equivalent to fitting ≥ 10% more parameters in same 16MB. |  |

| C3  Extended Context Eval via YaRN / LongRoPE    NOVEL  |  |
| :---- | :---- |
| **Hypothesis** | Training uses 1024-token sequences (fast, high throughput). At eval time, extend context to 8192 tokens using YaRN positional scaling. More context per document → lower cross-entropy → lower bpb. The model parameters don't change; only the eval procedure does. |
| **Expected bpb gain** | \~0.01–0.04 improvement by giving the model more context to predict from |
| **Compute (Tier 1\)** | \~1 H100-hour. Train with RoPE \+ YaRN scaling factor; eval at multiple sequence lengths. |
| **Key risk** | Attention at 8192 tokens requires 64× more memory than at 1024\. May OOM on H100 for larger models. Need to use flash attention. |
| **Implementation steps** Replace learned pos embeddings with RoPE in baseline model (prerequisite for extrapolation). Implement YaRN: modify RoPE frequency scaling with interpolation factor s and attention temperature correction to enable out-of-distribution sequence lengths. Train at seq\_len=1024 for full 10-min budget. At eval, test seq\_len ∈ {1024, 2048, 4096, 8192}. Enable FlashAttention-2 to handle long-context eval without OOM. Measure throughput at each seq\_len. Measure: bpb as function of context length. Find the optimal eval context window. Submit with optimal context. **Success criteria** Model trained at seq\_len=1024 achieves val\_bpb improvement ≥ 0.015 when evaluated at seq\_len=4096 vs. 1024\. |  |

# **Tier Application Strategy**

Use results from each tier to justify the next compute grant. Document every experiment with val\_bpb, compressed size, and a brief interpretation.

| Grant Tier | Budget | Experiments | Go/No-Go Criteria |
| :---- | :---- | :---- | :---- |
| **Tier 1** | $25 / 8h | A1 full, A2–A5 smoke, B1+B4 smoke | val\_bpb ≤ 1.21 on at least 1 config |
| **Tier 2** | $500 / 160h | Top-3 Track B full runs \+ C1–C3 validation | val\_bpb ≤ 1.19 on at least 1 Track B config |
| **Tier 3** | $1000 / 320h | Best architecture at scale, ensemble, full QAT sweep | Leaderboard submission, target val\_bpb ≤ 1.17 |

| Key Principle: Measure Everything Every experiment must log: val\_bpb (pre and post quant), compressed artifact size in bytes (code \+ weights separately), step throughput (tokens/sec), peak VRAM, and training loss curve. Without consistent logging, it is impossible to isolate which component drives each bpb improvement — and impossible to justify Tier 2/3 grants with credible evidence. |
| :---- |

