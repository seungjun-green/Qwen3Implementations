# Qwen3 from Scratch in NumPy

A from-scratch implementation of the **Qwen3-0.6B** architecture using only NumPy, verified against the official Hugging Face model.

The goal of this project is to understand every piece of a modern LLM by building it by hand: no PyTorch modules, no attention kernels, just matrix operations. The implementation loads the real pretrained weights and reproduces the official model's logits and greedy generations.

## Verification results

All three inference checks pass against `Qwen/Qwen3-0.6B` from Hugging Face:

| Check | What it tests | Result |
|---|---|---|
| Shape | Token IDs `(N, L)` produce logits `(N, L, 151936)` | ✅ PASS |
| Logits | Same logits as the official model (within float32 tolerance) and the same argmax at every position | ✅ PASS |
| Generation | Greedy decoding (temperature 0) produces the same tokens as Hugging Face `generate(do_sample=False)` | ✅ PASS |

## What's implemented

| Component | Details |
|---|---|
| **Token embedding** | Lookup table `W_E[token_ids]`, shape `(V, D)` |
| **Pre-Norm residual blocks** | `h = x + Attention(RMSNorm(x))`, `out = h + MLP(RMSNorm(h))` |
| **RMSNorm** | Re-scaling only (no mean subtraction, no bias), normalized over the last axis |
| **Grouped-Query Attention** | 16 query heads sharing 8 KV heads; KV heads expanded with `np.repeat` |
| **QK-Norm** | RMSNorm on each 128-dim query and key head vector before RoPE |
| **RoPE** | Rotary position embedding with base 1,000,000, using the Hugging Face half-split (`rotate_half`) layout and element-wise `cos`/`sin` instead of a rotation matrix |
| **Causal masking** | Supports both prefill and cached decoding (mask shape `(L_new, L_total)`) |
| **SwiGLU MLP** | `W_down(SiLU(x W_gate) ⊙ x W_up)` |
| **Final RMSNorm + output layer** | Output layer tied to the embedding matrix |
| **KV cache** | Keys cached after QK-Norm and RoPE, values after the head split; prefill once, then one token per step |
| **Greedy generation** | Batch-aware EOS handling (stops when every sequence has finished) |

### Qwen3-0.6B configuration

| Setting | Value |
|---|---|
| Layers | 28 |
| Hidden size `D` | 1024 |
| Query heads / KV heads | 16 / 8 |
| Head dimension `d_h` | 128 |
| MLP intermediate size `I` | 3072 |
| Vocabulary size | 151,936 |
| RoPE base | 1,000,000 |
| RMSNorm epsilon | 1e-6 |

## Repository structure

| File | Contents |
|---|---|
| `Qwen3` | The model: `Qwen3Config`, `RMSNorm`, `Rope`, `Attention` (GQA + QK-Norm + RoPE + KV cache), `FFN` (SwiGLU), `Qwen3DecoderLayer`, `Embedding`, `FinalOutputLayer`, `Qwen3`, and `generate` |
| `RealQwen3Weight` | `RealQwen3Weights`: loads the pretrained Hugging Face checkpoint and maps it onto the NumPy model |
| `RandomWeight.py` | `RandomQwen3Weights`: random weights for quick shape tests |
| `Generation` | Script version of the three inference checks |
| `inference_check.ipynb` | Colab notebook that clones this repo and runs all three checks |

## Running the checks

Open `inference_check.ipynb` in Google Colab and run all cells. The notebook:

1. Clones this repository.
2. Loads the class and function definitions from the repo files.
3. Builds the NumPy model with the real Qwen3-0.6B weights, plus the official Hugging Face model as a reference.
4. Runs the shape, logits, and generation checks and prints a PASS/FAIL summary.

**Requirements:** `numpy`, `torch`, and `transformers >= 4.51` (for Qwen3 support).

**Hardware note:** the NumPy model runs on the CPU, since NumPy doesn't use GPUs. A GPU, if available, is used only for the Hugging Face reference model (with TF32 disabled so its matmuls stay in full float32 precision). The model in float32 needs about 2.4 GB of RAM.

## Implementation notes

A few details that matter for matching the official model exactly:

- **Weight transposes.** PyTorch stores linear-layer weights as `(out_features, in_features)`, while this implementation computes `x @ W` with `W` as `(in, out)`. Every projection matrix (`q/k/v/o_proj`, `gate/up/down_proj`) is transposed when loading; norm weights and the embedding table are not.
- **RoPE layout.** Hugging Face pairs dimension `i` with dimension `i + d_h/2` (half-split), not neighboring dimensions as in the original RoPE paper. The `cos`/`sin` tables are built by concatenating the angle table with itself to match.
- **Attention scaling.** Scores are divided by `sqrt(d_h)`, as in the official implementation.
- **Tied embeddings.** Qwen3-0.6B reuses the embedding matrix as the output layer, so the output matrix is `W_E.T`. The loader uses a separate `lm_head` when a checkpoint provides one.
- **bfloat16 checkpoint.** NumPy doesn't support bfloat16, so the weights are loaded in float32 before conversion to NumPy arrays.
