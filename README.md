# FPGA LLM

In this project files from https://github.com/kulikov0/red_eyes_is_all_you_need/tree/main are used. 

## Description

A tiny transformer LLM running entirely on an Cyclone V (5CSXFC6D6F31C6N). Character-level text generation trained on Shakespeare, with the full inference pipeline implemented in synthesizable Verilog.

## Architecture

| Parameter     | Value |
|---------------|-------|
| Vocab size    | 256 (byte-level) |
| Embedding dim | 128 |
| Attention heads | 8 (head_dim = 16) |
| Layers        | 4 |
| Context length | 256 |
| FF hidden dim | 512 |
| Weight format | W8A16 (int8 weights, fp16 activations) |
| Total params  | ~854K (854,272 bytes quantized) |

### Inference pipeline

```
Token in -> Embedding -> 4x Transformer Layer -> LayerNorm (ln_f) -> Head Projection -> Sampler -> Token out
                              |                                           |
                         KV Cache (BRAM)                          Weight-tied with tok_emb
```

Each transformer layer:
```
x -> LN1 -> Attention -> +residual -> LN2 -> FF_up -> GELU -> FF_down -> +residual -> out
```

### Key Optimizations
1.  **Streaming Head Projection**: The final linear layer (`head_proj`) uses a streaming `matvec_fp16` module. Instead of storing all 256 logits in a wide register, results are written directly into a dedicated `logits_ram` (M10K).
2.  **Sequential Sampling**: The sampler reads logits sequentially from `logits_ram`, eliminating the need for a 4096-bit wide bus between the head projection and the sampler.
3.  **Activation BRAMs**: Inside `transformer_layer` and `attention`, intermediate activations are stored in M10K BRAMs (`sub_ram`, `ff_ram`, `qkv_ram`, `head_out_ram`), significantly reducing Logic Element pressure.
4.  **Optimized LayerNorm**: `layernorm.v` uses a two-pass approach reading gamma/beta directly from weight ROMs during computation, eliminating the need for large gamma/beta buffers and reducing latency.
