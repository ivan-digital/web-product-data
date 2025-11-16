## EmbeddingClassifier Architecture

The pure PyTorch model under `category_classification/embedding_classifier.py`
implements a lightweight encoder similar to Qwen’s embedding stack, with
bidirectional self-attention (no causal masking) so tokens can attend to both
left and right context.

### High-Level Structure

```
EmbeddingClassifier
├── EmbeddingBackbone
│   ├── Embed tokens (vocab_size × hidden_size), scale by sqrt(hidden)
│   ├── RMSNorm on embeddings
│   ├── Dropout
│   └── N × EmbeddingBlock (default N=10)
│       ├── RMSNorm → Multi-head self-attention (bidirectional) + DropPath + LayerScale
│       └── RMSNorm → SwiGLU feed-forward + DropPath + LayerScale
└── Classifier head
    └── Dropout → Linear(hidden → hidden) → GELU → Dropout → Linear(hidden → num_labels)
```

### Key Details

- **Attention**: bidirectional multi-head self-attention with grouped key/value
  heads; no causal mask is applied.
- **Rotary Positional Encoding**: applied to q/k inside each block.
- **Residual Stabilization**:
  - Optional DropPath (stochastic depth) with a linearly increasing schedule
    across layers (default max 0.1).
  - LayerScale parameters (`gamma_attn`, `gamma_mlp`, default init 1e‑2) to
    modulate residual branches.
- **Feed-Forward**: SwiGLU MLP with hidden size 4× the model dimension.
- **Pooling**: mean pooling over the sequence, respecting the attention mask.
- **Classifier**: 2-layer MLP head with dropout, enabling richer decision
  boundaries than a single linear layer.

### Precision and Training

- By default, weights/train ops run in FP32. Passing `--amp` to
  `train_embedding_classifier.py` enables `torch.amp.autocast` (FP16/BF16 ops)
  plus `GradScaler` on CUDA for mixed precision.
- Checkpoints store FP32 weights plus optimizer/RNG states; you can resume with
  `--resume path/to/checkpoint-epochX.pt`.
