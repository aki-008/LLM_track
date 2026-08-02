# Building GPT-124M from Scratch with CuPy + Custom Autograd

## Project Goal

Build a PyTorch-like deep learning framework on top of CuPy and use it to train a GPT model from scratch.

### Objectives

- Build a Micrograd-style autograd engine
- Create a neural network framework
- Implement transformer components from scratch
- Build GPT-124M architecture
- Train and scale models on GPU using CuPy
- Compare implementations against PyTorch and GPT-2

---

## Stage 1: Autograd Engine

### Goal

Implement a reverse-mode automatic differentiation engine similar to Micrograd.

**Example:**

```python
a = Tensor(...)
b = Tensor(...)

c = a * b
d = c + a

loss = d.sum()

loss.backward()
```

### Components

**Tensor Class**
- `data`
- `grad`
- `requires_grad`
- `parents`
- operation metadata
- backward function

**Graph Engine**
- Computation graph construction
- Topological sorting
- Reverse-mode backpropagation

**Operations**
- Addition (`+`)
- Subtraction (`-`)
- Multiplication (`*`)
- Division (`/`)
- Power (`**`)
- Sum
- Mean
- Matrix multiplication
- Transpose
- Reshape
- Exp
- Log

### Verification

For every operation, compare `torch_grad` vs `your_grad`.

**Expected:** `max_error < 1e-6`

### Checkpoint

Train XOR successfully. **If XOR does not converge, do not proceed further.**

---

## Stage 2: Neural Network Framework

### Goal

Build the basic abstractions required for neural networks.

**Core Classes**
- `Module`
- `Parameter`
- `Sequential`

**Layers**
- `Linear`
- `Embedding`
- `LayerNorm`
- `GELU`
- `Dropout`

### Important Learning Areas

**LayerNorm** — implement manually. Understand:
- Mean
- Variance
- Epsilon stabilization

**GELU** — implement from the equation. Understand why GPT uses GELU instead of ReLU.

### Verification

For each layer, compare `torch_output` vs `your_output` and `torch_grad` vs `your_grad`.

**Expected:** `max_error < 1e-5`

---

## Stage 3: Optimizers

### Goal

Implement optimization algorithms from scratch.

**Implement**
- SGD
- Adam
- AdamW

### Concepts to Understand
- Momentum
- Exponential moving average

**Adam**
- First moment estimate
- Second moment estimate
- Bias correction

**AdamW**
- Decoupled weight decay

### Verification

Train a small MLP. Compare PyTorch loss curve vs your framework's loss curve.

---

## Stage 4: GPT Building Blocks

### Goal

Implement all transformer primitives.

### Causal Mask

Create a mask with `-∞` above the diagonal.

**Purpose:** prevent tokens from attending to future tokens.

### Self-Attention

**Pipeline:**

```
QKᵀ → Scale → Mask → Softmax → Multiply by V
```

Implement manually.

### Multi-Head Attention

**Implement:**
- Q projection
- K projection
- V projection
- Head splitting
- Attention per head
- Head concatenation
- Output projection

### Feed-Forward Network

**Architecture:**

```
Linear → GELU → Linear
```

### Verification

Compare outputs and gradients against PyTorch.

**Expected:** `max_error < 1e-5`

---

## Stage 5: Transformer Block

### Goal

Assemble a complete transformer block.

**Architecture:**

```
LayerNorm → Multi-Head Attention → Residual Connection
LayerNorm → Feed-Forward Network → Residual Connection
```

### Verification

Initialize identical weights in PyTorch and your framework, then compare outputs.

**Expected:** `max_error < 1e-5`

---

## Stage 6: GPT Architecture

### Goal

Build the GPT-2 124M architecture.

### Components

- **Token embedding** — maps token IDs to vectors
- **Positional embedding** — learned positional embeddings
- **Transformer stack** — 12 transformer blocks
- **Final LayerNorm**
- **Language modeling head** — projects hidden states to vocabulary logits

### GPT-124M Configuration

| Parameter | Value |
|---|---|
| Layers | 12 |
| Heads | 12 |
| Embedding size | 768 |
| Context length | 1024 |

---

## Stage 7: Weight Loading Verification

### Goal

Verify implementation correctness before training.

Load pretrained GPT-2 weights and compare GPT-2 logits vs your logits for identical inputs.

### Why This Matters

If outputs match, it confirms:
- Attention is correct
- LayerNorm is correct
- Residuals are correct
- Embeddings are correct
- Architecture is correct

This is the strongest correctness check available.

---

## Stage 8: Training Infrastructure

### Goal

Build everything required for training.

**Implement**
- `CrossEntropyLoss`
- Gradient clipping
- Learning rate scheduler
- Warmup scheduler
- AdamW
- Checkpoint saving
- Checkpoint loading

### Verification

**Test 1 — Overfit a single batch**
Expected: loss approaches zero.

**Test 2 — Overfit 100 batches**
Expected: stable convergence.

Only proceed if both tests succeed.

---

## Stage 9: Dataset Pipeline

### Training Progression

| Phase | Dataset | Purpose |
|---|---|---|
| 1 | Tiny Shakespeare | Debug training |
| 2 | WikiText-2 | Validate language modeling |
| 3 | OpenWebText subset | Intermediate scale training |
| 4 | Larger datasets | Approach GPT-style training |

### Dataset Components

- **BPE tokenizer** — implement from scratch
- **Dataset loader** — implement tokenization, chunking, batch creation, shuffling

---

## Stage 10: Scaling Strategy

| Phase | Model | Layers | Hidden Size | Heads | Context Length | Purpose |
|---|---|---|---|---|---|---|
| 1 | Mini GPT | 2 | 128 | 4 | — | Fast debugging |
| 2 | Medium GPT | 6 | 384 | 6 | — | Intermediate validation |
| 3 | GPT-124M | 12 | 768 | 12 | 1024 | Full target architecture |

---

## Testing Philosophy

Every component must pass three checks:

1. **Forward Verification** — PyTorch output = your output
2. **Backward Verification** — PyTorch gradient = your gradient
3. **Training Verification** — PyTorch training behavior = your training behavior

---

## Recommended Development Order

```
Autograd Engine
    ↓
XOR Training
    ↓
NN Framework
    ↓
Optimizers
    ↓
Attention
    ↓
Transformer Block
    ↓
GPT Architecture
    ↓
Weight Loading Verification
    ↓
Training Infrastructure
    ↓
Tiny Shakespeare
    ↓
WikiText-2
    ↓
OpenWebText
    ↓
GPT-124M
    ↓
Scale Beyond 124M
```

---

## Final Deliverables

**Framework**
- Tensor class
- Autograd engine
- Module system
- Optimizers
- GPU backend (CuPy)

**Model**
- GPT implementation
- GPT-124M architecture
- Weight loader
- Training pipeline

### Research Skills Gained

- Automatic differentiation
- Backpropagation
- Transformer internals
- Attention mechanisms
- Numerical stability
- Optimizer theory
- GPU computing with CuPy
- Training large language models
- Framework design
- Model verification methodology
