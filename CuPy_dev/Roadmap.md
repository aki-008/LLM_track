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

# Stage 1: Autograd Engine

## Goal

Implement a reverse-mode automatic differentiation engine similar to Micrograd.

Example:

```python
a = Tensor(...)
b = Tensor(...)

c = a * b
d = c + a

loss = d.sum()

loss.backward()

Components

Tensor Class

data

grad

requires_grad

parents

operation metadata

backward function


Graph Engine

Computation graph construction

Topological sorting

Reverse-mode backpropagation


Operations

Addition (+)

Subtraction (-)

Multiplication (*)

Division (/)

Power (**)

Sum

Mean

Matrix Multiplication

Transpose

Reshape

Exp

Log


Verification

For every operation:

torch_grad
vs
your_grad

Expected:

max_error < 1e-6

Checkpoint

Train XOR successfully.

If XOR does not converge, do not proceed further.


---

Stage 2: Neural Network Framework

Goal

Build the basic abstractions required for neural networks.

Core Classes

Module
Parameter
Sequential

Layers

Linear
Embedding
LayerNorm
GELU
Dropout

Important Learning Areas

LayerNorm

Implement manually.

Understand:

Mean

Variance

Epsilon stabilization


GELU

Implement from equation.

Understand why GPT uses GELU instead of ReLU.

Verification

For each layer:

torch_output
vs
your_output

and

torch_grad
vs
your_grad

Expected:

max_error < 1e-5


---

Stage 3: Optimizers

Goal

Implement optimization algorithms from scratch.

Implement

SGD
Adam
AdamW

Concepts to Understand

Momentum

Exponential Moving Average


Adam

First moment estimate

Second moment estimate

Bias correction


AdamW

Decoupled weight decay


Verification

Train a small MLP.

Compare:

PyTorch loss curve
vs
Your framework loss curve


---

Stage 4: GPT Building Blocks

Goal

Implement all transformer primitives.


---

Causal Mask

Create:

-∞ above diagonal

Purpose:

Prevent tokens from attending to future tokens.


---

Self Attention

Pipeline:

QKᵀ
↓
Scale
↓
Mask
↓
Softmax
↓
Multiply by V

Implement manually.


---

Multi-Head Attention

Implement:

Q Projection
K Projection
V Projection

Head Splitting

Attention Per Head

Head Concatenation

Output Projection


---

Feed Forward Network

Architecture:

Linear
↓
GELU
↓
Linear


---

Verification

Compare outputs and gradients against PyTorch.

Expected:

max_error < 1e-5


---

Stage 5: Transformer Block

Goal

Assemble a complete transformer block.

Architecture:

LayerNorm
↓
Multi-Head Attention
↓
Residual Connection

LayerNorm
↓
Feed Forward Network
↓
Residual Connection

Verification

Initialize identical weights in:

PyTorch

Your Framework


Compare outputs.

Expected:

max_error < 1e-5


---

Stage 6: GPT Architecture

Goal

Build GPT-2 124M architecture.

Components

Token Embedding

Maps token IDs to vectors.

Positional Embedding

Learned positional embeddings.

Transformer Stack

12 Transformer Blocks

Final LayerNorm

Language Modeling Head

Projects hidden states to vocabulary logits.


---

GPT-124M Configuration

Layers: 12
Heads: 12
Embedding Size: 768
Context Length: 1024


---

Stage 7: Weight Loading Verification

Goal

Verify implementation correctness before training.

Load pretrained GPT-2 weights.

Compare:

GPT-2 logits
vs
Your logits

For identical inputs.

Why This Matters

If outputs match:

Attention is correct

LayerNorm is correct

Residuals are correct

Embeddings are correct

Architecture is correct


This is the strongest correctness check available.


---

Stage 8: Training Infrastructure

Goal

Build everything required for training.

Implement

CrossEntropyLoss
Gradient Clipping
Learning Rate Scheduler
Warmup Scheduler
AdamW
Checkpoint Saving
Checkpoint Loading


---

Verification

Test 1

Overfit a single batch.

Expected:

Loss approaches zero.

Test 2

Overfit 100 batches.

Expected:

Stable convergence.

Only proceed if both succeed.


---

Stage 9: Dataset Pipeline

Training Progression

Phase 1

Tiny Shakespeare

Purpose:

Debug training.


---

Phase 2

WikiText-2

Purpose:

Validate language modeling.


---

Phase 3

OpenWebText Subset

Purpose:

Intermediate scale training.


---

Phase 4

Larger Datasets

Purpose:

Approach GPT-style training.


---

Dataset Components

BPE Tokenizer

Implement from scratch.

Dataset Loader

Implement:

Tokenization
Chunking
Batch Creation
Shuffling


---

Stage 10: Scaling Strategy

Phase 1

Mini GPT

Layers: 2
Hidden Size: 128
Heads: 4

Purpose:

Fast debugging.


---

Phase 2

Medium GPT

Layers: 6
Hidden Size: 384
Heads: 6

Purpose:

Intermediate validation.


---

Phase 3

GPT-124M

Layers: 12
Hidden Size: 768
Heads: 12
Context Length: 1024

Purpose:

Full target architecture.


---

Testing Philosophy

Every component must pass three checks:

Forward Verification

PyTorch Output
=
Your Output


---

Backward Verification

PyTorch Gradient
=
Your Gradient


---

Training Verification

PyTorch Training Behavior
=
Your Training Behavior


---

Recommended Development Order

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


---

Final Deliverables

Framework

Tensor Class

Autograd Engine

Module System

Optimizers

GPU Backend (CuPy)


Model

GPT Implementation

GPT-124M Architecture

Weight Loader

Training Pipeline


Research Skills Gained

Automatic Differentiation

Backpropagation

Transformer Internals

Attention Mechanisms

Numerical Stability

Optimizer Theory

GPU Computing with CuPy

Training Large Language Models

Framework Design

Model Verification Methodology
