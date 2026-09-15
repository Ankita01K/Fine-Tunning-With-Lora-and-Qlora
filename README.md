# LoRA & QLoRA — Fine-Tuning LLMs (Beginner Guide)

## 1. What is Quantization?

When we train a neural network, the most important thing it learns is its **parameters** (also called **weights**).

Normally, these weights are stored using **32-bit floating point numbers (FP32)** — called **full precision**.

**Problem:** Big models have billions of parameters. Storing all in FP32 needs huge GPU memory and disk space.

**Solution: Quantization** — convert weights from FP32 (32-bit) to a smaller format like **INT8 (8-bit)** or **INT4 (4-bit)**. Model becomes smaller and faster.

```mermaid
flowchart LR
    A["Weight: 13.5678
    (FP32 - 32 bit)"] --> B["Quantization"]
    B --> C["Weight: 13
    (INT8 - 8 bit)"]
    C --> D["Smaller model
    Faster inference
    Slight accuracy loss"]
```

> Think of it like compressing a photo — file size goes down, quality drops slightly.

---

## 2. What is Fine-Tuning?

**Fine-tuning** = taking an already-trained model and training it a bit more on new data so it becomes good at a specific job.

| Type | What it means | Example |
|---|---|---|
| **Domain-specific** | Teach the model a specific field | Fine-tune on medical textbooks → model understands medical terms |
| **Task-specific** | Teach the model one job | Fine-tune only for sentiment analysis (positive/negative) |

---

## 3. Full Fine-Tuning — and Why It's Hard

**Full fine-tuning** = update **every single weight** in the model.

```mermaid
flowchart TD
    A["Pre-trained Model
    (billions of weights)"] --> B["Full Fine-Tuning"]
    B --> C["Update ALL weights"]
    C --> D["Need: model + gradients + optimizer states"]
    D --> E["4-6x model size in memory"]
    E --> F["❌ Most GPUs can't handle this"]
```

**Two big problems:**
1. Every weight must be updated → billions of calculations
2. Hardware limits → not enough GPU memory

👉 This is why **LoRA** and **QLoRA** exist.

---

## 4. LoRA (Low-Rank Adaptation)

### Simple Idea
- **Freeze** the original weights (don't touch them)
- Add **two small new matrices (A and B)** that learn the "change" needed
- Only train A and B

```mermaid
flowchart TD
    X["Input x"] --> W0["Frozen Pre-trained
    Weights (W₀)"]
    X --> A["Matrix A
    (trainable, small)"]
    A --> B["Matrix B
    (trainable, small)"]
    W0 --> SUM(("+"))
    B --> SUM
    SUM --> OUT["Output
    = W₀x + BAx"]
```

### Why It Saves Resources — Example

A weight matrix of size **4096 × 4096** = ~16.7 million numbers.

| Method | Trainable Parameters |
|---|---|
| Full fine-tuning | 16,777,216 (100%) |
| LoRA (rank = 8) | 65,536 (**~0.4%**) |
