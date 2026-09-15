# LoRA & QLoRA — Fine-Tuning LLMs 

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





## 🔧 Steps to Fine-Tune a Model

### Using LoRA

```mermaid
flowchart TD
    S1["1️⃣ Install libraries
    (transformers, peft, accelerate, datasets)"] --> S2["2️⃣ Load base model
    & tokenizer"]
    S2 --> S3["3️⃣ Configure LoRA
    (rank, alpha, target_modules)"]
    S3 --> S4["4️⃣ Wrap model with LoRA
    (get_peft_model)"]
    S4 --> S5["5️⃣ Prepare dataset
    (load + tokenize)"]
    S5 --> S6["6️⃣ Train
    (only A & B matrices update)"]
    S6 --> S7["7️⃣ Save adapters"]
```

| Step | Action |
|---|---|
| 1 | Install `transformers`, `peft`, `accelerate`, `datasets` |
| 2 | Load the pre-trained model and tokenizer from HuggingFace |
| 3 | Define `LoraConfig` — set rank (`r`), `lora_alpha`, `dropout`, `target_modules` |
| 4 | Apply `get_peft_model()` — freezes original weights, inserts trainable A & B matrices |
| 5 | Load and tokenize your training dataset |
| 6 | Run `Trainer` with `TrainingArguments` — only A & B matrices are updated |
| 7 | Save the LoRA adapters (not the full model) with `model.save_pretrained()` |

---

### Using QLoRA

```mermaid
flowchart TD
    Q1["1️⃣ Install libraries
    (+ bitsandbytes)"] --> Q2["2️⃣ Load base model
    in 4-bit (NF4)"]
    Q2 --> Q3["3️⃣ Prepare model
    for k-bit training"]
    Q3 --> Q4["4️⃣ Configure & apply
    LoRA (r, alpha, target_modules)"]
    Q4 --> Q5["5️⃣ Prepare dataset
    (load + tokenize)"]
    Q5 --> Q6["6️⃣ Train
    (base stays frozen + quantized)"]
    Q6 --> Q7["7️⃣ Save or merge
    adapters"]
```

| Step | Action |
|---|---|
| 1 | Install `transformers`, `peft`, `accelerate`, `datasets`, `bitsandbytes` |
| 2 | Load base model in 4-bit using `BitsAndBytesConfig` (`load_in_4bit=True`, `nf4`, double quant) |
| 3 | Call `prepare_model_for_kbit_training()` to stabilize the quantized model for training |
| 4 | Define `LoraConfig` and apply `get_peft_model()` — same as LoRA |
| 5 | Load and tokenize your training dataset |
| 6 | Run `Trainer` — only A & B matrices update; base model stays frozen and quantized |
| 7 | Save adapters, or run `merge_and_unload()` to bake LoRA into a deployable model |

> **Key difference:** QLoRA adds one extra step — loading the base model in 4-bit — before the LoRA setup. Everything else is the same.
