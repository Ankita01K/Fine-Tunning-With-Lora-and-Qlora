# LoRA & QLoRA — Fine-Tuning LLMs 

## 1. What is Quantization?

When we train a neural network, the most important thing it learns is its **parameters** (also called **weights**).

Normally, these weights are stored using **32-bit floating point numbers (FP32)** — this is called **full precision**.

**Problem:** Big models have billions of parameters. Storing all of them in FP32 needs huge GPU memory and disk space. Most people don't have that kind of hardware.

**Solution: Quantization**
Quantization converts weights from FP32 (32-bit) to a smaller format like **INT8 (8-bit)**. This makes the model smaller and faster to run.

## 2. What is Fine-Tuning?

**Fine-tuning** = Taking an already-trained model and training it a bit more on new data, so it becomes good at a specific job.

### Domain-Specific Fine-Tuning
Teaching a general model a specific **field/industry**.
> Example: Fine-tuning a chatbot on medical textbooks so it understands medical terms.

### Task-Specific Fine-Tuning
Teaching a model to do one **particular task** well.
> Example: Fine-tuning a model only for sentiment analysis (positive/negative reviews).

## 3. Full Fine-Tuning — and Why It's Hard

**Full fine-tuning** means updating **every single weight** in the model.

### Two Big Problems:
1. **Every weight must be updated** → billions of calculations.
2. **Hardware limits** → You need memory for:
   - The model weights
   - The gradients
   - The optimizer's extra data

   This can need **4-6x more memory** than the model's actual size. Most GPUs can't handle this for large models.


   👉 This is why **LoRA** and **QLoRA** were created.

---

## 4. LoRA (Low-Rank Adaptation)

### Simple Idea
Instead of changing **all** the weights, 
LoRA:
- **Freezes** the original weights (doesn't touch them)
- Adds **two small new matrices** that learn the "change" needed


