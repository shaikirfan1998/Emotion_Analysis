# 🎭 Emotion Classification Using Fine-Tuned LLMs
### A Comparative Study with Real-World Inference on YouTube Comments

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![LoRA](https://img.shields.io/badge/PEFT-QLoRA%20%2B%20LoRA-green)
![Colab](https://img.shields.io/badge/Platform-Google%20Colab%20Pro+-orange?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This project fine-tunes two state-of-the-art open-source LLMs — **Mistral-7B** and **LLaMA 3.1 8B Instruct** — for **multi-class emotion classification** using **QLoRA + LoRA** (parameter-efficient fine-tuning). It benchmarks **zero-shot**, **few-shot**, and **fine-tuned** performance, then applies the best model to classify emotions in real-world **YouTube comments**.

> 🔑 **Key insight:** Parameter-efficient fine-tuning (QLoRA) achieves near full fine-tune performance at a fraction of the GPU memory cost — making LLM fine-tuning accessible on free/low-cost hardware.

---

## 🎯 What This Project Demonstrates

- Fine-tuning 7B/8B LLMs on a **16,000-sample** labeled emotion dataset
- Comparing **zero-shot vs few-shot vs fine-tuned** performance on the same models
- Applying **QLoRA** (4-bit quantization + LoRA adapters) for memory-efficient training
- Real-world inference pipeline on **noisy, unstructured YouTube comments**
- Full evaluation suite: accuracy, precision, recall, F1, confusion matrix

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
│                                                                     │
│  training.csv (16,000)  ──►  data_loader.py  ──►  Prompt Builder   │
│  validation.csv (2,000) ──►  data_loader.py  ──►  Prompt Builder   │
│                                                                     │
│  Prompt format:                                                     │
│  "Classify the emotion in this text: {text}                         │
│   Emotions: joy, sadness, anger, fear, surprise, disgust            │
│   Answer:"                                                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │                                     │
          ▼                                     ▼
┌─────────────────────┐             ┌──────────────────────┐
│   Mistral-7B-Instruct│            │  LLaMA 3.1 8B Instruct│
│   (Base Model)      │             │  (Base Model)         │
└────────┬────────────┘             └──────────┬────────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     3-WAY EVALUATION                                │
│                                                                     │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────────────┐  │
│  │  Zero-Shot  │   │  Few-Shot   │   │     Fine-Tuned           │  │
│  │             │   │             │   │  QLoRA + LoRA Adapters   │  │
│  │ No examples │   │ 1-3 examples│   │  3 epochs, 16K samples   │  │
│  │ in prompt   │   │ in prompt   │   │  4-bit quantization      │  │
│  └──────┬──────┘   └──────┬──────┘   └────────────┬─────────────┘  │
│         │                 │                        │                │
│         └─────────────────┴────────────────────────┘               │
│                           │                                         │
│                    ┌──────▼──────┐                                  │
│                    │  metrics.py │                                  │
│                    │  Accuracy   │                                  │
│                    │  Precision  │                                  │
│                    │  Recall     │                                  │
│                    │  F1 Score   │                                  │
│                    │  Confusion  │                                  │
│                    │  Matrix     │                                  │
│                    └─────────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   REAL-WORLD INFERENCE                              │
│                                                                     │
│  YouTube Comments (unlabeled)                                       │
│         │                                                           │
│         ▼                                                           │
│  Best Fine-Tuned Model (loaded with LoRA adapter)                  │
│         │                                                           │
│         ▼                                                           │
│  Predicted Emotion per comment                                      │
│         │                                                           │
│         ▼                                                           │
│  inference_output.csv  (comment + predicted_emotion)               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Fine-Tuning Flow (QLoRA + LoRA)

```
Base LLM (7B / 8B params)
        │
        ▼
4-bit Quantization (bitsandbytes)
← Reduces GPU memory from ~28GB to ~6GB
        │
        ▼
LoRA Adapter Injection
← Trainable rank-decomposition matrices on attention layers
← Only ~0.1% of parameters are trained
        │
        ▼
Training Loop (3 epochs, 16K samples)
   ┌────────────────────────┐
   │  Forward pass          │
   │  Compute cross-entropy │
   │  Backprop (LoRA only)  │
   │  Update adapter weights│
   └────────────────────────┘
        │
        ▼
Save LoRA Adapter Checkpoint  →  models/
        │
        ▼
Load Base Model + Adapter for Inference
```

---

## 📊 Evaluation Approach

```
                    ┌─────────────────────┐
                    │  Validation Set      │
                    │  (2,000 samples)     │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        Zero-Shot         Few-Shot         Fine-Tuned
        Mistral-7B        Mistral-7B       Mistral-7B
        LLaMA-3.1         LLaMA-3.1        LLaMA-3.1
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     metrics.py      │
                    ├─────────────────────┤
                    │  Accuracy           │
                    │  Precision (macro)  │
                    │  Recall (macro)     │
                    │  F1 Score (macro)   │
                    │  Confusion Matrix   │
                    │  Per-class Report   │
                    └─────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Base Models** | Mistral-7B-Instruct, LLaMA 3.1 8B | Pretrained LLMs |
| **Fine-Tuning** | QLoRA + LoRA (PEFT) | Parameter-efficient training |
| **Quantization** | bitsandbytes (4-bit) | GPU memory reduction |
| **Framework** | HuggingFace Transformers | Model loading & training |
| **Evaluation** | scikit-learn | Metrics computation |
| **Visualization** | seaborn, matplotlib | Confusion matrix plots |
| **Platform** | Google Colab Pro+ | GPU runtime (A100) |
| **Dataset** | 16K train / 2K validation | Labeled emotion data |

---

## 📁 Project Structure

```
Emotion_Recognition/
│
├── Zero_Few_shot_code_emotion.ipynb   # Zero-shot & few-shot evaluation
├── Finetuning_emotion_classification.ipynb  # QLoRA fine-tuning
├── Inference.ipynb                    # Real-world YouTube inference
│
├── code/
│   └── utils/
│       ├── config.py                  # Paths, labels, model IDs, prompts
│       ├── data_loader.py             # Dataset loading & prompt creation
│       ├── model_utils.py             # Model loading, LoRA, generation
│       ├── gpu_memory_utils.py        # GPU memory monitoring
│       └── metrics.py                 # Accuracy, F1, confusion matrix
│
├── data/
│   ├── train/
│   │   └── training.csv              # 16,000 labeled samples
│   └── valid/
│       └── validation.csv            # 2,000 labeled samples
│
├── models/                           # Auto-created: LoRA adapter checkpoints
└── results/                          # Auto-created: evaluation outputs
```

---

## 🚀 Getting Started

### Prerequisites
- Google Colab Pro+ (A100 GPU recommended)
- Google Drive mounted
- HuggingFace account (for gated models like LLaMA)

### Step 1 — Install Dependencies

```python
!pip install -q transformers>=4.34.0 \
              datasets>=2.14.0 \
              accelerate>=0.23.0 \
              peft>=0.5.0 \
              scikit-learn>=1.3.0 \
              seaborn>=0.12.0 \
              bitsandbytes>=0.41.0 \
              tqdm
```

### Step 2 — Mount Google Drive & Set Paths

```python
from google.colab import drive
drive.mount('/content/drive')

BASE_PATH = '/content/drive/MyDrive/Emotion_Recognition/'
```

### Step 3 — Zero-Shot & Few-Shot Evaluation

Open `Zero_Few_shot_code_emotion.ipynb` and run all cells.
- Loads Mistral-7B and LLaMA 3.1 Instruct
- Runs classification on validation set
- Outputs accuracy, F1, confusion matrix

### Step 4 — Fine-Tune with QLoRA

Open `Finetuning_emotion_classification.ipynb` and run all cells.
- Loads base models in 4-bit quantization
- Applies LoRA adapters
- Trains for 3 epochs on 16K samples
- Saves adapter checkpoints to `models/`

### Step 5 — Real-World Inference on YouTube Comments

Open `Inference.ipynb` and run all cells.
- Loads fine-tuned model + LoRA adapter
- Reads YouTube comments
- Predicts emotion per comment
- Saves to `inference_output.csv`

---

## 🧠 Why QLoRA?

Full fine-tuning of a 7B model requires ~28GB of GPU VRAM — only available on expensive hardware. **QLoRA** solves this by:

1. **4-bit quantization** — compresses the frozen base model from FP16 to 4-bit, reducing memory from ~28GB to ~6GB
2. **LoRA adapters** — injects small trainable matrices into attention layers; only ~0.1% of parameters are updated
3. **Result** — near full fine-tune accuracy at 10x less GPU memory

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Authors

**Shaik Irfan Ahmed** · **Varun Kumar Atkuri** · **Bhavyaraj Nadimipalli**

> *Submitted as part of a graduate-level NLP course project demonstrating LLM fine-tuning, comparative evaluation, and real-world deployment.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/YOUR_HANDLE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/YOUR_USERNAME)
