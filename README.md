# 🎭 Emotion Classification Using Fine-Tuned LLMs
### A Comparative Study with Real-World Inference on YouTube Comments
#### Published Research · University of North Texas · Department of Data Science

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![LoRA](https://img.shields.io/badge/PEFT-QLoRA%20%2B%20LoRA-green)
![Colab](https://img.shields.io/badge/Platform-Google%20Colab%20Pro+-orange?logo=googlecolab)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-90.9%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This research fine-tunes **Mistral-7B** and **LLaMA 3.1 8B Instruct** for **6-class emotion classification** using **QLoRA + LoRA**, then stress-tests both models on **10,000 real-world YouTube comments** containing emojis, slang, and sarcasm.

We benchmark three learning paradigms — **zero-shot**, **few-shot**, and **fine-tuned** — revealing how much each approach gains and where each model wins. Fine-tuned LLaMA 3.1 achieves **90.9% accuracy** and **90.7% F1**, with zero invalid predictions.

> **Dataset:** [Kaggle Emotion Dataset](https://www.kaggle.com/datasets/parulpandey/emotion-dataset) — 16,000 train / 2,000 test · 6 emotion classes: joy, sadness, anger, fear, love, surprise

---

## 📊 Results at a Glance

### Overall Performance Comparison

| Model | Mode | Accuracy | Precision | Recall | F1-Score | Invalid Predictions |
|-------|------|----------|-----------|--------|----------|-------------------|
| Mistral-7B | Zero-Shot | 53.9% | 56.2% | 53.8% | 52.7% | 93 |
| LLaMA 3.1 8B | Zero-Shot | 11.9% | 15.2% | 11.2% | 10.9% | 57 |
| Mistral-7B | Few-Shot | 46.8% | 60.1% | 46.8% | 49.4% | 147 |
| LLaMA 3.1 8B | Few-Shot | 10.8% | 14.8% | 10.8% | 9.8% | 5 |
| **LLaMA 3.1 8B** | **Fine-Tuned** | **90.9%** | **90.8%** | **90.9%** | **90.7%** | **0** |
| Mistral-7B | Fine-Tuned | 89.3% | 89.2% | 89.3% | 89.0% | 0 |

### Key Finding
> Fine-tuning with QLoRA boosted LLaMA 3.1 from **11.9% → 90.9% accuracy** — a **79 percentage point jump**. Zero invalid predictions after fine-tuning confirms strong prompt alignment.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                             │
│                                                                      │
│  ┌─────────────────────┐        ┌──────────────────────────────────┐ │
│  │   Kaggle Dataset    │        │        Base LLMs                 │ │
│  │  training.csv       │        │                                  │ │
│  │  16,000 samples     ├───────►│  Mistral-7B-v0.1  (fine-tune)   │ │
│  │  validation.csv     │        │  Mistral-7B-v0.2  (zero/few)    │ │
│  │  2,000 samples      │        │  LLaMA 3.1 8B     (all modes)   │ │
│  │                     │        └──────────────┬───────────────────┘ │
│  │  6 Emotion Labels:  │                       │                     │
│  │  joy · sadness      │         ┌─────────────▼─────────────┐      │
│  │  anger · fear       │         │     3-Way Evaluation      │      │
│  │  love · surprise    │         │                           │      │
│  └─────────────────────┘         │  Zero-Shot  Few-Shot  FT  │      │
│                                  │  (no ex.)  (3/class) QLoRA│      │
│                                  └─────────────┬─────────────┘      │
│                                                │                     │
│                                  ┌─────────────▼─────────────┐      │
│                                  │       metrics.py           │      │
│                                  │  Accuracy · F1 · Precision │      │
│                                  │  Recall · Confusion Matrix │      │
│                                  └───────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       INFERENCE PIPELINE                             │
│                                                                      │
│  YouTube Data API                                                    │
│       │                                                              │
│       ▼                                                              │
│  10,000 Raw Comments (emojis, slang, sarcasm, HTML)                 │
│       │                                                              │
│       ▼                                                              │
│  Preprocessing                                                       │
│  ├── Remove URLs, HTML tags                                          │
│  └── Keep emojis & punctuation (preserve emotional context)         │
│       │                                                              │
│       ▼                                                              │
│  Fine-Tuned Model (Mistral-7B + LLaMA 3.1 with LoRA adapter)       │
│       │                                                              │
│       ├──► Predicted emotion per comment                             │
│       │                                                              │
│       ▼                                                              │
│  inference_output.csv  +  Manual Validation                         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 QLoRA Fine-Tuning Flow

```
  Mistral-7B / LLaMA 3.1 8B  (~28GB in FP16)
              │
              ▼
  ┌───────────────────────────┐
  │   4-bit Quantization      │  ← bitsandbytes
  │   FP16 → NF4 format       │  ← Memory: ~28GB → ~6GB
  └───────────────┬───────────┘
                  │
                  ▼
  ┌───────────────────────────┐
  │   LoRA Adapter Injection  │  ← PEFT library
  │   Rank-decomposition on   │  ← Only ~0.1% of params trained
  │   attention layers (Q, V) │
  └───────────────┬───────────┘
                  │
                  ▼
  ┌───────────────────────────┐
  │   Supervised Fine-Tuning  │
  │   16,000 labeled samples  │
  │   3 epochs                │
  │   Cross-entropy loss      │
  │   Google Colab A100 GPU   │
  └───────────────┬───────────┘
                  │
                  ▼
       LLaMA 3.1:  90.9% accuracy · 90.7% F1
       Mistral-7B: 89.3% accuracy · 89.0% F1
       0 invalid predictions (both models)
```

---

## 📈 Performance Progression

```
LLaMA 3.1 8B Accuracy:
  Zero-Shot  ██░░░░░░░░░░░░░░░░░░  11.9%
  Few-Shot   ██░░░░░░░░░░░░░░░░░░  10.8%
  Fine-Tuned ██████████████████░░  90.9%  ▲ +79 percentage points

Mistral-7B Accuracy:
  Zero-Shot  ██████████░░░░░░░░░░  53.9%
  Few-Shot   █████████░░░░░░░░░░░  46.8%
  Fine-Tuned █████████████████░░░  89.3%  ▲ +35 percentage points
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Base Models** | Mistral-7B-Instruct v0.1/v0.2, LLaMA 3.1 8B | Foundation LLMs |
| **Fine-Tuning** | QLoRA + LoRA (PEFT) | Parameter-efficient training |
| **Quantization** | bitsandbytes (4-bit NF4) | Reduces GPU memory 28GB → 6GB |
| **Framework** | HuggingFace Transformers | Model loading & training |
| **Evaluation** | scikit-learn | Accuracy, F1, confusion matrix |
| **Visualization** | seaborn, matplotlib | Confusion matrix heatmaps |
| **Platform** | Google Colab Pro+ (A100) | GPU training environment |
| **Data Source** | YouTube Data API + Kaggle | Training & real-world inference |

---

## 📁 Project Structure

```
Emotion_Recognition/
│
├── Zero_Few_shot_code_emotion.ipynb        # Zero-shot & few-shot evaluation
├── Finetuning_emotion_classification.ipynb # QLoRA fine-tuning pipeline
├── Inference.ipynb                         # Real-world YouTube inference
│
├── code/utils/
│   ├── config.py           # Paths, label mappings, model IDs, prompt templates
│   ├── data_loader.py      # Dataset loading & dynamic prompt construction
│   ├── model_utils.py      # Model loading, LoRA application, generation
│   ├── gpu_memory_utils.py # GPU memory monitoring & optimization (Colab)
│   └── metrics.py          # Accuracy, precision, recall, F1, confusion matrix
│
├── data/
│   ├── train/training.csv      # 16,000 labeled samples (Kaggle)
│   └── valid/validation.csv    # 2,000 labeled samples (Kaggle)
│
├── models/   # Auto-created: LoRA adapter checkpoints saved here
└── results/  # Auto-created: evaluation metrics and confusion matrix plots
```

---

## 🚀 Getting Started

### Prerequisites
- Google Colab Pro+ (A100 GPU recommended)
- HuggingFace account with LLaMA 3.1 access approved

### Step 1 — Install Dependencies

```python
!pip install -q transformers>=4.34.0 datasets>=2.14.0 \
             accelerate>=0.23.0 peft>=0.5.0 \
             scikit-learn>=1.3.0 seaborn>=0.12.0 \
             bitsandbytes>=0.41.0 tqdm
```

### Step 2 — Mount Drive & Configure Paths

```python
from google.colab import drive
drive.mount('/content/drive')
# Update config.py:
BASE_PATH = '/content/drive/MyDrive/Emotion_Recognition/'
```

### Step 3 — Zero-Shot & Few-Shot Evaluation
Open `Zero_Few_shot_code_emotion.ipynb` → Run all cells
Loads Mistral-7B v0.2 and LLaMA 3.1 8B and evaluates with prompt-only strategies

### Step 4 — Fine-Tune with QLoRA
Open `Finetuning_emotion_classification.ipynb` → Run all cells
Trains both models for 3 epochs and saves LoRA adapters to `models/`

### Step 5 — Inference on YouTube Comments
Open `Inference.ipynb` → Run all cells
Classifies 10,000 real YouTube comments and saves to `inference_output.csv`

---

## 💬 Real-World Inference Examples

| YouTube Comment | Predicted Emotion | Correct? |
|----------------|------------------|----------|
| "Thanks for nothing 😤" | Anger | ✅ |
| "This made me cry, so beautiful 😭" | Sadness | ✅ |
| "I literally can't stop smiling!!" | Joy | ✅ |
| "Just wow. Great quality." *(sarcastic)* | — | ⚠️ Challenging |
| "BRO HOW????" | Surprise | ⚠️ Challenging |

> Both models correctly identified prominent emotions (joy, anger, sadness). Sarcasm and implicit emotions remain an open challenge.

---

## 🧠 Why QLoRA Makes This Possible

Full fine-tuning of a 7B model requires ~28GB GPU VRAM — only available on expensive enterprise hardware. QLoRA solves this:

- **4-bit quantization** compresses frozen base model from FP16 → NF4 format (~28GB → ~6GB)
- **LoRA adapters** inject small trainable rank-decomposition matrices into attention layers
- Only **~0.1% of parameters** are updated during training
- Result: near full fine-tune performance on a **single A100 GPU** in Google Colab Pro+

---

## 📄 Citation

```bibtex
@article{shaik2024emotion,
  title={Emotion Classification Using Large Language Models:
         A Comparative Study with Real-World Inference on YouTube Comments},
  author={Shaik, Irfan Ahmed and Atkuri, Varun Kumar and
          Nadimipalli, Bhavyaraj and Wheeler, Stephen and
          Bevara, Ravi Varma Kumar and Annavaram, Krishna},
  institution={University of North Texas, Department of Data Science},
  year={2024}
}
```

---

## 👤 Authors

**Irfan Ahmed Shaik** · **Varun Kumar Atkuri** · **Bhavyaraj Nadimipalli**
**Stephen Wheeler** · **Ravi Varma Kumar Bevara** · **Krishna Annavaram**

*The Anuradha and Vikas Sinha Department of Data Science, University of North Texas*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/YOUR_HANDLE)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black?logo=github)](https://github.com/YOUR_USERNAME)
