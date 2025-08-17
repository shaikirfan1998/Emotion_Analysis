"""
Configuration for emotion recognition project
"""

import os
import torch

# Base directory in Google Drive
BASE_DIR = '/content/drive/MyDrive/Emotion_Recognition'

# Directory paths
TRAIN_PATH = os.path.join(BASE_DIR, 'data/train')
VALID_PATH = os.path.join(BASE_DIR, 'data/valid')
TEST_PATH = os.path.join(BASE_DIR, 'data/test')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for dir_path in [TRAIN_PATH, VALID_PATH, TEST_PATH, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Emotion labels (including 'surprise')
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Device configuration - optimized for Google Colab
DEVICE_CONFIG = {
    "use_gpu": True,
    "gpu_ids": [0],  # Colab typically has one GPU
    "mixed_precision": True  # Enable mixed precision training
}

# Model configurations - update with appropriate model IDs
MODEL_CONFIGS = {
    "mistral": {
        "zero_shot": "mistralai/Mistral-7B-Instruct-v0.2",
        "few_shot": "mistralai/Mistral-7B-Instruct-v0.2",
        "fine_tuning": "mistralai/Mistral-7B-v0.1"
    },
    "llama2": {
        "zero_shot": "meta-llama/Llama-2-7b-chat-hf",
        "few_shot": "meta-llama/Llama-2-7b-chat-hf",
        "fine_tuning": "meta-llama/Llama-2-7b-hf"
    },
    "llama3": {
        "zero_shot": "meta-llama/Llama-3.1-8B-Instruct",
        "few_shot": "meta-llama/Llama-3.1-8B-Instruct",
        "fine_tuning": "meta-llama/Llama-3.1-8B-Instruct"
    }
}

# Fine-tuning configuration - optimized for Colab (max GPU utilization)
FINE_TUNING_CONFIG = {
    "max_length": 512,
    "batch_size": 2,  # Smaller batch size to fit in memory
    "gradient_accumulation_steps": 2,  # Increase this to compensate for smaller batch size
    "learning_rate": 1e-4,
    "num_train_epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": True,  # Enable mixed precision
    "save_steps": 500,
    "eval_steps": 500,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05
}

# Zero-shot configuration
ZERO_SHOT_CONFIG = {
    "prompt_template": "Task: Classify the emotion in the following text into one of these categories: sadness, joy, love, anger, fear, surprise.\n\nText: {text}\n\nEmotion:"
}

# Few-shot configuration
FEW_SHOT_CONFIG = {
    "prompt_template": "Task: Classify the emotion in the following text into one of these categories: sadness, joy, love, anger, fear, surprise.\n\nHere are some examples:\n\n{examples}\n\nText: {text}\n\nEmotion:"
}

# Generation configuration for inference
GENERATION_CONFIG = {
    "max_new_tokens": 20,
    "temperature": 0.1,
    "top_p": 0.95,
    "do_sample": True
}