"""
Model utilities for emotion recognition
"""

import os
import torch
from typing import Dict, List, Any, Optional, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import config

def setup_gpu():
    """Set up GPU environment"""
    if not config.DEVICE_CONFIG["use_gpu"]:
        return "cpu"
    
    if not torch.cuda.is_available():
        print("Warning: GPU requested but not available. Using CPU instead.")
        return "cpu"
    
    # Set visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.DEVICE_CONFIG["gpu_ids"]))
    
    device = f"cuda:{config.DEVICE_CONFIG['gpu_ids'][0]}"
    
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    return device

def load_base_model(model_id: str, is_instruct: bool = True):
    """
    Load base model for inference or fine-tuning
    
    Args:
        model_id: Hugging Face model ID
        is_instruct: Whether this is an instruction model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_id}")
    
    # Your token - hard-coded to ensure it's used
    # Your token - hard-coded to ensure it's used
    token = "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"
    
    # Quantization config for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token  # Only pass token here, not use_auth_token
    )
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        token=token  # Only pass token here, not use_auth_token
    )
    
    return model, tokenizer


def prepare_model_for_fine_tuning(model, tokenizer, num_labels=len(config.EMOTION_LABELS)):
    """
    Prepare model for fine-tuning with LoRA
    
    Args:
        model: Base model
        tokenizer: Tokenizer
        num_labels: Number of labels for classification
        
    Returns:
        Prepared model ready for fine-tuning
    """
    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.FINE_TUNING_CONFIG["lora_r"],
        lora_alpha=config.FINE_TUNING_CONFIG["lora_alpha"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=config.FINE_TUNING_CONFIG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    return model

def save_fine_tuned_model(model, tokenizer, model_name: str, timestamp: str):
    """
    Save fine-tuned model
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        model_name: Name of the model (mistral, llama2, llama3)
        timestamp: Timestamp for the directory name
        
    Returns:
        Path to the saved model
    """
    output_dir = os.path.join(config.MODELS_DIR, model_name, f"fine_tuned_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    return output_dir

def generate_text(model, tokenizer, prompt: str, device: str = "cuda:0"):
    """
    Generate text using model
    
    Args:
        model: Model
        tokenizer: Tokenizer
        prompt: Input prompt
        device: Device to use
        
    Returns:
        Generated text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (not the prompt)
    response = generated_text[len(prompt):].strip()
    
    return response
