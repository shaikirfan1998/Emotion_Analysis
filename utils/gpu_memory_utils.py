"""
GPU memory optimization utilities for running large models in Google Colab
"""

import os
import gc
import time
import torch
import psutil
import numpy as np
from transformers import logging

def get_memory_info():
    """Get system and GPU memory information"""
    memory_info = {
        "system": {
            "total": psutil.virtual_memory().total / (1024**3),
            "available": psutil.virtual_memory().available / (1024**3),
            "used": psutil.virtual_memory().used / (1024**3),
            "percent": psutil.virtual_memory().percent
        }
    }
    
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(0)
        memory_info["gpu"] = {
            "name": gpu_properties.name,
            "total": gpu_properties.total_memory / (1024**3),
            "allocated": torch.cuda.memory_allocated(0) / (1024**3),
            "reserved": torch.cuda.memory_reserved(0) / (1024**3),
            "free": (gpu_properties.total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        }
    
    return memory_info

def print_memory_usage():
    """Print current memory usage"""
    memory_info = get_memory_info()
    
    print("\n===== Memory Usage =====")
    print(f"System RAM: {memory_info['system']['used']:.2f}GB / {memory_info['system']['total']:.2f}GB ({memory_info['system']['percent']}%)")
    
    if "gpu" in memory_info:
        print(f"GPU ({memory_info['gpu']['name']}): {memory_info['gpu']['allocated']:.2f}GB allocated, {memory_info['gpu']['free']:.2f}GB free")
    
    return memory_info

def optimize_gpu_memory():
    """Optimize GPU memory for large model inference and training"""
    print("\n===== Optimizing GPU Memory =====")
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("✓ Cleared PyTorch cache")
    
    # Set PyTorch to release memory
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Set PyTorch to release memory when possible")
    
    # Set up more aggressive garbage collection
    gc.collect()
    print("✓ Performed garbage collection")
    
    # Set up transformers logging
    logging.set_verbosity_error()
    print("✓ Set transformers logging to error level")
    
    # Set environment variables for memory efficiency
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("✓ Disabled tokenizers parallelism")
    
    # Enable memory efficient attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        print("✓ Using PyTorch's memory efficient attention")
    
    # Print memory after optimization
    memory_info = print_memory_usage()
    
    return memory_info

def monitor_memory(func):
    """Decorator to monitor memory usage before and after function execution"""
    def wrapper(*args, **kwargs):
        print(f"\n===== Monitoring memory for {func.__name__} =====")
        print_memory_usage()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        print(f"\n===== After {func.__name__} (execution time: {execution_time:.2f}s) =====")
        print_memory_usage()
        
        return result
    
    return wrapper

def estimate_batch_size(model, tokenizer, max_length=512, start_batch_size=8):
    """Estimate maximum batch size that can fit in GPU memory"""
    if not torch.cuda.is_available():
        print("No GPU available. Returning default batch size of 1.")
        return 1
    
    # Start with a reasonable batch size
    batch_size = start_batch_size
    
    while batch_size > 0:
        try:
            # Clean up first
            torch.cuda.empty_cache()
            gc.collect()
            
            # Generate dummy inputs
            dummy_input = "This is a test input." * 20  # Create a reasonable length input
            encoded = tokenizer([dummy_input] * batch_size, 
                                max_length=max_length, 
                                padding="max_length", 
                                truncation=True, 
                                return_tensors="pt").to("cuda")
            
            # Do a forward pass
            with torch.no_grad():
                outputs = model(**encoded)
            
            # If we get here without OOM error, this batch size works
            print(f"Batch size {batch_size} fits in GPU memory.")
            
            # Clean up
            del encoded, outputs
            torch.cuda.empty_cache()
            
            return batch_size
            
        except torch.cuda.OutOfMemoryError:
            # Reduce batch size and try again
            print(f"Batch size {batch_size} is too large. Trying {batch_size // 2}...")
            batch_size = batch_size // 2
            
            # Clean up after OOM
            torch.cuda.empty_cache()
            gc.collect()
    
    # If we get here, even batch size 1 doesn't work
    print("Warning: Cannot fit even a single example in GPU memory.")
    return 1

def adaptive_batch_processing(data, process_func, initial_batch_size=8, allow_dynamic=True):
    """
    Process data in adaptive batches to maximize GPU utilization
    
    Args:
        data: List of data items to process
        process_func: Function that processes a batch of data
        initial_batch_size: Initial batch size to try
        allow_dynamic: Whether to adjust batch size dynamically
        
    Returns:
        List of results
    """
    results = []
    
    batch_size = initial_batch_size
    i = 0
    
    while i < len(data):
        try:
            # Get current batch
            end_idx = min(i + batch_size, len(data))
            current_batch = data[i:end_idx]
            
            # Process batch
            batch_results = process_func(current_batch)
            results.extend(batch_results)
            
            # Move to next batch
            i = end_idx
            
            # If successful and dynamic adjustment is allowed, try increasing batch size
            if allow_dynamic and i < len(data) and batch_size < initial_batch_size * 2:
                batch_size += 1
                print(f"Increasing batch size to {batch_size}")
            
        except torch.cuda.OutOfMemoryError:
            # If OOM error, reduce batch size
            torch.cuda.empty_cache()
            gc.collect()
            
            if batch_size <= 1:
                raise RuntimeError("Cannot process even with batch size of 1. Input may be too large.")
            
            # Reduce batch size
            batch_size = max(1, batch_size // 2)
            print(f"Out of memory. Reducing batch size to {batch_size}")
    
    return results