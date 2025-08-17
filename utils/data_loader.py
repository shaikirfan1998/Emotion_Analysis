# data_loader.py

"""
Data loading utilities for emotion recognition
"""

import os
import json
import random
import pandas as pd
from typing import Dict, List
from datasets import Dataset, DatasetDict
import config

def load_json_data(file_path: str) -> List[Dict]:
    """Load data from a JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_emotion_dataset(train_dir: str, valid_dir: str, test_dir: str) -> DatasetDict:
    """
    Load emotion dataset from the specified directories
    
    Args:
        train_dir: Directory containing training data
        valid_dir: Directory containing validation data
        test_dir: Directory containing test data
        
    Returns:
        DatasetDict containing train, validation, and test datasets
    """
    # Find CSV files in each directory
    train_files = [os.path.join(train_dir, 'training.csv')]
    valid_files = [os.path.join(valid_dir, 'validation.csv')]
    test_files  = [os.path.join(test_dir,  'test.csv')]
    
    print(f"Loading files: {train_files + valid_files + test_files}")
    
    # Load data from CSV files
    train_df = pd.read_csv(train_files[0])
    valid_df = pd.read_csv(valid_files[0])
    test_df  = pd.read_csv(test_files[0])
    
    print(f"Loaded {len(train_df)} training examples")
    print(f"Loaded {len(valid_df)} validation examples")
    print(f"Loaded {len(test_df)} testing examples")
    print(f"Training data columns: {train_df.columns.tolist()}")
    
    # Map integer labels → string names via a dict
    label_map = { i: name for i, name in enumerate(config.EMOTION_LABELS) }
    train_df['label_name'] = train_df['label'].map(label_map)
    valid_df['label_name'] = valid_df['label'].map(label_map)
    test_df['label_name']  = test_df['label'].map(label_map)
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset  = Dataset.from_pandas(test_df)
    
    return DatasetDict({
        'train':      train_dataset,
        'validation': valid_dataset,
        'test':       test_dataset
    })

def get_few_shot_examples(dataset: Dataset, shots_per_class: int = 3) -> Dict[int, List[Dict]]:
    """
    Get few-shot examples for each class
    
    Args:
        dataset: Dataset to sample from
        shots_per_class: Number of examples per class
        
    Returns:
        Dictionary mapping class label indices to lists of examples
    """
    examples_by_class = {}
    num_labels = len(config.EMOTION_LABELS)
    
    # Group examples by class index
    for label_idx in range(num_labels):
        class_examples = [ex for ex in dataset if ex['label'] == label_idx]
        if len(class_examples) >= shots_per_class:
            examples_by_class[label_idx] = random.sample(class_examples, shots_per_class)
        else:
            examples_by_class[label_idx] = class_examples
    
    return examples_by_class

def create_few_shot_prompt(examples_by_class: Dict[int, List[Dict]], query_text: str) -> str:
    """
    Create a few-shot prompt using examples from each class
    
    Args:
        examples_by_class: Dictionary mapping class label indices to lists of examples
        query_text: The text to classify
        
    Returns:
        Formatted few-shot prompt
    """
    examples_str = ""
    for label_idx, examples in examples_by_class.items():
        for example in examples:
            examples_str += (
                f"Text: {example['text']}\n"
                f"Emotion: {config.EMOTION_LABELS[label_idx]}\n\n"
            )
    
    return config.FEW_SHOT_CONFIG['prompt_template'].format(
        examples=examples_str.strip(),
        text=query_text
    )

def create_zero_shot_prompt(text: str) -> str:
    """
    Create a zero-shot prompt for emotion classification
    
    Args:
        text: The text to classify
        
    Returns:
        Formatted zero-shot prompt
    """
    return config.ZERO_SHOT_CONFIG['prompt_template'].format(text=text)
