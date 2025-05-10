import json
import random
from pathlib import Path
import logging
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data(data_dir: str = "./data/processed") -> List[Dict]:
    """
    Load all processed data from the processed directory.
    """
    data = []
    data_dir = Path(data_dir)
    
    # Load all JSONL files
    for jsonl_file in data_dir.glob("*.jsonl"):
        logger.info(f"Loading data from {jsonl_file}")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Ensure the item has the correct format
                    if isinstance(item, dict) and "text" in item:
                        data.append({"text": item["text"]})
                    elif isinstance(item, str):
                        data.append({"text": item})
                    else:
                        logger.warning(f"Skipping malformed item: {item}")
    
    return data

def split_data(data: List[Dict], train_ratio: float = 0.8, valid_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train, validation, and test sets.
    """
    random.shuffle(data)
    n = len(data)
    train_size = int(n * train_ratio)
    valid_size = int(n * valid_ratio)
    
    train_data = data[:train_size]
    valid_data = data[train_size:train_size + valid_size]
    test_data = data[train_size + valid_size:]
    
    return train_data, valid_data, test_data

def save_jsonl(data: List[Dict], output_path: str):
    """
    Save data to a JSONL file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            # Ensure each item has the correct format
            if not isinstance(item, dict) or "text" not in item:
                logger.warning(f"Skipping malformed item: {item}")
                continue
            f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(data)} items to {output_path}")

def prepare_training_data(data_dir: str = "./data/processed", output_dir: str = "./data"):
    """
    Prepare training data by splitting into train/valid/test sets and saving as JSONL.
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and split data
    data = load_processed_data(data_dir)
    train_data, valid_data, test_data = split_data(data)
    
    # Save split datasets
    save_jsonl(train_data, output_path / "train.jsonl")
    save_jsonl(valid_data, output_path / "valid.jsonl")
    save_jsonl(test_data, output_path / "test.jsonl")
    
    logger.info(f"Data preparation complete:")
    logger.info(f"Train set: {len(train_data)} items")
    logger.info(f"Validation set: {len(valid_data)} items")
    logger.info(f"Test set: {len(test_data)} items")

if __name__ == "__main__":
    prepare_training_data() 