import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import json
import logging
from pathlib import Path
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_single_batch(data_path: str, batch_size: int = 1):
    """Load a single batch of data for testing."""
    texts = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= batch_size:
                break
            try:
                data = json.loads(line)
                if isinstance(data, dict) and 'text' in data:
                    texts.append(data['text'])
                else:
                    logger.warning(f"Invalid data format at line {i+1}: {data}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error at line {i+1}: {e}")
    
    return texts

def test_forward_backward():
    """Test a single forward/backward pass with detailed logging."""
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model_path = "./models/qwen3-1.7b-mlx"
    model, tokenizer = load(model_path)
    
    # Load a single batch of data
    logger.info("Loading test batch...")
    texts = load_single_batch("data/processed/train.jsonl", batch_size=1)
    if not texts:
        logger.error("No valid texts found in the data file")
        return
    
    # Tokenize using the underlying tokenizer
    logger.info("Tokenizing input...")
    tokens = tokenizer._tokenizer.batch_encode_plus(
        texts,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='np'
    )
    input_ids = mx.array(tokens["input_ids"])
    attention_mask = mx.array(tokens["attention_mask"])
    
    # Print shapes and sample data
    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Sample input: {input_ids[0][:10]}")  # First 10 tokens
    
    # Forward pass
    logger.info("Running forward pass...")
    try:
        outputs = model(input_ids)
        logits = outputs
        logger.info(f"Logits shape: {logits.shape}")
        logger.info(f"Logits sample: {logits[0][0][:5]}")  # First 5 logits of first token
        
        # Check for NaN in logits
        if mx.isnan(logits).any():
            logger.error("NaN detected in logits!")
        else:
            logger.info("No NaN in logits")
        
        # Compute loss
        logger.info("Computing loss...")
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")
        logger.info(f"Loss value: {loss.item()}")
        
        if mx.isnan(loss):
            logger.error("Loss is NaN!")
        else:
            logger.info("Loss computation successful")
            
            # Backward pass
            logger.info("Running backward pass...")
            mx.eval(loss)
            loss.backward()
            
            # Check gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = mx.norm(param.grad).item()
                    logger.info(f"Gradient norm for {name}: {grad_norm}")
                    if mx.isnan(param.grad).any():
                        logger.error(f"NaN detected in gradients for {name}!")
        
    except Exception as e:
        logger.error(f"Error during forward/backward pass: {str(e)}")
        raise

if __name__ == "__main__":
    test_forward_backward() 