import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate
from mlx_lm.tuners import lora
import json
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import argparse
import logging
from pathlib import Path
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FarmingModelTrainer:
    def __init__(self, model_path, data_path, output_dir="./checkpoints"):
        """
        Initialize the trainer with model and data paths.
        
        Args:
            model_path (str): Path to the base model
            data_path (str): Path to the training data
            output_dir (str): Directory to save checkpoints
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.learning_rate = 5e-5
        self.batch_size = 4
        self.num_epochs = 3
        self.max_length = 2048
        self.gradient_accumulation_steps = 4
        self.warmup_steps = 100
        self.save_steps = 500
        
        # Initialize model and dataset
        self._load_model_and_data()
        
    def _load_model_and_data(self):
        """Load model, tokenizer, and dataset."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model, self.tokenizer = load(str(self.model_path))
            
            logger.info(f"Loading dataset from {self.data_path}")
            self.dataset = load_dataset('json', data_files=str(self.data_path))['train']
            
            logger.info(f"Dataset loaded with {len(self.dataset)} examples")
        except Exception as e:
            logger.error(f"Error loading model or dataset: {str(e)}")
            raise
    
    def prepare_data(self):
        """Prepare and tokenize the dataset."""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="np"
            )
        
        logger.info("Tokenizing dataset...")
        tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names,
        )
        
        return tokenized_dataset
    
    def create_optimizer(self):
        """Create and configure the optimizer."""
        return mx.optim.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8
        )
    
    def save_checkpoint(self, epoch, step, loss):
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.output_dir / f"checkpoint_epoch{epoch}_step{step}_{timestamp}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        mx.save_checkpoint(str(checkpoint_path), self.model)
        
        # Save training metadata
        metadata = {
            "epoch": epoch,
            "step": step,
            "loss": float(loss),
            "timestamp": timestamp,
            "model_path": str(self.model_path),
            "training_params": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_length": self.max_length
            }
        }
        
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def compute_loss(self, input_ids, labels):
        """Compute the training loss."""
        outputs = self.model(input_ids)
        logits = outputs.logits
        
        loss = mx.nn.losses.cross_entropy(
            logits.reshape(-1, self.model.vocab_size),
            labels.reshape(-1)
        )
        
        return loss.mean()
    
    def train(self):
        """Main training loop."""
        dataset = self.prepare_data()
        optimizer = self.create_optimizer()
        
        def data_generator():
            for i in range(0, len(dataset), self.batch_size):
                batch = dataset[i:i + self.batch_size]
                input_ids = mx.array(batch['input_ids'])
                labels = mx.array(batch['input_ids'])
                yield input_ids, labels
        
        total_steps = 0
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{self.num_epochs}")
            epoch_loss = 0
            num_batches = 0
            
            for input_ids, labels in tqdm(data_generator(), desc=f"Epoch {epoch+1}"):
                # Forward pass and loss computation
                loss = self.compute_loss(input_ids, labels)
                
                # Gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (total_steps + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.update(self.model)
                    self.model.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
                total_steps += 1
                
                # Save checkpoint
                if total_steps % self.save_steps == 0:
                    self.save_checkpoint(epoch, total_steps, loss.item())
            
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(epoch, total_steps, avg_loss)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/3600:.2f} hours")
        logger.info(f"Final checkpoint saved to {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train farming model with MLX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    trainer = FarmingModelTrainer(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    # Update training parameters from args
    trainer.batch_size = args.batch_size
    trainer.num_epochs = args.num_epochs
    trainer.learning_rate = args.learning_rate
    
    trainer.train()

if __name__ == "__main__":
    main() 