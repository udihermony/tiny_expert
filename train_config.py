"""
Training configuration for continued pre-training of the farming model.
This configuration file contains all hyperparameters and settings for training.
"""

from pathlib import Path
import json
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Training configuration class with validation and loading capabilities."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize configuration with default values or load from dict.
        
        Args:
            config_dict: Optional dictionary to override default configuration
        """
        # Default configuration
        self.config = {
            # Paths
            "model_path": "./models/qwen3-1.7b-mlx",
            "data_path": "./data/processed/train.jsonl",
            "output_dir": "./farming-llm-output",
            
            # Training parameters - Ultra conservative settings
            "num_epochs": 3,
            "batch_size": 1,
            "learning_rate": 1e-7,  # Even lower learning rate
            "warmup_steps": 200,  # Longer warmup
            "max_length": 256,  # Shorter sequences
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": 16,  # More accumulation steps
            
            # Optimization parameters - Conservative settings
            "weight_decay": 0.001,  # Reduced weight decay
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-6,  # Increased epsilon
            "max_grad_norm": 0.1,  # Stronger gradient clipping
            
            # Saving and evaluation parameters
            "save_steps": 50,
            "eval_steps": 50,
            "logging_steps": 5,
            "save_total_limit": 3,
            
            # Model parameters
            "block_size": 256,  # Shorter block size
            "vocab_size": 151936,
            
            # Mixed precision training - Disabled
            "fp16": False,
            "fp16_opt_level": "O0",
            
            # Early stopping
            "early_stopping_patience": 5,
            "early_stopping_threshold": 0.001,
            
            # Data processing
            "num_workers": 1,
            "prefetch_factor": 2,
            "pin_memory": True,
            
            # Logging
            "log_level": "INFO",
            "log_file": "training.log",
            
            # Checkpointing
            "checkpoint_format": "epoch_{epoch}_step_{step}",
            "save_best_only": True,
            "save_optimizer": True,
            "save_scheduler": True,
        }
        
        # Update with provided configuration
        if config_dict:
            self.config.update(config_dict)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values."""
        # Ensure paths are Path objects
        self.config["model_path"] = Path(self.config["model_path"])
        self.config["data_path"] = Path(self.config["data_path"])
        self.config["output_dir"] = Path(self.config["output_dir"])
        
        # Validate numeric parameters
        assert self.config["num_epochs"] > 0, "num_epochs must be positive"
        assert self.config["batch_size"] > 0, "batch_size must be positive"
        assert self.config["learning_rate"] > 0, "learning_rate must be positive"
        assert self.config["warmup_steps"] >= 0, "warmup_steps must be non-negative"
        assert self.config["max_length"] > 0, "max_length must be positive"
        assert self.config["gradient_accumulation_steps"] > 0, "gradient_accumulation_steps must be positive"
        
        # Validate paths exist
        if not self.config["model_path"].exists():
            logger.warning(f"Model path does not exist: {self.config['model_path']}")
        if not self.config["data_path"].exists():
            logger.warning(f"Data path does not exist: {self.config['data_path']}")
        
        # Create output directory if it doesn't exist
        self.config["output_dir"].mkdir(parents=True, exist_ok=True)
    
    def save(self, path: str = None):
        """
        Save configuration to file.
        
        Args:
            path: Optional path to save configuration. If None, saves to output_dir/config.json
        """
        if path is None:
            path = self.config["output_dir"] / "config.json"
        
        # Convert Path objects to strings for JSON serialization
        config_dict = self.config.copy()
        config_dict["model_path"] = str(config_dict["model_path"])
        config_dict["data_path"] = str(config_dict["data_path"])
        config_dict["output_dir"] = str(config_dict["output_dir"])
        
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """
        Load configuration from file.
        
        Args:
            path: Path to configuration file
        
        Returns:
            TrainingConfig instance
        """
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to configuration."""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-like setting of configuration values."""
        self.config[key] = value
        self._validate_config()

# Default configuration instance
config = TrainingConfig()

if __name__ == "__main__":
    # Example usage
    config.save()  # Save default configuration
    
    # Load and modify configuration
    custom_config = TrainingConfig({
        "num_epochs": 5,
        "batch_size": 1,
        "learning_rate": 1e-7
    })
    custom_config.save() 