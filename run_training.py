import subprocess
import os
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import time
import signal
import psutil
from typing import Optional, Dict, Any
import mlx.core as mx



# Import configuration
from train_config import TrainingConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingRunner:
    """Manages the execution of the training pipeline."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the training runner.
        
        Args:
            config: TrainingConfig instance
        """
        self.config = config
        self.start_time = None
        self.current_process: Optional[subprocess.Popen] = None
        self.training_log = None
        
        # Set up directories
        self._setup_directories()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            Path("./data/processed"),
            Path("./logs"),
            Path("./checkpoints"),
            Path(self.config["output_dir"])
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully."""
        logger.info("Received interrupt signal. Cleaning up...")
        if self.current_process:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
        if self.training_log:
            self.training_log.close()
        sys.exit(0)
    
    def _log_system_info(self):
        """Log system information for debugging."""
        system_info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "mlx_version": mx.__version__,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent
        }
        
        log_file = Path("./logs/system_info.json")
        with open(log_file, 'w') as f:
            json.dump(system_info, f, indent=2)
        logger.info(f"System information logged to {log_file}")
    
    def _run_preprocessing(self) -> bool:
        """Run data preprocessing step."""
        try:
            logger.info("Step 1: Preprocessing data...")
            result = subprocess.run(
                ["python", "preprocess_data.py"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Preprocessing completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Preprocessing failed: {e.stderr}")
            return False
    
    def _run_training(self) -> bool:
        """Run the training step."""
        try:
            logger.info("Step 2: Starting continued pre-training...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = Path("./logs") / f"training_{timestamp}.log"
            
            with open(log_file, 'w') as f:
                self.training_log = f
                self.current_process = subprocess.Popen(
                    ["python", "-m", "mlx_lm.lora",
                     "--model", "models/qwen3-1.7b-mlx",
                     "--data", "./data/processed",
                     "--train",
                     "--batch-size", "2",
                     "--iters", "4000",
                     "--learning-rate", "0.00001",
                     "--fine-tune-type", "full"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Monitor and log output
                for line in self.current_process.stdout:
                    print(line, end='')
                    f.write(line)
                    f.flush()
                
                return_code = self.current_process.wait()
                if return_code != 0:
                    logger.error(f"Training failed with return code {return_code}")
                    return False
                
                logger.info("Training completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False
        finally:
            if self.training_log:
                self.training_log.close()
                self.training_log = None
    
    def _run_conversion(self) -> bool:
        """Convert the model to GGUF format."""
        try:
            logger.info("Step 3: Converting to GGUF format...")
            result = subprocess.run([
                "python", "-m", "mlx_lm.convert",
                "--model-path", str(self.config["output_dir"]),
                "--output-path", "./farming-llm-final",
                "--quantize",
                "--q-bits", "4"
            ], check=True, capture_output=True, text=True)
            
            logger.info("Model conversion completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Model conversion failed: {e.stderr}")
            return False
    
    def run(self) -> bool:
        """Execute the complete training pipeline."""
        self.start_time = time.time()
        self._log_system_info()
        
        # Save initial configuration
        self.config.save()
        
        # Run pipeline steps
        steps = [
            ("Preprocessing", self._run_preprocessing),
            ("Training", self._run_training),
            ("Conversion", self._run_conversion)
        ]
        
        success = True
        for step_name, step_func in steps:
            logger.info(f"Starting {step_name}...")
            if not step_func():
                logger.error(f"{step_name} failed")
                success = False
                break
            logger.info(f"{step_name} completed successfully")
        
        # Log completion status
        duration = time.time() - self.start_time
        status = "successfully" if success else "with errors"
        logger.info(f"Training pipeline completed {status} in {duration/3600:.2f} hours")
        
        return success

def main():
    """Main entry point for the training pipeline."""
    try:
        # Load configuration
        config = TrainingConfig()
        
        # Create and run training pipeline
        runner = TrainingRunner(config)
        success = runner.run()
        
        if success:
            logger.info("Training pipeline completed successfully!")
            logger.info(f"Final model saved to: ./farming-llm-final")
        else:
            logger.error("Training pipeline failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 