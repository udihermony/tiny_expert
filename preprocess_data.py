import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(input_file: str, output_file: str):
    """
    Preprocess farming texts from a JSONL file and save the processed data.

    Args:
        input_file: Path to the input JSONL file.
        output_file: Path to save the processed JSONL file.
    """
    try:
        logger.info(f"Reading data from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        logger.info("Preprocessing data...")
        processed_data = []
        for item in data:
            # Example preprocessing: convert text to lowercase and remove extra whitespace
            processed_item = {
                'id': item.get('id', ''),
                'text': item.get('text', '').lower().strip()
            }
            processed_data.append(processed_item)

        logger.info(f"Saving processed data to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item) + '\n')

        logger.info("Preprocessing completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "./data/processed/farming_texts.jsonl"
    output_file = "./data/processed/farming_texts_processed.jsonl"
    preprocess_data(input_file, output_file) 