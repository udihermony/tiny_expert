import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_md_to_jsonl(md_path: str, output_path: str, chunk_size: int = 1000):
    """
    Convert markdown file to JSONL format with chunks.
    
    Args:
        md_path: Path to markdown file
        output_path: Path to save JSONL file
        chunk_size: Size of text chunks in characters
    """
    try:
        # Read markdown file
        logger.info(f"Reading markdown file: {md_path}")
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            if len(chunk.strip()) > 100:  # Skip tiny chunks
                chunks.append({"text": chunk.strip()})
        
        # Save as JSONL
        logger.info(f"Saving {len(chunks)} chunks to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + '\n')
        
        logger.info("Conversion completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error converting file: {str(e)}")
        return False

if __name__ == "__main__":
    # Convert markdown to JSONL
    md_path = "./data/processed/serv1.md"
    output_path = "./data/processed/farming_texts.jsonl"
    
    success = convert_md_to_jsonl(md_path, output_path)
    
    if success:
        logger.info("Conversion completed successfully")
    else:
        logger.error("Conversion failed") 