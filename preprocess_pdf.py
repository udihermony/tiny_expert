import pymupdf4llm
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_pdf(pdf_path: str, output_dir: str = "./data/processed"):
    """
    Process PDF file and save as markdown and LlamaIndex document.
    
    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory to save processed files
    """
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get base filename without extension
        base_name = Path(pdf_path).stem
        
        # Convert to markdown
        logger.info(f"Converting {pdf_path} to markdown...")
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # Save markdown
        md_path = output_path / f"{base_name}.md"
        md_path.write_bytes(md_text.encode())
        logger.info(f"Saved markdown to {md_path}")
        
        # Convert to LlamaIndex document
        logger.info("Converting to LlamaIndex document...")
        llama_reader = pymupdf4llm.LlamaMarkdownReader()
        llama_docs = llama_reader.load_data(pdf_path)
        
        # Save LlamaIndex document
        llama_path = output_path / f"{base_name}_llama.json"
        with open(llama_path, 'w', encoding='utf-8') as f:
            f.write(str(llama_docs))
        logger.info(f"Saved LlamaIndex document to {llama_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return False

if __name__ == "__main__":
    # Process PDF
    pdf_path = "./data/raw_texts/serv1.pdf"
    success = process_pdf(pdf_path)
    
    if success:
        logger.info("PDF processing completed successfully")
    else:
        logger.error("PDF processing failed")