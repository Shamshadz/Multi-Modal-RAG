# Multi-Modal RAG

A Retrieval-Augmented Generation (RAG) system supporting both text and image modalities.

## Project Structure

- `data/` - Datasets and input files
- `models/` - Pretrained and fine-tuned models
- `scripts/` - Scripts for training, inference, and utilities
- `docs/` - Documentation and design notes
- `requirements.txt` - Python dependencies

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Features

- Text and image retrieval
- Integration with vector databases (e.g., FAISS)
- Extensible for new modalities and models

## Getting Started

- Add your data to the `data/` folder.
- Place or download models in the `models/` folder.
- Use scripts in `scripts/` to run training or inference.

## License

MIT
