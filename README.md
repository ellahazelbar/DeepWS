# ASL Translation System

This project implements an offline tool for translating American Sign Language (ASL) videos into English text. The system can process individual sign words and, as a stretch goal, simple ASL sentences.

## Features

- Process individual ASL sign videos
- Translate signs to English words
- Simple and intuitive user interface
- Offline operation
- Support for both word-level and sentence-level translation (stretch goal)

## Project Structure

```
asl_translator/
├── data/                  # Data storage and processing
│   ├── raw/              # Raw video data
│   └── processed/        # Processed frames and features
├── models/               # Model definitions and weights
├── src/                  # Source code
│   ├── data/            # Data processing modules
│   ├── models/          # Model architecture definitions
│   ├── training/        # Training scripts
│   └── inference/       # Inference and prediction code
├── notebooks/           # Jupyter notebooks for development
└── demo/                # Demo interface
```

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

3. Download the dataset:
- ASL Alphabet and Word Datasets from Kaggle
- Place the data in the `data/raw` directory

## Usage

1. Training:
```bash
python src/training/train.py
```

2. Running the demo:
```bash
python demo/app.py
```

## Model Architecture

The system uses a CNN + LSTM architecture for word-level translation:
- CNN for spatial feature extraction
- LSTM for temporal sequence processing
- Final classification layer for word prediction

For sentence-level translation (stretch goal):
- Transformer-based model for sequence-to-sequence translation
- Gloss-to-English translation pipeline

## Dataset

Primary dataset:
- ASL Alphabet and Word Datasets from Kaggle

Optional datasets:
- PHOENIX-2014T
- ASLLVD
- HOW2SIGN

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- ~5-10GB storage for data and models

## License

MIT License 