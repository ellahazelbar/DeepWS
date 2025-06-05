# ASL Translator

This project implements an offline tool for translating American Sign Language (ASL) videos into English text. The system can process individual sign words and, as a stretch goal, simple ASL sentences.

## Project Structure

```
asl_translator/
├── data/
│   ├── raw/              # Raw video data
│   └── processed/        # Processed data
├── models/              # Trained models
├── src/
│   ├── data/           # Data processing modules
│   ├── models/         # Model architecture
│   ├── training/       # Training scripts
│   └── inference/      # Inference scripts
├── notebooks/          # Jupyter notebooks
└── demo/              # Demo interface
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the ASL dataset and place it in the `data/raw` directory, organized by class (each sign should be in its own folder).

## Usage

### Training

To train the model:
```bash
python src/training/train.py
```

### Inference

To predict a sign from a video:
```bash
python src/inference/predict.py path/to/video.mp4
```

### Interactive Demo

To use the interactive demo:
```bash
jupyter notebook demo/asl_translator_demo.ipynb
```

## Model Architecture

The system uses a CNN+LSTM architecture:
- ResNet18 as the backbone CNN for spatial feature extraction
- LSTM for temporal sequence processing
- Final classification layer for word prediction

## Dataset

The model is trained on the ASL Alphabet and Word Datasets from Kaggle. Each sign should be organized in its own folder within the `data/raw` directory.

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- ~5-10GB storage for data and models 