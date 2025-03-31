# EfficientNetB3 Training for Spectrogram Classification

This repository contains code for training an EfficientNetB3 model on grayscale spectrogram images.

## Requirements

- Python 3.8+
- CUDA 12.4.1
- RTX 2050 GPU
- All packages in `requirements.txt`

### Key Dependencies
- PyTorch (>=2.0.0) - Deep learning framework
- torchvision (>=0.15.0) - Computer vision utilities
- pandas (>=2.0.0) - Data manipulation
- numpy (>=1.24.0) - Numerical computing
- scikit-learn (>=1.0.0) - Machine learning utilities
- matplotlib (>=3.6.0) - Plotting
- seaborn (>=0.12.0) - Statistical visualizations
- Pillow (>=9.0.0) - Image processing
- opencv-python (>=4.6.0) - Computer vision
- pyarrow (>=12.0.0) - Parquet file handling
- scipy (>=1.10.0) - Scientific computing
- joblib (>=1.2.0) - Parallel computing

## Project Structure

```
.
├── train_efficientnet.py    # Main training script
├── requirements.txt         # Project dependencies
├── train.csv               # Training data labels
├── train_spectrograms/     # Training spectrograms (Parquet files)
├── test_spectrograms/      # Test spectrograms (Parquet files)
├── checkpoints/            # Model checkpoints
└── env/                    # Virtual environment
```

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

To train the model:

```bash
python train_efficientnet.py
```

## Model Details

- Architecture: EfficientNetB3 with pretrained ImageNet weights
- Input size: 300x300 RGB images (converted from grayscale spectrograms)
- Optimization: AdamW with cosine annealing learning rate scheduler
- Loss function: CrossEntropyLoss
- Training: 20 epochs with batch size 32
- Data split: 80% training, 20% validation
- Mixed precision training enabled
- Early stopping implemented to prevent overfitting

## Output Files

The training script generates the following files:
- `best_model.pth`: Best model weights
- `training_curves.png`: Loss and accuracy curves
- `confusion_matrix.png`: Confusion matrix visualization
- `model_predictions.png`: Visual examples of model predictions
- `grad_cam_visualization.png`: Grad-CAM visualization for model interpretability

## Notes

- The `train_spectrograms/`, `test_spectrograms/`, `checkpoints/`, and `env/` directories are ignored by git
- Make sure you have sufficient GPU memory for training (RTX 2050 recommended)
- The model uses mixed precision training for better memory efficiency 