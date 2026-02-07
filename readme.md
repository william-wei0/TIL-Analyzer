# Neural Network Survival Analysis for Medical Imaging

A deep learning framework for survival analysis using medical imaging data, implementing discrete-time survival models with ResNet-18 architectures.

## Overview

This project implements a neural network-based survival analysis model that predicts patient survival probabilities from medical images. The model uses a modified ResNet-18 architecture with custom survival prediction heads and includes comprehensive tools for model evaluation, visualization, and statistical analysis.

## Features

- **Custom Survival Model**: A ResNet-18-based architecture with configurable intermediate layers
- **Survival Analysis**: Discrete-time survival modeling with custom loss functions
- **Comprehensive Evaluation**: 
  - Concordance index (C-index) calculation at multiple time points (1, 2, 2.5, and 3 years)
  - Kaplan-Meier survival curve generation
  - Log-rank statistical testing between groups
- **Dimensionality Reduction**: UMAP visualization of learned feature representations
- **K-means Clustering**: Automated clustering analysis with elbow method visualization
- **Class Weighting**: Adaptive sample weighting based on image quality and class imbalance

## Requirements

```
torch
torchvision
lifelines
scikit-learn
matplotlib
numpy
pandas
Pillow
umap-learn
opencv-python
```

## Installation

```bash
pip install torch torchvision lifelines scikit-learn matplotlib numpy pandas Pillow umap-learn opencv-python
```

## Project Structure

```
.
├── model.py                      # Neural network architectures
├── main.py                       # Training script
├── main_test.py                  # Testing and validation utilities
├── nnet_survival_pytorch.py      # Survival analysis utilities
├── CustomTransformations.py      # Image transformations (not included)
├── annotations.csv               # Training data annotations
└── chip annotations/             # Chip experiment annotations
    ├── annotations_chip_high_surv.csv
    ├── annotations_chip_med_surv.csv
    └── annotations_chip_low_surv.csv
```

## Data Format

### Annotations CSV Structure
Each CSV file should contain:
- Column 0: Image file path
- Column 1: Survival time (in days)
- Column 2: Event indicator (0 = censored/alive, 1 = death occurred)

## Usage

### Training a Model

```python
python main.py
```

Training parameters can be configured in `main.py`:
- `training_split`: Train/validation split ratio (default: 175/216)
- `batch_size`: Batch size for training (default: 16)
- `epochs`: Number of training epochs (default: 400)
- `learning_rate`: Learning rate for AdamW optimizer (default: 0.00025)

### Testing and Validation

```python
python main_test.py
```

This will:
1. Load trained model weights
2. Generate survival curves for different CAF (Cancer-Associated Fibroblast) concentrations
3. Create UMAP visualizations of learned features
4. Perform K-means clustering analysis
5. Calculate statistical significance using log-rank tests

## Model Architectures

### DifferentResNet186
- Base: ResNet-18 (pretrained)
- FC1: 512 → 256 (configurable)
- FC2: 256 → 256 (configurable)
- Output: Survival predictions for each time interval

### GoodUmapandGraph
- Base: ResNet-18 (pretrained)
- FC: 512 → 512 (configurable)
- Output: Survival predictions for each time interval

Both models include:
- Sigmoid activations
- Batch normalization
- Dropout regularization (default: 0.5)

## Key Features

### Custom Loss Function
The model uses a custom survival likelihood loss function that accounts for:
- Censored observations
- Time-interval-specific survival probabilities
- Sample weights based on image quality and class balance

### Image Weighting
Two weighting strategies are implemented:
1. **Quality-based weighting**: Images with <20% non-black pixels receive zero weight
2. **Class-based weighting**: Balances censored vs. uncensored samples

### Evaluation Metrics
- **C-index**: Concordance index at 1, 1.5, 2, 2.5, and 3 years
- **Log-rank tests**: Statistical comparison between survival groups
- **Kaplan-Meier curves**: Visual comparison of predicted vs. actual survival

### Visualization Tools
- Individual and aggregate survival curves
- UMAP projections with configurable neighbors (2-50) and min_dist (0.05-0.70)
- K-means clustering with elbow plots
- Survival stratification by risk groups

## Output Files

The project generates several types of output:

1. **Model Checkpoints**: Saved in `./results/saved_models/`
   - `epoch_*.pt`: Periodic checkpoints
   - `best_train_acc.pt`: Best training accuracy
   - `best_val_acc.pt`: Best validation accuracy
   - `lowest_train_loss.pt`: Lowest training loss
   - `lowest_val_loss.pt`: Lowest validation loss

2. **Metrics**: Excel files with training/validation metrics
   - Loss curves
   - Accuracy scores
   - C-index values at multiple time points

3. **Visualizations**:
   - Survival curves (`.png`)
   - UMAP projections (`.png`)
   - K-means clustering plots (`.png`)
   - Elbow method plots (`.png`)

## Reproducibility

The code includes random seed initialization for reproducibility:
```python
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True
```

## Learning Rate Scheduling

The model uses `ReduceLROnPlateau` scheduler with:
- Mode: minimize loss
- Factor: 0.4
- Patience: 15 epochs
- Threshold: 2.0


### Additional Tools & Libraries

This project also utilizes:
- **PyTorch**: Deep learning framework
- **torchvision**: Pre-trained ResNet models
- **lifelines**: Kaplan-Meier fitting and concordance index calculation
- **UMAP**: Dimensionality reduction for feature visualization
- **scikit-learn**: K-means clustering

## Contact

For questions about the survival analysis methodology, please contact:
- William Wei: wjw9857@nyu.edu
- Lunan Liu: ll4255@nyu.edu

For questions about this specific implementation, please open an issue in the repository.

---

## Acknowledgements & Citations

```
Gensheimer, M. F., & Narasimhan, B. (2019). 
A scalable discrete-time survival model for neural networks. 
PeerJ, 7, e6257. 
https://doi.org/10.7717/peerj.6257
```