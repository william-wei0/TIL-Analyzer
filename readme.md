# Neural Network Survival Analysis for Medical Imaging

A deep learning framework for survival analysis using medical imaging data, implementing discrete-time survival models with ResNet-18 architectures.

## Overview

This project implements a neural network-based survival analysis model that predicts patient survival probabilities from the infiltration patterns of the tumor infiltrating lymphocytes (TILs) in hematoxylin and eosin (H&E) slides of pancreatic adenocarcinoma. The model uses a modified ResNet-18 architecture with custom survival prediction heads and includes comprehensive tools for model evaluation, visualization, and statistical analysis.

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

## Data Source and Image Preprocessing

### 1. Download the TCGA-PAAD Dataset

The H&E images used in this project was sourced from **The Cancer Genome Atlas Pancreatic Adenocarcinoma (TCGA-PAAD)** project, a comprehensive dataset maintained by the National Cancer Institute's Genomic Data Commons (NCI GDC).

TCGA-PAAD data can be accessed through the **NCI Genomic Data Commons (GDC) Data Portal**:
**Portal URL**: https://portal.gdc.cancer.gov/projects/TCGA-PAAD

### 2. Image Processing of H&E Bicolor Maps to Red-Blue Bicolor Maps

To improve the learning of TIL infiltration patterns, the H&E images are reduced to a red-blue bi-color map using the deep-learning model designed by Saltz et al. (2018).

To convert the H&E images obtained from the National Cancer Institute's Genomic Data Commons (NCI GDC), download the pretrained model from [https://github.com/SBU-BMI/u24_lymphocyte](https://github.com/SBU-BMI/u24_lymphocyte) and follow the instructions to convert the H&E images to red-blue bicolor maps.

### 3. Annotations CSV Structure
To associate the PAAD H&E images to each patients survival, an annotations CSV is used. 
Each annotation CSV file should contain:
- Column 0: File path to red-blue bicolor TIL map 
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

## Contact

For questions about the survival analysis methodology, please contact:
- William Wei: wjw9857@nyu.edu
- Lunan Liu: ll4255@nyu.edu

For questions about this specific implementation, please open an issue in the repository.

---

## Acknowledgements & Citations

**The implementation of the suvival loss function**:
```
Gensheimer, M. F., & Narasimhan, B. (2019). 
A scalable discrete-time survival model for neural networks. 
PeerJ, 7, e6257. 
https://doi.org/10.7717/peerj.6257
```

**The Deep Learning-based Red-Blue Bicolor Map Transformer**
```
Saltz, J., Gupta, R., Hou, L., Kurc, T., Singh, P., Nguyen, V., … Thorsson, V. (2018). 
Spatial Organization and Molecular Correlation of Tumor-Infiltrating Lymphocytes Using Deep Learning on Pathology Images. 
Cell Reports, 23(1), 181-193.e7. 
doi:10.1016/j.celrep.2018.03.086
```

**Primary TCGA-PAAD Study**:
```
Cancer Genome Atlas Research Network. (2017).
Integrated Genomic Characterization of Pancreatic Ductal Adenocarcinoma.
Cancer Cell, 32(2), 185-203.e13.
https://doi.org/10.1016/j.ccell.2017.07.007
```

**TCGA Pan-Cancer Clinical Data**:
```
Liu, J., Lichtenberg, T., Hoadley, K. A., et al. (2018).
An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics.
Cell, 173(2), 400-416.e11.
https://doi.org/10.1016/j.cell.2018.02.052
```

**Data Curation Considerations**:
```
Collisson, E. A., Bailey, P., Chang, D. K., & Biankin, A. V. (2019).
Molecular subtypes of pancreatic cancer.
Nature Reviews Gastroenterology & Hepatology, 16(4), 207-220.
https://doi.org/10.1038/s41575-019-0109-y
```
