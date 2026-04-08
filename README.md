# ASC (Adenosquamous Carcinoma) Pathology-Radiomics Analysis Project

A comprehensive framework for imaging feature extraction, pathology feature analysis, and risk prediction modeling for Adenosquamous Carcinoma (ASC).

## Project Overview

This project provides a complete pipeline for:
- **Pathology Segmentation**: Deep learning-based segmentation of SCC (Squamous Cell Carcinoma) and ADC (Adenocarcinoma) regions
- **Imaging Feature Extraction**: Radiomics feature extraction from medical images
- **Pathology Feature Analysis**: Quantitative analysis of pathological characteristics
- **Risk Prediction Modeling**: Machine learning models for SPPI (Surgical Pathology Prognostic Index) risk prediction
- **Visualization**: Comprehensive visualization tools for feature analysis

## Directory Structure

```
ASC-main/
|-- imaging_features/           # Imaging feature modeling
|   |-- build_models_final.py   # Main model building script (Clinical, Radiomics, Combined models)
|   |-- sppi_prediction_analysis.py
|
|-- pathology_features/         # Pathology feature analysis
|   |-- pathology_radiomics_correlation.py  # Correlation analysis between pathology and radiomics features
|   |-- sppi_analysis.py        # SPPI risk prediction analysis
|
|-- radiomics_extraction/       # Radiomics feature extraction scripts
|   |-- 25d_radio_feature_yes.py
|   |-- timchen_liuneiliuzhoue_yes.py
|   |-- timchen_liuzhou_kuo.py
|
|-- visualization/              # Visualization tools
|   |-- visualize_features_merged.py        # Merged visualization for 17 features
|   |-- visualize_5_features.py             # Visualization for 5 core features
|   |-- visualize_features_individual_no1112_nobili.py  # Individual feature visualization
|   |-- visualize_sppi_features.py          # SPPI feature distribution visualization
|   |-- visualize_sppi_individual.py         # Individual SPPI feature visualization
|   |-- visualize_sppi_patient.py            # Patient-level SPPI visualization
|   |-- plot_roc_curves.py                   # ROC curve plotting with confidence intervals
|   |-- delong_test.py                       # DeLong test for model comparison
|   |-- combine.py                           # Utility for combining results
|
|-- segmentation_model/         # Pathology segmentation model (DeepLabV3+)
|   |-- deeplabv3_plus_pytorch/  # DeepLabV3+ implementation
|       |-- train.py             # Training script
|       |-- predict.py           # Prediction/inference script
|       |-- deeplab.py           # Main model implementation
|       |-- get_miou.py          # mIoU calculation
|       |-- nets/                # Network architectures
|           |-- deeplabv3_plus.py
|           |-- mobilenetv2.py
|           |-- xception.py
|       |-- utils/               # Utility functions
|       |-- datasets/            # Dataset handling
|
|-- requirements.txt            # Python dependencies
```

## Pre-trained Model Weights

The pre-trained pathology segmentation model weights are available at:

**[Google Drive - Pathology Segmentation Model Weights](https://drive.google.com/drive/folders/1PmZwmBvKbomEQbyQ0bkPsaibSUxR74tR?usp=sharing)**

Download and place the weights in `segmentation_model/deeplabv3_plus_pytorch/model_data/` directory.

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.8+
- See `requirements.txt` for complete dependencies

### Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Pathology Segmentation

Train or use the pre-trained DeepLabV3+ model for SCC and ADC segmentation:

```bash
cd segmentation_model/deeplabv3_plus_pytorch

# Training
python train.py

# Prediction
python predict.py
```

### 2. Imaging Feature Extraction and Model Building

Build prediction models using clinical and radiomics features:

```bash
cd imaging_features
python build_models_final.py
```

This script implements three modeling strategies:
- **Clinical Model**: Uses all clinical features directly
- **Radiomics Model**: Collinearity analysis + LASSO feature selection
- **Combined Model**: Merges clinical + radiomics features, then applies feature selection

### 3. Pathology Feature Analysis

Analyze correlations between pathology features and radiomics features:

```bash
cd pathology_features
python pathology_radiomics_correlation.py
python sppi_analysis.py
```

### 4. Visualization

Generate visualizations for features and model performance:

```bash
cd visualization

# Feature visualization
python visualize_features_merged.py
python visualize_sppi_features.py

# ROC curves and model comparison
python plot_roc_curves.py
python delong_test.py
```

## Feature Descriptions

### Imaging/Radiomics Features

Features extracted from medical images with prefixes:
- `a_`: Arterial phase features
- `c_`: Contrast-enhanced features
- `p_`: Parenchymal phase features
- `v_`: Venous phase features

### Pathology Features (SPPI Features)

| Feature | Description |
|---------|-------------|
| C_raw | Composition - SCC proportion within tumor area |
| D1_raw | Dispersion - 1 minus dominant cluster ratio (DCR) |
| D2_raw | Multifocality (binary) |
| D3_raw | Log-transformed connected component count |
| D4_raw | Log-transformed significant lesion count |
| D5_raw | Second largest cluster fraction |
| B_raw | Boundary irregularity (log shape index) |
| S1_raw | SCC-ADC contact boundary fraction |
| S2_raw | SCC at invasion front fraction |
| S3_raw | Proximity to invasion front (negative log distance) |
| V_raw | Vascular involvement indicator |

### Clinical Features

- Demographics: Sex, Age, BMI
- Tumor characteristics: Location, Maximum diameter
- Imaging signs: Calcification, Cystic change, Ring enhancement
- Ductal changes: Pancreatic duct dilation, Bile duct dilation
- Secondary changes: Upstream pancreatic atrophy, Retention cyst
- Inflammatory changes: Obstructive pancreatitis
- Invasion indicators: Lymphadenopathy, Vascular invasion, Tumor thrombus

## Model Performance

The models output the following metrics:
- AUC (Area Under ROC Curve)
- Accuracy
- Sensitivity
- Specificity
- Precision
- F1-Score

Optimal thresholds are determined using Youden's Index (Sensitivity + Specificity - 1).

## Output Files

### Model Files
- `model_clinical_final.pkl` - Trained clinical model
- `model_radiomics_final.pkl` - Trained radiomics model
- `model_combined_final.pkl` - Trained combined model
- `scaler_clinical_final.pkl` - Feature scaler for clinical features
- `scaler_radiomics_final.pkl` - Feature scaler for radiomics features

### Result Files
- `model_predictions_final.csv` - Model predictions for all samples
- `model_performance_metrics_final.csv` - Performance metrics summary
- `feature_selection_summary_final.csv` - Feature selection summary
- `selected_features_*.csv` - Selected features and coefficients

### Visualization Files
- `roc_curves_*.png` - ROC curves for train/val/test sets
- `pathology_radiomics_*_heatmap.png` - Correlation heatmaps
- Feature visualization images in respective output directories

## Citation

If you use this code in your research, please cite the relevant publications.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and issues, please open an issue on the repository.
