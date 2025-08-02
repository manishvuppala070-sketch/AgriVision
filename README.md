# AgriVision

A machine learning application for satellite image classification using the EuroSAT dataset. This project implements both traditional Decision Tree Classifier and deep learning VGG16 model for land use and land cover classification.

## Features

- **Dataset Processing**: Automated preprocessing of EuroSAT satellite images
- **Multiple ML Models**: 
  - Decision Tree Classifier
  - VGG16 Convolutional Neural Network
- **Performance Metrics**: Comprehensive evaluation with accuracy, precision, recall, F1-score, sensitivity, and specificity
- **Visual Analysis**: Confusion matrix visualization and classification reports
- **Real-time Prediction**: Upload and classify new satellite images
- **GUI Interface**: User-friendly Tkinter-based interface

## Dataset

The project uses the EuroSAT dataset with 10 land use classes:
- Annual Crop
- Forest
- Herbaceous Vegetation
- Highway
- Industrial
- Pasture
- Permanent Crop
- Residential
- River
- Sea Lake

## Requirements

```
opencv-python
numpy
scikit-learn
tensorflow
matplotlib
seaborn
joblib
tkinter
```

## Installation

1. Clone or download the project
2. Install required dependencies:
   ```bash
   pip install opencv-python numpy scikit-learn tensorflow matplotlib seaborn joblib
   ```
3. Ensure the EuroSAT dataset is in the `EuroSAT/` directory
4. Run the application:
   ```bash
   python Main.py
   ```

## Usage

### GUI Interface

1. **Upload Dataset**: Load the EuroSAT dataset
2. **Image Processing**: Preprocess images (resize to 64x64, normalize)
3. **Dataset Splitting**: Split data into training (80%) and testing (20%)
4. **Train Models**:
   - Decision Tree Classifier
   - VGG16 CNN Model
5. **Predict**: Upload new images for classification

### Model Performance

The application provides detailed metrics for both models:
- Accuracy, Precision, Recall, F1-Score
- Sensitivity and Specificity
- Confusion Matrix visualization
- Classification Report

## Project Structure

```
21 AgriVision/
├── EuroSAT/                 # Dataset directory
│   ├── AnnualCrop/
│   ├── Forest/
│   └── ...
├── model/                   # Saved models and preprocessed data
│   ├── DTC_model.pkl
│   ├── VGG16_model.json
│   ├── VGG16_model_weights.h5
│   └── ...
├── test/                    # Test images
├── Main.py                  # Main application
└── README.md
```

## Models

### Decision Tree Classifier
- Traditional machine learning approach
- Max depth: 3
- Uses flattened image features

### VGG16 CNN
- Transfer learning with pre-trained VGG16
- Custom classification layers
- Input size: 64x64x3
- Frozen base layers for feature extraction

## Output

The application displays:
- Model training progress
- Performance metrics comparison
- Confusion matrices
- Real-time image classification results

## Notes

- Models are automatically saved and loaded for faster subsequent runs
- Preprocessed data is cached to avoid reprocessing
- Images are resized to 64x64 pixels for consistency
- The GUI provides step-by-step workflow guidance