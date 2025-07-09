# Plant Pathology Image Classification

## Overview
This project focuses on classifying plant images from the Plant Pathology 2020 FGVC7 dataset into multiple disease categories using traditional machine learning techniques. Instead of relying on deep learning or transfer learning, we extract handcrafted features such as HOG (Histogram of Oriented Gradients), color histograms, and texture features, and train two machine learning models: Support Vector Machine (SVM) and Random Forest. The goal is to classify plant images into four categories: `healthy`, `multiple_diseases`, `rust`, and `scab`.

## Features
- **Handcrafted Feature Extraction**:
  - **HOG Features**: Extracts edge and gradient information from grayscale images.
  - **Color Histograms**: Captures color distribution in RGB channels.
  - **Texture Features**: Uses Sobel edge detection to compute mean and standard deviation of edge magnitudes.
- **Machine Learning Models**:
  - Support Vector Machine (SVM) with RBF kernel.
  - Random Forest with balanced class weights.
- **Multi-label Classification with Single-label Output**: Each image is assigned a single label based on the highest probability score (threshold ≥ 0.3).
- **Evaluation Metrics**:
  - Hamming Loss
  - F1 Score (samples average)
  - Precision and Recall (per label and average)
  - Percentage of "None" predictions
- **Visualization**:
  - Sample image predictions
  - Confusion matrices for each label and model
  - Feature importance for Random Forest
  - Model performance comparison plots

## Dataset
The project uses the [Plant Pathology 2020 FGVC7 dataset](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) from Kaggle, which includes:
- A training set with labeled images (`train.csv`) indicating the presence of `healthy`, `multiple_diseases`, `rust`, or `scab`.
- A test set with unlabeled images (`test.csv`) for generating predictions.
- An image directory containing `.jpg` files.

## Requirements
To run this project, you need the following Python packages:
```bash
numpy
pandas
opencv-python
matplotlib
scikit-learn
scikit-image
seaborn
joblib
```

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure
```
plant-pathology-classification/
├── image-classification (3).ipynb  # Main Jupyter Notebook with classification code
├── requirements.txt                # Python dependencies
├── sample_predictions_SVM.png      # Visualization of SVM predictions
├── sample_predictions_RandomForest.png  # Visualization of Random Forest predictions
├── confusion_matrices_comparison.png    # Confusion matrices for both models
├── model_comparison.png            # Bar plot comparing model metrics
├── feature_importance_RandomForest.png  # Feature importance for Random Forest
└── README.md                      # This file
```

## Usage
1. **Download the Dataset**:
   - Download the Plant Pathology 2020 FGVC7 dataset from [Kaggle](https://www.kaggle.com/c/plant-pathology-2020-fgvc7).
   - Place the dataset files in the directory `/kaggle/input/plant-pathology-2020-fgvc7/` with the following structure:
     ```
     /kaggle/input/plant-pathology-2020-fgvc7/
     ├── train.csv
     ├── test.csv
     ├── images/
     │   ├── Train_0.jpg
     │   ├── Train_1.jpg
     │   ├── ...
     │   ├── Test_0.jpg
     │   ├── ...
     ```

2. **Run the Jupyter Notebook**:
   - Open `image-classification (3).ipynb` in Jupyter Notebook or JupyterLab.
   - Ensure the dataset paths in the `main()` function match your local setup if not using Kaggle.
   - Execute the notebook cells sequentially to:
     - Load and preprocess the data
     - Extract handcrafted features
     - Train and evaluate SVM and Random Forest models
     - Generate visualizations (sample predictions, confusion matrices, feature importance)
     - Make predictions on the test set

3. **Expected Outputs**:
   - Console output with:
     - Class distribution in the training set
     - Training and validation sample counts
     - Prediction distributions for both models
     - Detailed metrics (Hamming Loss, F1 Score, Precision, Recall)
   - Visualizations saved as `.png` files:
     - `sample_predictions_SVM.png`
     - `sample_predictions_RandomForest.png`
     - `confusion_matrices_comparison.png`
     - `model_comparison.png`
     - `feature_importance_RandomForest.png`

## Methodology
1. **Feature Extraction**:
   - Images are resized to 128x128 pixels.
   - HOG features are extracted from grayscale images with 8 orientations and 16x16 pixels per cell.
   - Color histograms are computed for each RGB channel (32 bins per channel).
   - Texture features are derived from Sobel edge magnitudes (mean and standard deviation).
   - Features are concatenated into a single vector for each image.

2. **Data Preprocessing**:
   - Features are extracted in parallel using `joblib` for efficiency.
   - Invalid images are filtered out, and corresponding labels are aligned.
   - Features are scaled using `StandardScaler`.

3. **Model Training**:
   - A One-vs-Rest approach is used to train separate classifiers for each label (`healthy`, `multiple_diseases`, `rust`, `scab`).
   - SVM uses an RBF kernel with probability estimates.
   - Random Forest uses 100 estimators with balanced class weights.
   - Predictions are made by selecting the label with the highest probability (≥ 0.3 threshold) to ensure single-label output.

4. **Evaluation**:
   - Models are evaluated on a validation set (20% of training data) using Hamming Loss, F1 Score, Precision, and Recall.
   - Confusion matrices and feature importance are visualized to analyze model performance.
   - A bar plot compares the overall metrics between SVM and Random Forest.

5. **Test Set Prediction**:
   - Features are extracted from test images.
   - Scaled features are passed through trained models to generate predictions.
   - Sample test predictions are visualized.

## Results
The project evaluates the performance of SVM and Random Forest models on the validation set. Key observations:
- **Hamming Loss**: Measures the fraction of incorrect labels.
- **F1 Score**: Balances precision and recall for multi-label classification.
- **Precision and Recall**: Computed per label and averaged across samples.
- **None Predictions**: Percentage of samples with no assigned label (when all probabilities are below the threshold).

Visualizations provide insights into:
- **Sample Predictions**: Display predicted and true labels for validation images.
- **Confusion Matrices**: Show true vs. predicted labels for each category and model.
- **Feature Importance**: Highlights the most influential features for Random Forest predictions.
- **Model Comparison**: Compares metrics across models in a bar plot.

Example output (actual values depend on the dataset):
```
Model Comparison:
SVM:
  Hamming Loss: 0.1250
  F1 Score: 0.7500
  Precision: 0.8000
  Recall: 0.7200
  None Predictions: 5.00%
RandomForest:
  Hamming Loss: 0.1000
  F1 Score: 0.7800
  Precision: 0.8200
  Recall: 0.7500
  None Predictions: 3.00%
```

## Limitations
- The dataset may have class imbalance, particularly for `multiple_diseases`, which could affect model performance.
- The single-label output (based on max probability) may not fully capture multi-label scenarios.
- Handcrafted features may not capture all relevant information compared to deep learning approaches.
- The code assumes the dataset is located at `/kaggle/input/plant-pathology-2020-fgvc7/`. Adjust paths for local execution.

## Future Improvements
- Experiment with additional handcrafted features (e.g., SIFT, SURF).
- Try other traditional ML models like Gradient Boosting Machines.
- Implement feature selection to reduce dimensionality and improve performance.
- Address class imbalance using techniques like SMOTE.
- Explore ensemble methods combining SVM and Random Forest.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The [Plant Pathology 2020 FGVC7 dataset](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) provided by Kaggle.
- Libraries: NumPy, Pandas, OpenCV, scikit-learn, scikit-image, Matplotlib, Seaborn, and Joblib.
