# Neural Network vs. KNN for Handwriting Recognition

## Overview
This project compares the performance of **K-Nearest Neighbors (KNN)** and **Neural Networks** in handwriting recognition using a dataset of handwritten characters. The goal is to identify the best-performing model for accurate classification.

## Dataset
The dataset used for training and evaluation is `letters.csv`, which contains labeled handwritten characters.

## Models Implemented
1. **K-Nearest Neighbors (KNN) Model**  
   - Uses 20 neighbors  
   - Distance metric: Euclidean  
   - Accuracy: **67%**  

2. **Neural Network Model**  
   - Consists of dense layers with **ReLU** activation  
   - Output layer with **SoftMax** activation  
   - Accuracy: **70%**  

## Key Performance Metrics
| Metric       | KNN Model | Neural Network Model |
|-------------|----------|---------------------|
| Accuracy    | 67%      | 70%                 |
| Precision   | 67%      | 71%                 |
| Recall      | 66%      | 70%                 |
| F1-Score    | 66%      | 70%                 |

## Findings
- The **Neural Network model** outperformed the KNN model across all key performance metrics.
- The **Neural Network model** demonstrated better generalization in distinguishing different handwriting styles.
- **Challenges faced** include class imbalance, hyperparameter tuning, and dimensionality issues.

## Recommendation
The **Neural Network model** is recommended for handwriting recognition due to its superior performance and adaptability.

## Repository Structure

📂 Neural-Network-vs-KNN

📄 README.md  # Project overview

📄 TextClassification.ipynb  # Jupyter Notebook with implementation

📄 TextClassification.docx  # Detailed report with analysis

📂 data

letters.csv  # Handwriting dataset

📂 models

knn_model.pkl  # Trained KNN model

neural_network_model.h5  # Trained Neural Network model


## How to Run
1. Install dependencies:  
   ```bash
   pip install numpy pandas scikit-learn tensorflow
   ```
2. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook TextClassification.ipynb
   ```

## Contributors
- **Akhila Singaraju**
