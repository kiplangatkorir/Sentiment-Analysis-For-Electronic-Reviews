# Sentiment Analysis for Electronics Reviews

## Overview
This project performs sentiment analysis on electronics product reviews using DistilBERT-base-uncased, comparing zero-shot and fine-tuned performance for a Machine Learning Interview Task.

## Dataset
- `electronics_reviews.csv`: Contains review text, sentiment (positive, neutral, negative), product category, feature mentioned, and rating.
- Training set: 6999 reviews (2333 positive, 2333 negative, 2333 neutral).
- Validation and test sets: 1500 each (500 per class).

## Methodology
- **Model**: DistilBERT-base-uncased, chosen for efficiency and performance.
- **Zero-shot Evaluation**: Tested pretrained model on the test set without training.
- **Fine-tuning**: Trained for 3 epochs with learning rate 2e-5, batch size 16, using Hugging Face `Trainer`.
- **Metrics**: Accuracy, macro-averaged precision, recall, F1-score, confusion matrix.
- **Error Analysis**: Misclassified zero-shot reviews saved in `results/zero_shot_errors.csv`.

## Results
- **Zero-shot Metrics**:
  - Accuracy: 0.2107
  - Precision: 0.1376
  - Recall: 0.2107
  - F1: 0.1664
  - Confusion Matrix: `[[86, 414, 0], [270, 230, 0], [357, 143, 0]]`
- **Fine-tuned Metrics**:
  - Accuracy: 1.0000
  - Precision: 1.0000
  - Recall: 1.0000
  - F1: 1.0000
  - Confusion Matrix: `[[500, 0, 0], [0, 500, 0], [0, 0, 500]]`
- **Training Progress**:
  - Epoch 1: Training Loss: 0.001200, Validation Loss: 0.000902
  - Epoch 2: Training Loss: 0.000500, Validation Loss: 0.000354
  - Epoch 3: Training Loss: 0.000400, Validation Loss: 0.000270
- **Loss Curves**: `results/loss_curve.png`
- **Confusion Matrix Plot**: `results/confusion_matrix.png`
- **Performance Comparison**: `results/performance_comparison.csv`

## Insights
- **Zero-shot Performance**:
  - Poor performance (Accuracy: 0.2107, F1: 0.1664) due to negative bias, misclassifying most neutral (270/500) and positive (357/500) reviews.
  - Common errors include neutral reviews with ambiguous phrasing (e.g., “It’s okay, nothing special”) predicted as negative.
- **Fine-tuned Performance**:
  - Perfect metrics (Accuracy: 1.0000) suggest potential data leakage or an overly simplistic test set.
  - Validation loss decreased steadily (0.000902 → 0.000270), indicating effective learning.
- **Error Analysis**:
  - Zero-shot errors saved in `results/zero_shot_errors.csv`. Patterns include ambiguous neutral reviews and mild positive reviews misclassified as negative.
  - No fine-tuned errors, warranting investigation into test set validity.

## Challenges
- Zero-shot model’s bias toward negative predictions.
- Perfect fine-tuned metrics suggest possible data leakage or test set simplicity.
- Handling ambiguous neutral reviews (e.g., “not bad, not great”).

## Recommendations
- Investigate test set for duplicates or overlap with training data.
- Perform k-fold cross-validation to validate fine-tuned performance.
- Analyze additional test data to ensure robustness.

## How to Run
1. Clone this repository.
2. Install dependencies: `pip install transformers torch pandas scikit-learn matplotlib seaborn numpy`.
3. Run `sentiment_analysis.ipynb` in Google Colab with GPU enabled.
4. Place `electronics_reviews.csv` in the working directory or Google Drive.

## Deliverables
- Notebook: `sentiment_analysis.ipynb`
- Dataset: `electronics_reviews.csv`
- Results: `results/zero_shot_errors.csv`, `results/performance_comparison.csv`, `results/loss_curve.png`, `results/confusion_matrix.png`
- Fine-tuned model: `results/fine_tuned_model/`
