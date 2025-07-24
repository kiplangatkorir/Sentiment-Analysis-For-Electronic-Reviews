# Sentiment Analysis for Electronics Reviews

## Overview
This project performs sentiment analysis on electronics product reviews using DistilBERT-base-uncased, comparing zero-shot, fine-tuned, and 5-fold cross-validated performance for a Machine Learning Interview Task.

## Dataset
- `electronics_reviews.csv`: Contains review text, sentiment (positive, neutral, negative), product category, feature mentioned, and rating.
- **Data Issues**: Original dataset had 9573 duplicate reviews and 404 train-test overlaps, resolved by deduplication and re-splitting.
- **Cleaned Dataset**: 426 reviews (298 training: 101 negative, 101 neutral, 96 positive; 64 validation; 64 test).
- **Note**: Small dataset size may lead to overfitting and inflated metrics.

## Methodology
- **Model**: DistilBERT-base-uncased.
- **Zero-shot Evaluation**: Tested pretrained model on the test set.
- **Fine-tuning**: Trained for 3 epochs with learning rate 2e-5, batch size 16, weight decay 0.01.
- **Cross-Validation**: 5-fold stratified cross-validation on training data (~239 train, ~59 validation per fold).
- **Metrics**: Accuracy, macro-averaged precision, recall, F1-score, confusion matrix.
- **Error Analysis**: Misclassified zero-shot reviews in `results/zero_shot_errors.csv`.

## Results
- **Zero-shot Metrics**:
  - Accuracy: 0.3438
  - Precision: 0.1146
  - Recall: 0.3333
  - F1: 0.1705
  - Confusion Matrix: `[[22, 0, 0], [21, 0, 0], [21, 0, 0]]`
- **Fine-tuned Metrics (Test Set)**:
  - Accuracy: 1.0000
  - Precision: 1.0000
  - Recall: 1.0000
  - F1: 1.0000
  - Confusion Matrix: `[[22, 0, 0], [0, 21, 0], [0, 0, 21]]`
- **Cross-Validation Metrics**:
  - Average: Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1: 1.0000
  - Per-fold: See `results/cross_val_metrics.csv`
- **Training Progress (Full Training)**:
  - Epoch 1: Training Loss: 1.050700, Validation Loss: 0.731144
  - Epoch 2: Training Loss: 0.638800, Validation Loss: 0.369486
  - Epoch 3: Training Loss: 0.352600, Validation Loss: 0.264225
- **Loss Curves**: `results/loss_curve.png`
- **Confusion Matrices**: `results/confusion_matrix.png` (fine-tuned), `results/fold_{1-5}_confusion_matrix.png` (cross-validation)
- **Performance Comparison**: `results/performance_comparison.csv`

## Insights
- **Zero-shot**: Strong negative bias, predicting all 64 test reviews as negative (Accuracy: 0.3438). Neutral reviews (e.g., “not bad, not great”) and positive reviews (e.g., “works as expected”) misclassified due to lack of domain-specific training.
- **Fine-tuned/Cross-Validation**: Perfect metrics (1.0000) suggest the cleaned dataset (426 samples) is too small and likely contains simplistic reviews with explicit sentiment keywords (e.g., “great,” “terrible”).
- **Error Analysis**: Zero-shot errors (42/64 reviews) in `results/zero_shot_errors.csv` show neutral and positive reviews misclassified as negative due to ambiguous phrasing or subtle praise.
- **Data Issue**: Small dataset size (426 reviews) limits model generalization and inflates metrics.

## Challenges
- Resolved data leakage (9573 duplicates, 404 overlaps) but resulting dataset is too small (426 samples).
- Perfect metrics indicate potential overfitting to simplistic reviews.
- Handling ambiguous neutral reviews in zero-shot settings.

## Recommendations
- **Test on External Dataset**: Evaluate the fine-tuned model on a larger, diverse dataset (e.g., Amazon reviews) to confirm generalization.
- **Data Augmentation**: Use paraphrasing (e.g., via T5 or GPT) to increase dataset size.
- **Increase Epochs**: If dataset is augmented, try 5 epochs to improve learning.
- **Analyze Reviews**: Inspect `electronics_reviews.csv` for simplistic patterns (e.g., explicit keywords).

## How to Run
1. Clone this repository.
2. Install dependencies: `pip install transformers torch pandas scikit-learn matplotlib seaborn numpy`.
3. Run `sentiment_analysis.ipynb` in Google Colab with GPU.
4. Place `electronics_reviews.csv` in the working directory or Google Drive.

## Deliverables
- Notebook: `sentiment_analysis.ipynb`
- Dataset: `electronics_reviews.csv`
- Results: `results/zero_shot_errors.csv`, `results/performance_comparison.csv`, `results/cross_val_metrics.csv`, `results/loss_curve.png`, `results/confusion_matrix.png`, `results/fold_{1-5}_confusion_matrix.png`
- Fine-tuned model: `results/fine_tuned_model/`
