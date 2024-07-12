# Disaster Tweet Classification

## Overview
This project aims to classify tweets to determine whether they are about real disasters or not. The project involves three main experiments using different machine learning and deep learning models to perform this classification task.

## Data
The dataset consists of tweets that are labeled as either related to disasters (1) or not (0). The data is divided into training and test sets:
- `train.csv`: Contains tweet texts and their corresponding labels.
- `test.csv`: Contains tweet texts only.

## Experiments
### Experiment 1: Logistic Regression
- **Description**: A simple logistic regression model with TF-IDF vectorization.
- **Validation F1 Score**: 0.79742

### Experiment 2: LSTM (Long Short-Term Memory)
- **Description**: An LSTM model that captures long-term dependencies in the text data.
- **Validation F1 Score**: 0.78210

### Experiment 3: BERT (Bidirectional Encoder Representations from Transformers)
- **Description**: A BERT model fine-tuned for the disaster tweet classification task.
- **Validation F1 Score**: 0.83205

## Results
The BERT model outperformed the Logistic Regression and LSTM models, achieving the highest F1 score on the validation set. Detailed results and analysis are included in the notebook.

## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/srlausten/disaster-tweet-classification.git
   ```
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook to see the experiments and results:
   ```sh
   jupyter notebook disaster_tweet_classification.ipynb
   ```

## Repository Structure
- `data/`: Directory containing the dataset files.
- `disaster_tweet_classification.ipynb`: Jupyter notebook with detailed experiments and analysis.

## Conclusion
The project demonstrates the effectiveness of using BERT for text classification tasks, particularly for disaster tweet classification. Future work could involve further hyperparameter tuning, data augmentation, and exploring more advanced models.

## References
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Logistic Regression Explained](https://www.statisticssolutions.com/what-is-logistic-regression/)
