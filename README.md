# Twitter-Sentiment-Analysis-using-NLP
A sentiment analysis program developed in Python to classify tweets as positive or negative. This project uses natural language processing (NLP) techniques for text preprocessing, feature engineering with TF-IDF, and machine learning models for classification.

📜 Project Overview
The goal of this project is to build a model that can accurately predict the sentiment of a tweet. The dataset contains tweets labeled as either non-hateful (label 0) or hateful/negative (label 1). The project workflow involves cleaning the raw tweet data, transforming the text into numerical features, and then training two powerful classification models: Support Vector Machine (SVM) and Random Forest. Both models achieved over 98% accuracy on the training data, demonstrating strong performance in identifying sentiment.

✨ Features
Data Handling: Efficiently loads and processes data using the Pandas library.

Text Preprocessing: Cleans raw tweet text by:

Removing URLs, user mentions (@user), and hashtags (#).

Stripping punctuation and numbers.

Converting text to lowercase for uniformity.

Feature Engineering: Converts cleaned text into a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features using Scikit-learn's TfidfVectorizer.

Model Training: Implements and trains two distinct machine learning models:

Support Vector Machine (SVC)

Random Forest Classifier

Evaluation: Assesses model performance using metrics like Accuracy, Precision, Recall, F1-Score, and a detailed Classification Report.

Prediction: Applies the trained models to a separate test dataset to predict sentiment labels and saves the results.

Data Visualization: Includes various plots (histograms, bar plots, scatter plots, box plots, and pair plots) to analyze the distribution of tweet lengths.

🛠️ Technologies Used
Python 3.10

Pandas: For data manipulation and CSV I/O.

Scikit-learn: For TF-IDF feature extraction, model training (SVM, Random Forest), and performance metrics.

Re (Regex): For text cleaning and pattern matching.

Matplotlib & Seaborn: For data visualization.

Jupyter Notebook: For interactive development and analysis.

📊 Dataset
The project uses a Twitter sentiment dataset, which is split into two files:

train_E6oV3lV.csv: Contains the tweet ID, the tweet text, and its corresponding sentiment label.

label = 0: Non-hateful (Positive/Neutral)

label = 1: Hateful (Negative)

test_tweets_anuFYb8.csv: Contains the tweet ID and the tweet text for prediction.

⚙️ Setup and Installation
To run this project locally, follow these steps:

Clone the repository:

Bash

git clone https://github.com/your-username/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
Create a virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required dependencies:

Bash

pip install -r requirements.txt
(Note: If a requirements.txt file is not provided, you can create one with the following content or install the libraries manually):

pandas
scikit-learn
matplotlib
seaborn
notebook
🚀 Usage
Place the dataset files (train_E6oV3lV.csv and test_tweets_anuFYb8.csv) in the root directory of the project.

Launch the Jupyter Notebook:

Bash

jupyter notebook
Open the phase5.ipynb notebook and run the cells sequentially to see the data processing, model training, evaluation, and prediction steps.

The final output with predictions on the test set will be saved as test_predictions.csv.

📈 Methodology
The project follows a standard machine learning workflow for NLP:

Data Loading & Cleaning: The training and testing datasets are loaded. A custom function clean_text_simple is applied to each tweet to remove noise like URLs, mentions, punctuation, and numbers, and to convert the text to lowercase.

Feature Extraction: The cleaned tweets are converted into numerical data using TfidfVectorizer. This technique reflects the importance of a word in a tweet relative to its frequency across all tweets. The vectorizer was configured to consider the top 5000 features.

Model Training:

Support Vector Machine (SVM): An SVC model is trained on the TF-IDF features from the training data.

Random Forest: A RandomForestClassifier is also trained on the same data.

Evaluation & Prediction:

The performance of both models is evaluated on the training data to establish a baseline of their learning capability.

The trained models are then used to predict sentiment on the unseen test data.

The predictions are mapped to "positive" and "negative" labels for clarity.

📊 Results
Both models performed exceptionally well on the training data. The evaluation metrics are as follows:

Model	Accuracy	Precision (Weighted)	Recall (Weighted)	F1-Score (Weighted)
SVM	98.34%	98.37%	98.34%	98.24%
Random Forest	99.97%	99.97%	99.97%	99.97%

Export to Sheets
The near-perfect score of the Random Forest model on the training set suggests it has learned the data very well, though this could also indicate some overfitting. The SVM model also shows very strong generalization capabilities with 98.34% accuracy. The classification reports show that both models are highly effective at identifying non-hateful tweets (label 0), with the SVM's recall for hateful tweets (label 1) at 77%.

📂 File Structure
.
├── phase5.ipynb              # Main Jupyter Notebook with all the code
├── train_E6oV3lV.csv         # Training data
├── test_tweets_anuFYb8.csv     # Test data for prediction
├── test_predictions.csv      # Output file with predicted sentiments
└── README.md                 # This file
