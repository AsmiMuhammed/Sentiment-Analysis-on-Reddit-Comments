SENTIMENT ANALYSIS ON REDDIT COMMENTS

This project performs sentiment analysis on Reddit comments to determine whether a comment expresses a positive, negative, or neutral opinion. It uses Python, the Reddit API (PRAW) for data collection, and machine learning models for classification.

PROJECT OVERVIEW

Reddit contains millions of user comments that reflect people’s opinions on various topics. This project collects comments from selected Reddit threads and uses a trained model to predict the sentiment of each comment.

TECHNOLOGIES USED

Python 3

PRAW – Reddit API wrapper for Python

Scikit-learn (SVM classifier)

Pandas & NumPy – Data processing

Matplotlib / Seaborn – Data visualization

NLTK / TextBlob – Text preprocessing and sentiment analysis

HOW IT WORKS

Data Collection

Reddit comments are fetched using the Reddit API (praw).

Preprocessing

Text is cleaned (removing stopwords, punctuation, and URLs).

Feature Extraction

Text data is converted into numerical form using TF-IDF.

Model Training

An SVM classifier is trained on labeled sentiment data.

Prediction

The model predicts sentiment for new Reddit comments.

Visualization

The results are visualized using charts and a confusion matrix.

INSTALLATION & USAGE

Step 1: Clone this repository

git clone https://github.com/AsmiMuhammed/Sentiment-Analysis-on-Reddit-Comments.git
cd Sentiment-Analysis-on-Reddit-Comments


Step 2: Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate      (On Windows: venv\Scripts\activate)


Step 3: Install dependencies

pip install -r requirements.txt


Step 4: Configure Reddit API

Create a Reddit App here: https://www.reddit.com/prefs/apps

Note your client_id, client_secret, and user_agent

Add them to a config.py file or directly in the script:

REDDIT_CLIENT_ID = "your_client_id"
REDDIT_CLIENT_SECRET = "your_client_secret"
REDDIT_USER_AGENT = "your_user_agent"


Step 5: Run the project

python main.py


The script will fetch Reddit comments, analyze sentiment, and display results with visualizations.

SAMPLE OUTPUT

Sentiment counts (positive/negative/neutral)

Confusion matrix for model performance

Charts visualizing sentiment distribution

NOTES

Make sure you have an active Reddit account and API credentials.

You can modify the list of subreddits or threads in the script to fetch different data.
