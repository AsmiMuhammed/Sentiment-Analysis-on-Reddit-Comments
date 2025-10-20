# preprocess_india_data.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# ---- Step 1: Load the annotated dataset ----
print("Loading annotated dataset...")
df = pd.read_csv("india_comments_hf_sentiment.csv")

print("Data loaded successfully!")
print("Total comments:", len(df))
print(df.head())

# ---- Step 2: Download stopwords ----
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---- Step 3: Define text cleaning function ----
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)           # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)          # remove punctuation/numbers
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ---- Step 4: Apply cleaning to all comments ----
print("Cleaning text data...")
df['clean_comment'] = df['comment_body'].apply(clean_text)

# ---- Step 5: Remove empty or NaN comments ----
df = df[df['clean_comment'].str.strip() != ""]
df = df.dropna(subset=['clean_comment'])

print("After cleaning, total comments:", len(df))

# ---- Step 6: Split data into train and test sets ----
X = df['clean_comment']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Step 7: Save processed datasets ----
train_df = pd.DataFrame({'comment': X_train, 'sentiment': y_train})
test_df = pd.DataFrame({'comment': X_test, 'sentiment': y_test})

train_df.to_csv("india_train_preprocessed.csv", index=False, encoding="utf-8")
test_df.to_csv("india_test_preprocessed.csv", index=False, encoding="utf-8")

print("\n✅ Preprocessing complete!")
print("Saved files:")
print("- india_train_preprocessed.csv")
print("- india_test_preprocessed.csv")

print("\nSample cleaned comments:")
print(train_df.head())
