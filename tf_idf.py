# sentiment_analysis_tfidf_svm.py

import pandas as pd
from nltk.tokenize import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

# --------------------------
# 1. Load Data
# --------------------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# --------------------------
# 2. Tokenization & Lemmatization
# --------------------------
lemmatizer = WordNetLemmatizer()
tokenizer = ToktokTokenizer()

for df in [train_df, test_df]:
    df['lemmatized_tokens'] = df['comment'].apply(
        lambda x: [lemmatizer.lemmatize(token) for token in tokenizer.tokenize(str(x))]
    )

# --------------------------
# 3. TF-IDF Feature Extraction
# --------------------------
vectorizer_tfidf = TfidfVectorizer(max_features=5000)

X_train = vectorizer_tfidf.fit_transform(train_df['lemmatized_tokens'].apply(lambda x: ' '.join(x)))
X_test = vectorizer_tfidf.transform(test_df['lemmatized_tokens'].apply(lambda x: ' '.join(x)))

X_train = normalize(X_train)
X_test = normalize(X_test)

y_train = train_df['sentiment']
y_test = test_df['sentiment']

print("TF-IDF Train Shape:", X_train.shape)
print("TF-IDF Test Shape:", X_test.shape)

# --------------------------
# 4. SVM Model Training
# --------------------------
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# --------------------------
# 5. Evaluation
# --------------------------
y_train_pred = svm_model.predict(X_train)
y_test_pred = svm_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
classification_rep = classification_report(y_test, y_test_pred)

print("Feature Extraction: TF-IDF")
print("Model: SVM")
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("\nClassification Report:\n", classification_rep)

# --------------------------
# 6. Confusion Matrix
# --------------------------
unique_labels = sorted(list(set(y_test)))
conf_mat = confusion_matrix(y_test, y_test_pred, labels=unique_labels)
print("\nConfusion Matrix:\n", conf_mat)

plt.figure(figsize=(6,5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Purples',
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (TF-IDF + SVM)')
plt.show()

# --------------------------
# 7. Tkinter GUI for Sentiment Prediction
# --------------------------
emoticon_to_emoji = {
    'NEG': '😠 Negative',
    'NEU': '😐 Neutral',
    'POS': '😊 Positive'
}

def analyze_sentiment():
    user_text = text_input.get("1.0", tk.END).strip()
    if not user_text:
        messagebox.showwarning("Input Error", "Please enter a comment!")
        return

    # Preprocess input
    tokens = [lemmatizer.lemmatize(tok) for tok in tokenizer.tokenize(user_text)]
    vectorized = vectorizer_tfidf.transform([' '.join(tokens)])
    vectorized = normalize(vectorized)

    # Predict sentiment
    pred = svm_model.predict(vectorized)[0]
    result_label.config(text=f"Predicted Sentiment: {emoticon_to_emoji.get(pred, pred)}")

# GUI Setup
root = tk.Tk()
root.title("Sentiment Analysis (TF-IDF + SVM)")
root.geometry("500x400")
root.configure(bg="#F0F4F8")

tk.Label(root, text="Enter your comment:", bg="#F0F4F8", font=("Arial", 12)).pack(pady=10)
text_input = tk.Text(root, height=6, width=50, font=("Arial", 11))
text_input.pack(pady=5)

tk.Button(root, text="Analyze Sentiment", command=analyze_sentiment,
          bg="#8A2BE2", fg="white", font=("Arial", 12), padx=10, pady=5).pack(pady=10)

result_label = tk.Label(root, text="", bg="#F0F4F8", font=("Arial", 13, "bold"))
result_label.pack(pady=20)

root.mainloop()
