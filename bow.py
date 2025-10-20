import pandas as pd
from nltk.tokenize import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
# 3. Bag-of-Words Feature Extraction
# --------------------------
vectorizer_bow = CountVectorizer()

X_train = vectorizer_bow.fit_transform(train_df['lemmatized_tokens'].apply(lambda x: ' '.join(x)))
X_test = vectorizer_bow.transform(test_df['lemmatized_tokens'].apply(lambda x: ' '.join(x)))

X_train = normalize(X_train)
X_test = normalize(X_test)

y_train = train_df['sentiment']
y_test = test_df['sentiment']

print("BOW Train Shape:", X_train.shape)
print("BOW Test Shape:", X_test.shape)

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

print("Feature Extraction: BOW")
print("Model: SVM")
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("\nClassification Report:\n", classification_rep)

# --------------------------
# 6. Confusion Matrix
# --------------------------
# Automatically get labels present in y_test
labels = sorted(y_test.unique())

conf_mat = confusion_matrix(y_test, y_test_pred, labels=labels)
print("\nConfusion Matrix:\n", conf_mat)

plt.figure(figsize=(6,5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
