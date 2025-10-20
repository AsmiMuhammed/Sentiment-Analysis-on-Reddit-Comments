import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# 1️⃣ Load train and test datasets
# ------------------------------
print("Loading train and test datasets...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Drop missing values
train_df.dropna(subset=['comment', 'sentiment'], inplace=True)
test_df.dropna(subset=['comment', 'sentiment'], inplace=True)

# ------------------------------
# 2️⃣ Prepare BERT model and tokenizer
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)
model.eval()

# ------------------------------
# 3️⃣ Function to get BERT embeddings
# ------------------------------
def get_bert_embeddings(text_list):
    embeddings = []
    with torch.no_grad():
        for text in tqdm(text_list, desc="Generating embeddings"):
            inputs = tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)
            outputs = model(**inputs)
            # Use the [CLS] token representation (first token)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding[0])
    return embeddings

# ------------------------------
# 4️⃣ Get BERT embeddings for train and test sets
# ------------------------------
print("Generating BERT embeddings for training data...")
X_train = get_bert_embeddings(train_df['comment'].tolist())
y_train = train_df['sentiment'].tolist()

print("Generating BERT embeddings for test data...")
X_test = get_bert_embeddings(test_df['comment'].tolist())
y_test = test_df['sentiment'].tolist()

# ------------------------------
# 5️⃣ Train SVM classifier
# ------------------------------
print("Training SVM classifier...")
svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(X_train, y_train)

# ------------------------------
# 6️⃣ Evaluate on training and test sets
# ------------------------------
print("Evaluating model...")

# Training performance
y_train_pred = svm.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)

# Test performance
y_test_pred = svm.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

# Reports
print("\n✅ Training Accuracy:", round(train_acc, 4))
print("✅ Testing Accuracy:", round(test_acc, 4))

print("\n📊 Classification Report (Test Set):\n", classification_report(y_test, y_test_pred))

# ------------------------------
# 7️⃣ Confusion Matrix
# ------------------------------
unique_labels = sorted(list(set(y_test)))
cm = confusion_matrix(y_test, y_test_pred, labels=unique_labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.title("Confusion Matrix (BERT + SVM)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
