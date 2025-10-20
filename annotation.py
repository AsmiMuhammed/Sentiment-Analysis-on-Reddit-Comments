# india_sentiment_hf_plot.py

import pandas as pd
import json
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns


# ---- Step 1: Load existing JSON file ----
with open("india_comments.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# ---- Step 2: Load Hugging Face sentiment model ----
classifier = pipeline("sentiment-analysis")  # default: distilbert-base-uncased

# ---- Step 3: Define sentiment function with Neutral detection ----
def get_sentiment(comment):
    result = classifier(comment[:512])[0]  # limit to first 512 chars
    label = result['label']
    score = result['score']
    if score < 0.6:
        return "Neutral"
    else:
        return "Positive" if label == "POSITIVE" else "Negative"

# ---- Step 4: Apply sentiment analysis ----
print("Running sentiment analysis on comments...")
df['sentiment'] = df['comment_body'].apply(get_sentiment)

# ---- Step 5: Save results ----
output_file = "labelled_data.csv"
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Sentiment analysis completed. Results saved as {output_file}")

# ---- Step 6: Plot sentiment distribution ----
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='sentiment', data=df, palette='Set2')
plt.title("Sentiment Distribution of r/India Comments")
plt.xlabel("Sentiment")
plt.ylabel("Number of Comments")

# Add percentage labels on top of bars
total = len(df)
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height} ({height/total:.1%})', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("sentiment_distribution.png", dpi=300)  # saves the plot
plt.show()


