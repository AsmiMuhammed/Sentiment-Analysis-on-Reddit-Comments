import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import pickle
from pathlib import Path
import time

# ------------------------------
# Configuration
# ------------------------------
MODEL_PATH_BERT = "bert_svm_model.pkl"
MODEL_PATH_TFIDF = "tfidf_svm_model.pkl"
MODEL_PATH_BOW = "bow_svm_model.pkl"
VECTORIZER_PATH_TFIDF = "tfidf_vectorizer.pkl"
VECTORIZER_PATH_BOW = "bow_vectorizer.pkl"

# ------------------------------
# 1️⃣ Load Data
# ------------------------------
@st.cache_data
def load_data():
    """Load training and testing data"""
    try:
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")
        
        train_df.dropna(subset=['comment', 'sentiment'], inplace=True)
        test_df.dropna(subset=['comment', 'sentiment'], inplace=True)
        
        return train_df, test_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# ------------------------------
# 2️⃣ BERT Feature Extraction
# ------------------------------
@st.cache_resource
def get_bert_model():
    """Initialize BERT model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()
    return tokenizer, model, device

def extract_bert_features(texts, tokenizer, model, device, batch_size=16):
    """Extract BERT embeddings"""
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT embeddings"):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, 
                             truncation=True, max_length=128).to(device)
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(cls_emb)
    return np.array(embeddings)

# ------------------------------
# 3️⃣ TF-IDF Feature Extraction
# ------------------------------
def extract_tfidf_features(train_texts, test_texts, max_features=5000):
    """Extract TF-IDF features"""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train.toarray(), X_test.toarray(), vectorizer

# ------------------------------
# 4️⃣ BoW Feature Extraction
# ------------------------------
def extract_bow_features(train_texts, test_texts, max_features=5000):
    """Extract Bag of Words features"""
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train.toarray(), X_test.toarray(), vectorizer

# ------------------------------
# 5️⃣ Train and Evaluate
# ------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, method_name):
    """Train SVM and return metrics"""
    start_time = time.time()
    
    svm = SVC(kernel='linear', C=1, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    y_pred = svm.predict(X_test)
    
    metrics = {
        'method': method_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'training_time': training_time,
        'predictions': y_pred,
        'model': svm
    }
    
    return metrics

# ------------------------------
# 6️⃣ Visualization Functions
# ------------------------------
def plot_comparison_metrics(results):
    """Create comparison bar chart for all metrics"""
    metrics_df = pd.DataFrame([
        {
            'Method': r['method'],
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1-Score': r['f1_score']
        }
        for r in results
    ])
    
    fig = go.Figure()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
    
    for idx, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df['Method'],
            y=metrics_df[metric],
            marker_color=colors[idx],
            text=metrics_df[metric].apply(lambda x: f'{x:.3f}'),
            textposition='outside'
        ))
    
    fig.update_layout(
        title='Performance Comparison: BERT vs TF-IDF vs BoW',
        xaxis_title='Feature Extraction Method',
        yaxis_title='Score',
        barmode='group',
        yaxis=dict(range=[0, 1.1]),
        height=500,
        template='plotly_dark'
    )
    
    return fig

def plot_confusion_matrices(results, y_test):
    """Create confusion matrices for all methods"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, result in enumerate(results):
        cm = confusion_matrix(y_test, result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=result['model'].classes_,
                   yticklabels=result['model'].classes_)
        axes[idx].set_title(f"{result['method']} Confusion Matrix")
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    return fig

def plot_training_time(results):
    """Plot training time comparison"""
    time_df = pd.DataFrame([
        {'Method': r['method'], 'Time (seconds)': r['training_time']}
        for r in results
    ])
    
    fig = px.bar(time_df, x='Method', y='Time (seconds)', 
                 color='Method',
                 color_discrete_map={
                     'BERT': '#3b82f6',
                     'TF-IDF': '#10b981',
                     'BoW': '#f59e0b'
                 },
                 text='Time (seconds)',
                 title='Training Time Comparison')
    
    fig.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
    fig.update_layout(showlegend=False, template='plotly_dark', height=400)
    
    return fig

def plot_feature_architecture():
    """Create architecture diagram showing feature extraction process"""
    fig = go.Figure()
    
    # Define positions for each component
    stages = {
        'Input': {'x': 0, 'y': 3, 'color': '#94a3b8'},
        'BERT': {'x': 1, 'y': 5, 'color': '#3b82f6'},
        'TF-IDF': {'x': 1, 'y': 3, 'color': '#10b981'},
        'BoW': {'x': 1, 'y': 1, 'color': '#f59e0b'},
        'SVM': {'x': 2, 'y': 3, 'color': '#ef4444'},
        'Output': {'x': 3, 'y': 3, 'color': '#8b5cf6'}
    }
    
    # Add nodes
    for name, pos in stages.items():
        fig.add_trace(go.Scatter(
            x=[pos['x']], y=[pos['y']],
            mode='markers+text',
            marker=dict(size=60, color=pos['color']),
            text=name,
            textposition='middle center',
            textfont=dict(color='white', size=12, family='Arial Black'),
            hoverinfo='text',
            hovertext=f'{name} Layer',
            showlegend=False
        ))
    
    # Add arrows (edges)
    arrows = [
        ('Input', 'BERT'), ('Input', 'TF-IDF'), ('Input', 'BoW'),
        ('BERT', 'SVM'), ('TF-IDF', 'SVM'), ('BoW', 'SVM'),
        ('SVM', 'Output')
    ]
    
    for start, end in arrows:
        fig.add_trace(go.Scatter(
            x=[stages[start]['x'], stages[end]['x']],
            y=[stages[start]['y'], stages[end]['y']],
            mode='lines',
            line=dict(color='#475569', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title='Feature Extraction Pipeline Architecture',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#0f172a',
        paper_bgcolor='#0f172a',
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# ------------------------------
# 7️⃣ Streamlit UI
# ------------------------------
st.set_page_config(page_title="Feature Extraction Comparison", page_icon="🔬", layout="wide")

st.title("🔬 Sentiment Analysis: Feature Extraction Comparison")
st.write("**Compare BERT vs TF-IDF vs Bag of Words** for Indian Reddit Comments")

# Sidebar
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.selectbox("Choose Mode", 
    ["📊 Compare All Methods", "🧪 Train Individual Method", "🔮 Predict with Best Method"])

st.sidebar.info("""
**Feature Extraction Methods:**
- **BERT**: Deep learning embeddings (768-dim)
- **TF-IDF**: Statistical word importance
- **BoW**: Word frequency counts
""")

# ------------------------------
# MODE 1: Compare All Methods
# ------------------------------
if mode == "📊 Compare All Methods":
    st.header("📊 Comprehensive Comparison")
    
    # Show architecture diagram
    st.subheader("🏗️ Feature Extraction Architecture")
    st.plotly_chart(plot_feature_architecture(), use_container_width=True)
    
    # Load data
    train_df, test_df = load_data()
    if train_df is None:
        st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Samples", len(train_df))
    with col2:
        st.metric("Testing Samples", len(test_df))
    
    if st.button("🚀 Run Comparison", type="primary"):
        train_texts = train_df['comment'].tolist()
        test_texts = test_df['comment'].tolist()
        y_train = train_df['sentiment'].tolist()
        y_test = test_df['sentiment'].tolist()
        
        results = []
        
        # 1. BERT
        st.subheader("1️⃣ Training with BERT...")
        tokenizer, model, device = get_bert_model()
        with st.spinner("Extracting BERT embeddings..."):
            X_train_bert = extract_bert_features(train_texts, tokenizer, model, device)
            X_test_bert = extract_bert_features(test_texts, tokenizer, model, device)
        
        with st.spinner("Training BERT+SVM model..."):
            bert_results = train_and_evaluate(X_train_bert, X_test_bert, y_train, y_test, "BERT")
            results.append(bert_results)
        st.success(f"✅ BERT Accuracy: {bert_results['accuracy']:.4f}")
        
        # 2. TF-IDF
        st.subheader("2️⃣ Training with TF-IDF...")
        with st.spinner("Extracting TF-IDF features..."):
            X_train_tfidf, X_test_tfidf, _ = extract_tfidf_features(train_texts, test_texts)
        
        with st.spinner("Training TF-IDF+SVM model..."):
            tfidf_results = train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test, "TF-IDF")
            results.append(tfidf_results)
        st.success(f"✅ TF-IDF Accuracy: {tfidf_results['accuracy']:.4f}")
        
        # 3. BoW
        st.subheader("3️⃣ Training with Bag of Words...")
        with st.spinner("Extracting BoW features..."):
            X_train_bow, X_test_bow, _ = extract_bow_features(train_texts, test_texts)
        
        with st.spinner("Training BoW+SVM model..."):
            bow_results = train_and_evaluate(X_train_bow, X_test_bow, y_train, y_test, "BoW")
            results.append(bow_results)
        st.success(f"✅ BoW Accuracy: {bow_results['accuracy']:.4f}")
        
        # Display Results
        st.header("📈 Comparison Results")
        
        # Metrics comparison
        st.plotly_chart(plot_comparison_metrics(results), use_container_width=True)
        
        # Training time comparison
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_training_time(results), use_container_width=True)
        
        with col2:
            # Best method
            best_method = max(results, key=lambda x: x['accuracy'])
            st.metric("🏆 Best Method", best_method['method'], 
                     f"{best_method['accuracy']:.4f}")
            
            # Detailed metrics table
            st.subheader("📋 Detailed Metrics")
            metrics_table = pd.DataFrame([
                {
                    'Method': r['method'],
                    'Accuracy': f"{r['accuracy']:.4f}",
                    'Precision': f"{r['precision']:.4f}",
                    'Recall': f"{r['recall']:.4f}",
                    'F1-Score': f"{r['f1_score']:.4f}",
                    'Time (s)': f"{r['training_time']:.2f}"
                }
                for r in results
            ])
            st.dataframe(metrics_table, use_container_width=True)
        
        # Confusion matrices
        st.subheader("🔹 Confusion Matrices Comparison")
        fig_cm = plot_confusion_matrices(results, y_test)
        st.pyplot(fig_cm)
        
        # Per-class performance
        st.subheader("📊 Per-Class Performance")
        
        for result in results:
            with st.expander(f"📄 {result['method']} Classification Report"):
                report = classification_report(y_test, result['predictions'], 
                                              target_names=result['model'].classes_)
                st.text(report)

# ------------------------------
# MODE 2: Train Individual Method
# ------------------------------
elif mode == "🧪 Train Individual Method":
    st.header("🧪 Train Individual Method")
    
    method = st.selectbox("Select Feature Extraction Method", 
                         ["BERT", "TF-IDF", "Bag of Words"])
    
    train_df, test_df = load_data()
    if train_df is None:
        st.stop()
    
    if st.button(f"🚀 Train {method} Model", type="primary"):
        train_texts = train_df['comment'].tolist()
        test_texts = test_df['comment'].tolist()
        y_train = train_df['sentiment'].tolist()
        y_test = test_df['sentiment'].tolist()
        
        if method == "BERT":
            tokenizer, model, device = get_bert_model()
            with st.spinner("Extracting BERT embeddings..."):
                X_train = extract_bert_features(train_texts, tokenizer, model, device)
                X_test = extract_bert_features(test_texts, tokenizer, model, device)
        
        elif method == "TF-IDF":
            with st.spinner("Extracting TF-IDF features..."):
                X_train, X_test, vectorizer = extract_tfidf_features(train_texts, test_texts)
                with open(VECTORIZER_PATH_TFIDF, 'wb') as f:
                    pickle.dump(vectorizer, f)
        
        else:  # BoW
            with st.spinner("Extracting BoW features..."):
                X_train, X_test, vectorizer = extract_bow_features(train_texts, test_texts)
                with open(VECTORIZER_PATH_BOW, 'wb') as f:
                    pickle.dump(vectorizer, f)
        
        with st.spinner(f"Training {method}+SVM model..."):
            results = train_and_evaluate(X_train, X_test, y_train, y_test, method)
        
        # Save model
        model_path = MODEL_PATH_BERT if method == "BERT" else \
                     MODEL_PATH_TFIDF if method == "TF-IDF" else MODEL_PATH_BOW
        with open(model_path, 'wb') as f:
            pickle.dump(results['model'], f)
        
        st.success(f"✅ {method} Model Trained and Saved!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
        with col2:
            st.metric("F1-Score", f"{results['f1_score']:.4f}")
        with col3:
            st.metric("Training Time", f"{results['training_time']:.2f}s")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, results['predictions'])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"{method} Confusion Matrix")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

# ------------------------------
# MODE 3: Predict with Best Method
# ------------------------------
elif mode == "🔮 Predict with Best Method":
    st.header("🔮 Predict Sentiment")
    
    method = st.selectbox("Select Method", ["BERT", "TF-IDF", "Bag of Words"])
    
    text_input = st.text_area("Enter Indian Reddit comment:", 
                             placeholder="Type your comment here...", 
                             height=150)
    
    if st.button("Analyze Sentiment", type="primary"):
        if text_input.strip():
            # Load model
            model_path = MODEL_PATH_BERT if method == "BERT" else \
                        MODEL_PATH_TFIDF if method == "TF-IDF" else MODEL_PATH_BOW
            
            if not Path(model_path).exists():
                st.error(f"Model not found. Please train {method} model first.")
                st.stop()
            
            with open(model_path, 'rb') as f:
                svm_model = pickle.load(f)
            
            # Extract features
            with st.spinner("Extracting features..."):
                if method == "BERT":
                    tokenizer, model, device = get_bert_model()
                    features = extract_bert_features([text_input], tokenizer, model, device)
                elif method == "TF-IDF":
                    with open(VECTORIZER_PATH_TFIDF, 'rb') as f:
                        vectorizer = pickle.load(f)
                    features = vectorizer.transform([text_input]).toarray()
                else:  # BoW
                    with open(VECTORIZER_PATH_BOW, 'rb') as f:
                        vectorizer = pickle.load(f)
                    features = vectorizer.transform([text_input]).toarray()
            
            # Predict
            prediction = svm_model.predict(features)[0]
            probabilities = svm_model.predict_proba(features)[0]
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sentiment", prediction)
            with col2:
                st.metric("Confidence", f"{max(probabilities):.2%}")
            with col3:
                emoji = "😊" if prediction == "positive" else "😞" if prediction == "negative" else "😐"
                st.metric("Emoji", emoji)
            
            # Probability chart
            prob_df = pd.DataFrame({
                'Sentiment': svm_model.classes_,
                'Probability': probabilities
            })
            fig = px.bar(prob_df, x='Sentiment', y='Probability', 
                        color='Sentiment',
                        color_discrete_map={'positive': '#10b981', 
                                          'negative': '#ef4444', 
                                          'neutral': '#6b7280'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter text to analyze.")

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Dataset:** Indian Reddit Users Comments")
st.sidebar.markdown("**🎯 Task:** Sentiment Classification")