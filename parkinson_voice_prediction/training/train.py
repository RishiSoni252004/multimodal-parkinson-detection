import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Headless mode

from feature_extraction.extract_features import process_audio_directory

def load_or_extract_data(dataset_path='dataset/', default_csv='dataset/parkinsons.data'):
    """
    Loads dataset from a CSV/Data file, or extracts features from an audio directory if CSV is not found.
    Handles 'status' column as the target label.
    """
    if os.path.exists(default_csv):
        print(f"Loading dataset from {default_csv}...")
        df = pd.read_csv(default_csv)
        # Assuming we don't need 'name' for training
        if 'name' in df.columns:
            df = df.drop(columns=['name'])
    else:
        print(f"Dataset not found at {default_csv}. Attempting to extract audio features from {dataset_path}...")
        features_csv = 'dataset/extracted_features.csv'
        if os.path.exists(features_csv):
            df = pd.read_csv(features_csv)
        else:
            df = process_audio_directory(dataset_path, features_csv)
            if df.empty:
                raise FileNotFoundError("Dataset is empty or could not extract features. Provide a valid dataset.")
            
    return df

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluates a model and prints metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("-" * 30)
    
    return acc, prec, rec, f1

def train_and_compare_models():
    """
    Trains RF, SVM, XGBoost, compares them, and saves the best model.
    """
    df = load_or_extract_data()
    
    # The uploaded dataset has 'status' as the target column.
    if 'status' in df.columns:
        X = df.drop(columns=['status'])
        y = df['status']
        # Re-assign 'target' so downstream pie chart logic doesn't break
        df['target'] = df['status']
    elif 'target' in df.columns:
        X = df.drop(columns=['target'])
        y = df['target']
    else:
        # Fallback if user provides a generic CSV
        columns = df.columns
        X = df.drop(columns=[columns[-1]])
        y = df[columns[-1]]
        df['target'] = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_model_name = None
    best_model = None
    best_f1 = -1
    
    # Stores metrics to plot comparison
    model_metrics = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
    
    # Create plot directory
    plot_dir = 'frontend/static/plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Dataset Class Distribution Chart
    plt.figure(figsize=(6, 6))
    df['target'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Healthy', 'Parkinson\'s'], startangle=90, colors=['#4ade80', '#f87171'])
    plt.title('Dataset Class Distribution')
    plt.ylabel('')
    plt.savefig(os.path.join(plot_dir, 'dataset_distribution.png'))
    plt.close()
    
    # 2. Feature Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = X.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'feature_correlation.png'))
    plt.close()

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc, prec, rec, f1 = evaluate_model(y_test, y_pred, name)
        
        model_metrics['Model'].append(name)
        model_metrics['Accuracy'].append(acc)
        model_metrics['Precision'].append(prec)
        model_metrics['Recall'].append(rec)
        model_metrics['F1 Score'].append(f1)
        
        # We select the best model primarily based on F1 score as it's typically balanced
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    # 3. Model Comparison Chart
    metrics_df = pd.DataFrame(model_metrics)
    metrics_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_melted, x="Metric", y="Score", hue="Model", palette="viridis")
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'model_comparison.png'))
    plt.close()

    print(f">>> Best Model: {best_model_name} with F1 Score: {best_f1:.4f} <<<")
    
    # Save the best model
    # We save feature names to use them in SHAP later
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(list(X.columns), 'models/feature_names.pkl')
    print("Best model, scaler, and feature names saved to 'models/' directory.")

if __name__ == "__main__":
    train_and_compare_models()
