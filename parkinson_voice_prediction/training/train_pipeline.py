import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from preprocessing.data_processor import DataProcessor
from models.svm_model import train_svm
from models.random_forest_model import train_random_forest
from models.decision_tree_model import train_decision_tree
from models.knn_model import train_knn
from models.fnn_model import train_fnn
from models.wav2vec_model import Wav2VecParkinsonModel
from evaluation.evaluator import Evaluator
from visualization.visualizer import Visualizer

def main():
    print("=== Parkinson's Disease Prediction Training Pipeline ===")
    
    # 1. Preprocessing
    processor = DataProcessor()
    X_train, X_test, y_train, y_test = processor.preprocess_data(feature_selection=None)
    
    feature_names = joblib.load("models/selected_features.pkl")
    vis = Visualizer(output_dir="frontend/static/plots")
    metrics_list = []
    roc_dict = {}
    
    best_model = None
    best_f1 = 0
    best_name = ""
    
    # 2. Train ML Models
    models_to_train = {
        "KSVM": train_svm,
        "Random Forest": train_random_forest,
        "Decision Tree": train_decision_tree,
        "KNN": train_knn,
        "FNN": train_fnn
    }
    
    os.makedirs("models", exist_ok=True)
    
    for name, train_func in models_to_train.items():
        print(f"\n--- {name} ---")
        model = train_func(X_train, y_train)
        
        joblib.dump(model, f"models/{name.replace(' ', '_').lower()}_model.pkl")
        y_pred = model.predict(X_test)
        
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            
        eval_result = Evaluator.evaluate_model(y_test, y_pred, y_prob, model_name=name)
        metrics_list.append(eval_result)
        
        if eval_result["f1"] > best_f1:
            best_f1 = eval_result["f1"]
            best_model = model
            best_name = name
            joblib.dump(model, "models/best_model.pkl")
            
        roc_dict[name] = eval_result["roc_data"]
        vis.plot_confusion_matrix(eval_result["confusion_matrix"], name)
        
        if name == "Random Forest":
            vis.plot_feature_importance(model, feature_names, top_n=20, model_name=name)
            
    # 3. Train Wav2Vec2 Model
    print("\n--- Wav2Vec 2.0 ---")
    wav2vec_pipeline = Wav2VecParkinsonModel()
    wav2vec_pipeline.train(dataset_dir="dataset/")
    
    # 4. Generate Comparisons
    print("\nGenerating Visualizations...")
    vis.plot_roc_curves(roc_dict)
    vis.plot_model_comparison(metrics_list)
    
    metrics_df = pd.DataFrame(metrics_list).drop(columns=["confusion_matrix", "roc_data"])
    metrics_df.to_csv("models/evaluation_metrics.csv", index=False)
    
    print(f"\n=== Training Complete ===")
    print(f"Best Multi-Class Model: {best_name} (F1: {best_f1:.4f}) saved.")
    
if __name__ == "__main__":
    main()
