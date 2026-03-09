import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

class Visualizer:
    def __init__(self, output_dir="frontend/static/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_confusion_matrix(self, cm, model_name):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Parkinson'], yticklabels=['Healthy', 'Parkinson'])
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        out_path = os.path.join(self.output_dir, f'cm_{model_name.replace(" ", "_").lower()}.png')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_roc_curves(self, roc_data_dict):
        plt.figure(figsize=(8, 6))
        for model_name, data in roc_data_dict.items():
            if data is not None:
                fpr, tpr, roc_auc = data
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
                
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        out_path = os.path.join(self.output_dir, 'roc_curves.png')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_model_comparison(self, metrics_list):
        df = pd.DataFrame(metrics_list)
        df_plot = df[["model", "accuracy", "precision", "recall", "f1"]]
        df_melted = df_plot.melt(id_vars="model", var_name="Metric", value_name="Score")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melted, x="Metric", y="Score", hue="model", palette="viridis")
        plt.title('Model Performance Comparison')
        plt.ylim(0, 1.1)
        # Move legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        out_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path
        
    def plot_feature_importance(self, model, feature_names, top_n=20, model_name="Random Forest"):
        if not hasattr(model, 'feature_importances_'):
            return None
            
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.title(f"Top {top_n} Feature Importances ({model_name})")
        plt.bar(range(top_n), importances[indices], align="center")
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, top_n])
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(out_path)
        plt.close()
        return out_path
