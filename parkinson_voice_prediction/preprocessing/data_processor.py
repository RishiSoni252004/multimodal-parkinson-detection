import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

class DataProcessor:
    def __init__(self, dataset_path="dataset/parkinsons.data"):
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_and_clean_data(self):
        print(f"Loading dataset from {self.dataset_path}...")
        try:
            df = pd.read_csv(self.dataset_path)
        except Exception:
            raise FileNotFoundError(f"Could not load dataset from {self.dataset_path}")

        # Remove duplicates
        initial_len = len(df)
        df.drop_duplicates(inplace=True)
        print(f"Removed {initial_len - len(df)} duplicate rows.")

        # Handle missing values (fill with median)
        df.fillna(df.median(numeric_only=True), inplace=True)

        # Remove 'name' column as requested
        if 'name' in df.columns:
            df.drop(columns=['name'], inplace=True)

        # Handle Outliers using IQR method for features (not target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'status' in numeric_cols:
            numeric_cols.remove('status')
            
        print("Capping outliers using the IQR method...")
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap the values
            df[col] = np.where(df[col] > upper_bound, upper_bound, 
                               np.where(df[col] < lower_bound, lower_bound, df[col]))
        
        # Save cleaned dataset
        cleaned_path = self.dataset_path.replace(".csv", "_cleaned.csv")
        df.to_csv(cleaned_path, index=False)
        print(f"Saved cleaned dataset to {cleaned_path}")

        # Extract features and target (target column is 'status')
        target_col = 'status'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
            
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.feature_names = list(X.columns)
        print(f"Dataset Shape: X={X.shape}, y={y.shape}")
        
        return X, y

    def preprocess_data(self, test_size=0.2, random_state=42, feature_selection=None, k_features=50):
        """
        Executes full pipeline: Train/Test Split -> SMOTE -> Standardization -> Feature Selection
        feature_selection options: 'selectkbest', 'rfe', None
        """
        X, y = self.load_and_clean_data()

        # Split into train/test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Apply SMOTE to training data ONLY to prevent data leakage
        print("Applying SMOTE...")
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"Class distribution after SMOTE: {dict(pd.Series(y_train_res).value_counts())}")

        # Standardization 
        X_train_scaled = self.scaler.fit_transform(X_train_res)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler for later use
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.scaler, "models/scaler.pkl")

        # Convert back to DataFrame to preserve column names
        X_train_df = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test_df = pd.DataFrame(X_test_scaled, columns=self.feature_names)

        # Feature Selection
        if feature_selection == 'selectkbest':
            print(f"Applying SelectKBest (k={k_features})...")
            selector = SelectKBest(score_func=f_classif, k=k_features)
            selector.fit(X_train_df, y_train_res)
            
            selected_cols = X_train_df.columns[selector.get_support()]
            X_train_final = X_train_df[selected_cols]
            X_test_final = X_test_df[selected_cols]
            
            joblib.dump(selected_cols, "models/selected_features.pkl")
            
        elif feature_selection == 'rfe':
            print(f"Applying Recursive Feature Elimination (n_features_to_select={k_features})...")
            # Using Random Forest as estimator for RFE
            estimator = RandomForestClassifier(n_estimators=50, random_state=random_state)
            selector = RFE(estimator, n_features_to_select=k_features, step=10)
            selector.fit(X_train_df, y_train_res)
            
            selected_cols = X_train_df.columns[selector.get_support()]
            X_train_final = X_train_df[selected_cols]
            X_test_final = X_test_df[selected_cols]
            
            joblib.dump(selected_cols, "models/selected_features.pkl")
            
        else:
            print("No feature selection applied.")
            X_train_final = X_train_df
            X_test_final = X_test_df
            joblib.dump(self.feature_names, "models/selected_features.pkl")

        print(f"Final Training Data Shape: {X_train_final.shape}")
        print(f"Final Testing Data Shape: {X_test_final.shape}")

        return X_train_final, X_test_final, y_train_res, y_test
