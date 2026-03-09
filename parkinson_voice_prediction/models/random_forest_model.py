from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_random_forest(X_train, y_train):
    print("Training Random Forest...")
    param_dist = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=8, cv=3, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best RF Params: {search.best_params_}")
    return search.best_estimator_
