from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

def train_svm(X_train, y_train):
    print("Training KSVM...")
    param_dist = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
    svm = SVC(probability=True, random_state=42)
    search = RandomizedSearchCV(svm, param_distributions=param_dist, n_iter=8, cv=3, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best KSVM Params: {search.best_params_}")
    return search.best_estimator_
