from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_knn(X_train, y_train):
    print("Training KNN...")
    param_dist = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    knn = KNeighborsClassifier()
    search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=8, cv=3, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best KNN Params: {search.best_params_}")
    return search.best_estimator_
