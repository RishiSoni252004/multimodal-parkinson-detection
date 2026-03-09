from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_fnn(X_train, y_train):
    print("Training FNN (Feed Forward Neural Network)...")
    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }
    fnn = MLPClassifier(max_iter=500, random_state=42, early_stopping=True)
    search = RandomizedSearchCV(fnn, param_distributions=param_dist, n_iter=8, cv=3, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best FNN Params: {search.best_params_}")
    return search.best_estimator_
