from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_decision_tree(X_train, y_train):
    print("Training Decision Tree...")
    param_dist = {
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 20],
        'criterion': ['gini', 'entropy']
    }
    dt = DecisionTreeClassifier(random_state=42)
    search = RandomizedSearchCV(dt, param_distributions=param_dist, n_iter=8, cv=3, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best DT Params: {search.best_params_}")
    return search.best_estimator_
