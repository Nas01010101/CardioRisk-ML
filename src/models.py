from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def get_model(name, params=None):
    """Return a model instance by name."""
    params = params or {}
    
    if name == 'logistic_regression':
        return LogisticRegression(random_state=42, solver='liblinear', **params)
    elif name == 'random_forest':
        return RandomForestClassifier(random_state=42, **params)
    elif name == 'xgboost':
        return XGBClassifier(random_state=42, eval_metric='logloss', **params)
    else:
        raise ValueError(f"Unknown model: {name}")


class ModelTrainer:
    """Train and tune models via GridSearchCV."""
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def tune(self, model_name, param_grid):
        """Run grid search and return best model."""
        print(f"Tuning {model_name}...")
        
        grid = GridSearchCV(
            get_model(model_name),
            param_grid,
            cv=self.cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        grid.fit(self.X_train, self.y_train)
        
        print(f"Best: {grid.best_params_}, AUC={grid.best_score_:.3f}")
        return grid.best_estimator_

    def train_all(self):
        """Train LR, RF, and XGBoost with default grids."""
        models = {}
        
        models['logistic_regression'] = self.tune('logistic_regression', {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2']
        })
        
        models['random_forest'] = self.tune('random_forest', {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        })
        
        models['xgboost'] = self.tune('xgboost', {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        })
        
        return models
