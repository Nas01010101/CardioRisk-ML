import argparse
import matplotlib.pyplot as plt
import shap

from src.data_loader import DataLoader
from src.models import ModelTrainer
from src.evaluation import Evaluation


def main():
    parser = argparse.ArgumentParser(description="Heart Failure Prediction")
    parser.add_argument('--model', default='all', help='logistic_regression, random_forest, xgboost, or all')
    parser.add_argument('--data', default='data/heart_failure_clinical_records_dataset.csv')
    args = parser.parse_args()

    # Load data
    loader = DataLoader(args.data)
    X_train, X_test, y_train, y_test = loader.get_split_data()

    # Setup trainer
    trainer = ModelTrainer(X_train, y_train)
    
    if args.model == 'all':
        model_names = ['logistic_regression', 'random_forest', 'xgboost']
    else:
        model_names = [args.model]

    # Train and evaluate each model
    for name in model_names:
        print(f"\n=== {name} ===")
        
        # Tune
        if name == 'logistic_regression':
            grid = {'C': [0.1, 1, 10]}
        elif name == 'random_forest':
            grid = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
        else:
            grid = {'n_estimators': [100], 'learning_rate': [0.05, 0.1]}
        
        model = trainer.tune(name, grid)
        
        # Evaluate
        y_prob = model.predict_proba(X_test)[:, 1]
        ev = Evaluation(y_test, y_prob, name)
        
        print(ev.get_metrics())
        
        ev.calibration_plot(f"reports/{name}_calibration.png")
        ev.decision_curve(f"reports/{name}_dca.png")

        # SHAP for tree models
        if name in ['random_forest', 'xgboost']:
            explainer = shap.Explainer(model, X_train)
            shap_vals = explainer(X_test)
            
            plt.figure()
            shap.summary_plot(shap_vals, X_test, show=False)
            plt.savefig(f"reports/{name}_shap.png")
            plt.close()


if __name__ == "__main__":
    main()
