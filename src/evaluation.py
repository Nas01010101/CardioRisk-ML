import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, brier_score_loss, classification_report

# Refs:
# - Chicco & Jurman (2020), BMC Medical Informatics
# - Vickers & Elkin (2006), Decision curve analysis


class Evaluation:
    """Compute metrics and generate evaluation plots."""
    
    def __init__(self, y_true, y_prob, model_name="Model"):
        self.y_true = np.array(y_true)
        self.y_prob = np.array(y_prob)
        self.model_name = model_name

    def calibration_plot(self, save_path=None):
        """Plot reliability diagram."""
        prob_true, prob_pred = calibration_curve(self.y_true, self.y_prob, n_bins=10)
        brier = brier_score_loss(self.y_true, self.y_prob)
        
        plt.figure(figsize=(7, 5))
        plt.plot(prob_pred, prob_true, 'o-', label=f'{self.model_name} (Brier={brier:.3f})')
        plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect')
        plt.xlabel('Predicted probability')
        plt.ylabel('Observed frequency')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def net_benefit(self, threshold):
        """Calculate net benefit at a given threshold."""
        n = len(self.y_true)
        pred = (self.y_prob >= threshold).astype(int)
        
        tp = np.sum((pred == 1) & (self.y_true == 1))
        fp = np.sum((pred == 1) & (self.y_true == 0))
        
        w = threshold / (1 - threshold)
        return (tp / n) - (w * fp / n)

    def decision_curve(self, save_path=None):
        """Plot decision curve analysis."""
        thresholds = np.linspace(0.01, 0.99, 50)
        nb_model = [self.net_benefit(t) for t in thresholds]
        
        # Treat all baseline
        pos_rate = self.y_true.mean()
        neg_rate = 1 - pos_rate
        nb_all = [pos_rate - (t / (1-t)) * neg_rate for t in thresholds]
        
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, nb_model, label=self.model_name)
        plt.plot(thresholds, nb_all, '--', label='Treat All')
        plt.axhline(0, color='gray', linestyle=':', label='Treat None')
        plt.xlabel('Threshold')
        plt.ylabel('Net Benefit')
        plt.title('Decision Curve Analysis')
        plt.legend()
        plt.xlim(0, 1)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def get_metrics(self):
        """Return dict of standard metrics."""
        y_pred = (self.y_prob >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        
        report = classification_report(self.y_true, y_pred, output_dict=True)
        
        return {
            'auc': auc(fpr, tpr),
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1-score'],
            'brier': brier_score_loss(self.y_true, self.y_prob)
        }
