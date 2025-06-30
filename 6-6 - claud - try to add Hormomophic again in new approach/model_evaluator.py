import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from model import LoanClassifier
from utils import load_dataset
import pickle
import os

class GlobalModelEvaluator:
    """Class to evaluate saved global model on separate datasets"""
    
    def __init__(self, model_path: str, model_info_path: str, scaler_path: str):
        """
        Initialize evaluator with saved model
        
        Args:
            model_path: Path to saved model state dict
            model_info_path: Path to model info pickle file
            scaler_path: Path to fitted scaler
        """
        self.model_path = model_path
        self.model_info_path = model_info_path
        self.scaler_path = scaler_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the saved global model"""
        try:
            # Load model info
            with open(self.model_info_path, 'rb') as f:
                model_info = pickle.load(f)
            
            print(f"Loading model with architecture: {model_info['model_class']}")
            print(f"Input size: {model_info['input_size']}")
            print(f"Total parameters: {model_info['total_parameters']}")
            
            # Create model instance
            model = LoanClassifier(input_size=model_info['input_size'])
            
            # Load model weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            print(f"Model loaded successfully from: {self.model_path}")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def evaluate_dataset(self, data_path: str, target_column: str = "Personal Loan"):
        """
        Evaluate model on a separate dataset
        
        Args:
            data_path: Path to evaluation dataset CSV
            target_column: Name of target column
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\nEvaluating model on dataset: {data_path}")
        
        # Load and preprocess data
        X, y = load_dataset(data_path, self.scaler_path, target_column=target_column)
        X, y = X.to(self.device), y.to(self.device)
        
        print(f"Evaluation data shape: X={X.shape}, y={y.shape}")
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = outputs.cpu().numpy().flatten()
            predictions = (outputs > 0.5).float().cpu().numpy().flatten()
            true_labels = y.cpu().numpy().flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        # Calculate loss
        criterion = nn.BCELoss()
        with torch.no_grad():
            loss = criterion(outputs, y).item()
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'loss': loss,
            'num_samples': len(true_labels),
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels
        }
        
        return metrics
    
    def print_evaluation_report(self, metrics: dict, dataset_name: str = "Dataset"):
        """Print detailed evaluation report"""
        print(f"\n{'='*50}")
        print(f"EVALUATION REPORT - {dataset_name}")
        print(f"{'='*50}")
        print(f"Number of samples: {metrics['num_samples']}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"Loss:      {metrics['loss']:.4f}")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(metrics['true_labels'], metrics['predictions'],
                                  target_names=['No Loan', 'Loan']))
        
        # Confusion Matrix
        cm = confusion_matrix(metrics['true_labels'], metrics['predictions'])
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"Actual    No Loan  Loan")
        print(f"No Loan   {cm[0,0]:7d}  {cm[0,1]:4d}")
        print(f"Loan      {cm[1,0]:7d}  {cm[1,1]:4d}")
    
    def plot_evaluation_results(self, metrics: dict, save_path: str = None):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Confusion Matrix
        cm = confusion_matrix(metrics['true_labels'], metrics['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # Prediction Distribution
        axes[0,1].hist(metrics['probabilities'], bins=30, alpha=0.7, color='skyblue')
        axes[0,1].axvline(0.5, color='red', linestyle='--', label='Threshold')
        axes[0,1].set_title('Prediction Probability Distribution')
        axes[0,1].set_xlabel('Probability')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        
        # Metrics Bar Plot
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['f1_score']]
        
        bars = axes[1,0].bar(metric_names, metric_values, color=['green', 'blue', 'orange', 'red'])
        axes[1,0].set_title('Performance Metrics')
        axes[1,0].set_ylabel('Score')
        axes[1,0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom')
        
        # ROC-like plot (Probability vs True Labels)
        true_pos_probs = metrics['probabilities'][metrics['true_labels'] == 1]
        true_neg_probs = metrics['probabilities'][metrics['true_labels'] == 0]
        
        axes[1,1].hist(true_neg_probs, bins=20, alpha=0.5, label='No Loan', color='blue')
        axes[1,1].hist(true_pos_probs, bins=20, alpha=0.5, label='Loan', color='red')
        axes[1,1].axvline(0.5, color='black', linestyle='--', label='Threshold')
        axes[1,1].set_title('Probability Distribution by True Label')
        axes[1,1].set_xlabel('Probability')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation plots saved to: {save_path}")
        
        plt.show()
    
    def compare_datasets(self, dataset_paths: list, dataset_names: list = None):
        """Compare model performance across multiple datasets"""
        if dataset_names is None:
            dataset_names = [f"Dataset_{i+1}" for i in range(len(dataset_paths))]
        
        results = {}
        
        for path, name in zip(dataset_paths, dataset_names):
            try:
                metrics = self.evaluate_dataset(path)
                results[name] = metrics
                self.print_evaluation_report(metrics, name)
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        # Create comparison plot
        if len(results) > 1:
            self._plot_comparison(results)
        
        return results
    
    def _plot_comparison(self, results: dict):
        """Plot comparison across datasets"""
        datasets = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(datasets))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results[dataset][metric] for dataset in datasets]
            ax.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Datasets')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison Across Datasets')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()

# Example usage and main evaluation script
def main():
    """Main evaluation function"""
    
    # Configuration
    MODEL_PATH = "saved_models/global_model_final.pth"
    MODEL_INFO_PATH = "saved_models/global_model_final_info.pkl"
    SCALER_PATH = "scalers/global_scaler.pkl"
    
    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        print("Please run the federated learning server first to generate the model.")
        return
    
    try:
        # Create evaluator
        evaluator = GlobalModelEvaluator(MODEL_PATH, MODEL_INFO_PATH, SCALER_PATH)
        
        # Example: Evaluate on a single dataset
        test_dataset_path = "Data/Test/separate_test_dataset.csv"  # Update this path
        
        if os.path.exists(test_dataset_path):
            metrics = evaluator.evaluate_dataset(test_dataset_path)
            evaluator.print_evaluation_report(metrics, "Separate Test Dataset")
            evaluator.plot_evaluation_results(metrics, "evaluation_results.png")
        else:
            print(f"Test dataset not found: {test_dataset_path}")
            print("Please provide the correct path to your separate test dataset.")
        
        # Example: Compare multiple datasets
        test_datasets = [
            "Data/Test/client_1_test.csv",
            "Data/Test/client_2_test.csv", 
            "Data/Test/separate_test_dataset.csv"
        ]
        
        existing_datasets = [path for path in test_datasets if os.path.exists(path)]
        
        if existing_datasets:
            dataset_names = [f"Client_{i+1}_Test" for i in range(len(existing_datasets)-1)]
            dataset_names.append("Separate_Test")
            
            comparison_results = evaluator.compare_datasets(existing_datasets, dataset_names)
            print("\nComparison completed!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()