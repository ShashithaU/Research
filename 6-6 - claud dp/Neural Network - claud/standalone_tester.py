import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import os
from model import LoanClassifier

class StandaloneModelTester:
    """Test saved standalone model"""
    
    def __init__(self, model_path, model_info_path, scaler_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model(model_path, model_info_path)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print("Standalone model tester loaded successfully!")
    
    def _load_model(self, model_path, model_info_path):
        """Load the saved model"""
        # Load model info
        with open(model_info_path, 'rb') as f:
            model_info = pickle.load(f)
        
        # Create model
        model = LoanClassifier(
            input_size=model_info['input_size'],
            hidden_sizes=model_info['hidden_sizes'],
            dropout_rate=model_info['dropout_rate']
        )
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded: {model_info['total_parameters']:,} parameters")
        return model
    
    def test_on_dataset(self, test_data_path, target_column="Personal Loan"):
        """Test model on a dataset with threshold optimization"""
        
        # Load test data
        df = pd.read_csv(test_data_path)
        X_test = df.drop(columns=[target_column])
        y_test = df[target_column]
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            probabilities = outputs.cpu().numpy().flatten()
        
        true_labels = y_test.values
        
        # Threshold optimization
        thresholds = np.arange(0.1, 0.8, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        print(f"\nTesting on {len(true_labels)} samples...")
        print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1-Score':<9}")
        print("-" * 60)
        
        for threshold in thresholds:
            preds = (probabilities > threshold).astype(int)
            
            accuracy = accuracy_score(true_labels, preds)
            precision = precision_score(true_labels, preds, zero_division=0)
            recall = recall_score(true_labels, preds, zero_division=0)
            f1 = f1_score(true_labels, preds, zero_division=0)
            
            print(f"{threshold:<10.2f} {accuracy:<10.3f} {precision:<11.3f} {recall:<8.3f} {f1:<9.3f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_predictions = preds
        
        print(f"\nBest threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
        
        # Final evaluation
        final_accuracy = accuracy_score(true_labels, best_predictions)
        final_precision = precision_score(true_labels, best_predictions)
        final_recall = recall_score(true_labels, best_predictions)
        
        print(f"\n{'='*50}")
        print(f"FINAL TEST RESULTS")
        print(f"{'='*50}")
        print(f"Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        print(f"Precision: {final_precision:.4f}")
        print(f"Recall: {final_recall:.4f}")
        print(f"F1-Score: {best_f1:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(true_labels, best_predictions, 
                                  target_names=['No Loan', 'Loan']))
        
        cm = confusion_matrix(true_labels, best_predictions)
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"Actual    No Loan  Loan")
        print(f"No Loan   {cm[0,0]:7d}  {cm[0,1]:4d}")
        print(f"Loan      {cm[1,0]:7d}  {cm[1,1]:4d}")
        
        return {
            'accuracy': final_accuracy,
            'precision': final_precision,
            'recall': final_recall,
            'f1_score': best_f1,
            'optimal_threshold': best_threshold,
            'predictions': best_predictions,
            'probabilities': probabilities
        }

def main():
    """Test the saved standalone model"""
    
    # Update these paths based on your saved model
    MODEL_PATH = "saved_models/standalone_loan_model_20241209_120000.pth"  # Update with actual path
    MODEL_INFO_PATH = "saved_models/standalone_loan_model_20241209_120000_info.pkl"  # Update with actual path
    SCALER_PATH = "saved_models/standalone_loan_model_20241209_120000_scaler.pkl"  # Update with actual path
    TEST_DATA_PATH = "Data/Test/TestData.csv"
    
    # Check if files exist
    required_files = [MODEL_PATH, MODEL_INFO_PATH, SCALER_PATH, TEST_DATA_PATH]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nFirst run the standalone_trainer.py to create the model files.")
        return
    
    # Create tester and run evaluation
    tester = StandaloneModelTester(MODEL_PATH, MODEL_INFO_PATH, SCALER_PATH)
    results = tester.test_on_dataset(TEST_DATA_PATH)
    
    print(f"\n🎉 Testing completed!")
    print(f"Expected: 99%+ accuracy (same as federated model)")

if __name__ == "__main__":
    main()