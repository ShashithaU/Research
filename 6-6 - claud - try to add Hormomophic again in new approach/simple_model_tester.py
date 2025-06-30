"""
Simple script to test the saved global model on a separate dataset
This is a simplified version if you don't need all the advanced features
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model import LoanClassifier
from utils import load_dataset
import pickle

# Add this function at the top of simple_model_tester.py (after imports):

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#newly added -------------------------
def find_optimal_threshold(probabilities, true_labels):
    """Find the optimal threshold that maximizes accuracy or F1-score"""
    
    thresholds = np.arange(0.45, 0.70, 0.005)  # Test thresholds from 0.1 to 0.55
    best_accuracy = 0
    best_f1 = 0
    best_threshold_acc = 0.2
    best_threshold_f1 = 0.2
    
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION")
    print("="*80)
    print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1-Score':<9} {'TP':<4} {'FP':<4} {'TN':<4} {'FN':<4}")
    print("-" * 80)
    
    results = []
    
    for threshold in thresholds:
        # Make predictions with current threshold
        preds = (probabilities > threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, preds)
        precision = precision_score(true_labels, preds, zero_division=0)
        recall = recall_score(true_labels, preds, zero_division=0)
        f1 = f1_score(true_labels, preds, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, preds)
        tn, fp, fn, tp = cm.ravel()
        
        # Store results
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'predictions': preds
        })
        
        # Print results
        print(f"{threshold:<10.2f} {accuracy:<10.3f} {precision:<11.3f} {recall:<8.3f} {f1:<9.3f} {tp:<4} {fp:<4} {tn:<4} {fn:<4}")
        
        # Track best thresholds
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold_acc = threshold
            
        if f1 > best_f1:
            best_f1 = f1
            best_threshold_f1 = threshold
    
    print("-" * 80)
    print(f"Best Accuracy: {best_accuracy:.4f} at threshold {best_threshold_acc:.2f}")
    print(f"Best F1-Score: {best_f1:.4f} at threshold {best_threshold_f1:.2f}")
    print("="*80)
    
    # Return results for best F1-score (better for imbalanced data)
    best_result = next(r for r in results if r['threshold'] == best_threshold_f1)
    return best_threshold_f1, best_result['predictions'], results
#newly added -------------------------

def load_and_test_model(model_path, model_info_path, test_data_path, scaler_path):
    """
    Load saved model and test on separate dataset
    
    Args:
        model_path: Path to saved model (.pth file)
        model_info_path: Path to model info (.pkl file)
        test_data_path: Path to test dataset (.csv file)
        scaler_path: Path to fitted scaler (.pkl file)
    
    Returns:
        Dictionary with test results
    """
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load model info
        print("Loading model information...")
        with open(model_info_path, 'rb') as f:
            model_info = pickle.load(f)
        
        print(f"Model class: {model_info['model_class']}")
        print(f"Input size: {model_info['input_size']}")
        print(f"Total parameters: {model_info['total_parameters']}")
        
        # Create and load model
        print("Loading model weights...")
        model = LoanClassifier(input_size=model_info['input_size'])
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        
        # Load test data
        print(f"Loading test data from: {test_data_path}")
        X_test, y_test = load_dataset(test_data_path, scaler_path)
        X_test, y_test = X_test.to(device), y_test.to(device)
        
        print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
        
        # Make predictions
        # print("Making predictions...")
        # with torch.no_grad():
        #     outputs = model(X_test)
        #     probabilities = outputs.cpu().numpy().flatten()
            #predictions = (outputs > 0.2).float().cpu().numpy().flatten()
            
        print("Making predictions...")
        with torch.no_grad():
            outputs = model(X_test)
            probabilities = outputs.cpu().numpy().flatten()
            true_labels = y_test.cpu().numpy().flatten()

# Optimize threshold
            optimal_threshold, optimal_predictions, all_results = find_optimal_threshold(probabilities, true_labels)

# Use optimal predictions
            predictions = optimal_predictions

            print(f"\nUsing optimal threshold: {optimal_threshold:.2f}")
            #newly adddded ------------
            true_labels = y_test.cpu().numpy().flatten()
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels, predictions)
        
        # Calculate loss
        criterion = nn.BCELoss()
        with torch.no_grad():
            loss = criterion(outputs, y_test).item()
        
        # Print results
        print(f"\n{'='*50}")
        print(f"TEST RESULTS")
        print(f"{'='*50}")
        print(f"Test samples: {len(true_labels)}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Loss: {loss:.4f}")
        
        # Detailed classification report
        print(f"\nClassification Report:")
        print(classification_report(true_labels, predictions, 
                                  target_names=['No Loan', 'Loan']))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"Actual    No Loan  Loan")
        print(f"No Loan   {cm[0,0]:7d}  {cm[0,1]:4d}")
        print(f"Loan      {cm[1,0]:7d}  {cm[1,1]:4d}")


        #newly adddded -----------------------------------

        print(f"\n{'='*60}")
        print("DETAILED ANALYSIS")
        print(f"{'='*60}")

        # Business impact analysis
        total_loans = np.sum(true_labels)
        total_no_loans = len(true_labels) - total_loans
        loans_caught = np.sum((true_labels == 1) & (predictions == 1))
        loans_missed = np.sum((true_labels == 1) & (predictions == 0))
        false_alarms = np.sum((true_labels == 0) & (predictions == 1))

        print(f"Business Impact:")
        print(f"  Total loan applications: {total_loans}")
        print(f"  Loans correctly identified: {loans_caught} ({loans_caught/total_loans*100:.1f}%)")
        print(f"  Loans missed: {loans_missed} ({loans_missed/total_loans*100:.1f}%)")
        print(f"  False alarms: {false_alarms} ({false_alarms/total_no_loans*100:.1f}%)")

        # Risk assessment
        print(f"\nRisk Assessment:")
        if loans_missed/total_loans < 0.1:
            print("  ✅ Low risk: Missing less than 10% of loans")
        elif loans_missed/total_loans < 0.2:
            print("  ⚠️  Medium risk: Missing 10-20% of loans")
        else:
            print("  ❌ High risk: Missing more than 20% of loans")

        if false_alarms/total_no_loans < 0.1:
            print("  ✅ Low false alarm rate: Less than 10%")
        elif false_alarms/total_no_loans < 0.2:
            print("  ⚠️  Medium false alarm rate: 10-20%")
        else:
            print("  ❌ High false alarm rate: More than 20%")


        #newly adddded -----------------------------------
        
        # Additional statistics
        total_positive = np.sum(true_labels)
        total_negative = len(true_labels) - total_positive
        correct_positive = np.sum((predictions == 1) & (true_labels == 1))
        correct_negative = np.sum((predictions == 0) & (true_labels == 0))
        
        print(f"\nAdditional Statistics:")
        print(f"Total positive cases (Loan): {total_positive}")
        print(f"Total negative cases (No Loan): {total_negative}")
        print(f"Correctly predicted positive: {correct_positive}")
        print(f"Correctly predicted negative: {correct_negative}")
        
        # Probability statistics
        print(f"\nPrediction Probability Statistics:")
        print(f"Mean probability: {np.mean(probabilities):.4f}")
        print(f"Min probability: {np.min(probabilities):.4f}")
        print(f"Max probability: {np.max(probabilities):.4f}")
        print(f"Std probability: {np.std(probabilities):.4f}")
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels,
            'confusion_matrix': cm
        }
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return None

def test_multiple_datasets(model_path, model_info_path, scaler_path, test_datasets):
    """
    Test model on multiple datasets
    
    Args:
        model_path: Path to saved model
        model_info_path: Path to model info
        scaler_path: Path to scaler
        test_datasets: List of tuples (dataset_path, dataset_name)
    """
    
    results = {}
    
    print(f"Testing model on {len(test_datasets)} datasets...")
    
    for dataset_path, dataset_name in test_datasets:
        print(f"\n{'*'*60}")
        print(f"Testing on: {dataset_name}")
        print(f"{'*'*60}")
        
        # try:
        #     result = load_and_test_model(model_path, model_info_path, 
        #                                dataset_path, scaler_path)
        #     if result:
        #         results[dataset_name] = result

        try:
        # ... existing loading code ...
        
            print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
        
            # Make predictions with threshold optimization
            print("Making predictions with threshold optimization...")
            with torch.no_grad():
                outputs = model(X_test)
                probabilities = outputs.cpu().numpy().flatten()
                true_labels = y_test.cpu().numpy().flatten()
        
        # Find optimal threshold
            optimal_threshold, optimal_predictions, all_results = find_optimal_threshold(probabilities, true_labels)
        
        # Use optimal predictions
            predictions = optimal_predictions
        
            print(f"\nUsing optimal threshold: {optimal_threshold:.2f}")
        
        # Calculate final metrics with optimal threshold
            accuracy = accuracy_score(true_labels, predictions)
        
        # Calculate loss (using original threshold for consistency)
            criterion = nn.BCELoss()
            loss = criterion(torch.tensor(probabilities), torch.tensor(true_labels.astype(float))).item()
        
        # Display results
            print(f"\n{'='*60}")
            print(f"FINAL RESULTS (Optimized)")
            print(f"{'='*60}")
            print(f"Dataset: {test_data_path}")
            print(f"Test samples: {len(true_labels)}")
            print(f"Optimal threshold: {optimal_threshold:.2f}")
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Loss: {loss:.4f}")
        
        # ... rest of your existing code for classification report and confusion matrix ...
        
            return {
                'accuracy': accuracy,
                'loss': loss,
                'threshold': optimal_threshold,
                'all_thresholds': all_results
        }





            
        except Exception as e:
            print(f"Error testing {dataset_name}: {e}")
    
    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"SUMMARY COMPARISON")
        print(f"{'='*60}")
        print(f"{'Dataset':<20} {'Accuracy':<10} {'Loss':<10}")
        print(f"{'-'*40}")
        
        for name, result in results.items():
            print(f"{name:<20} {result['accuracy']:<10.4f} {result['loss']:<10.4f}")
    
    return results

# Main execution
if __name__ == "__main__":
    # Configuration - Update these paths according to your setup
    MODEL_PATH = "saved_models/global_model_final_he.pth"
    MODEL_INFO_PATH = "saved_models/global_model_final_he_info.pkl"
    SCALER_PATH = "scalers/global_scaler.pkl"
    
    # Test on single dataset
    print("=== SINGLE DATASET TEST ===")
    TEST_DATA_PATH = "Data/Test/TestData.csv"  # Update this path
    
    result = load_and_test_model(MODEL_PATH, MODEL_INFO_PATH, 
                               TEST_DATA_PATH, SCALER_PATH)
    
    if result:
        print("Single dataset test completed successfully!")
    
    # Test on multiple datasets (optional)
    print("\n=== MULTIPLE DATASET TEST ===")
    test_datasets = [
        ("Data/Test/client_1_test.csv", "Client_1_Test"),
        ("Data/Test/client_2_test.csv", "Client_2_Test"),
        ("Data/Test/separate_test_dataset.csv", "Separate_Test"),
        # Add more datasets as needed
    ]
    
    # Filter to only existing files
    existing_datasets = []
    for path, name in test_datasets:
        try:
            pd.read_csv(path)  # Check if file exists and is readable
            existing_datasets.append((path, name))
        except:
            print(f"Skipping {name} - file not found or not readable: {path}")
    
    if existing_datasets:
        results = test_multiple_datasets(MODEL_PATH, MODEL_INFO_PATH, 
                                       SCALER_PATH, existing_datasets)
        print(f"\nTesting completed on {len(results)} datasets!")
    else:
        print("No valid test datasets found. Please check your file paths.")