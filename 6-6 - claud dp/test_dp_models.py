"""
Script to test the Differential Privacy models
"""

import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from model import LoanClassifier
from utils import load_dataset

def load_model_from_files(model_path, info_path, scaler_path=None):
    """Load model from saved files"""
    try:
        # Load model info
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)
        
        # Create model
        input_size = model_info.get('input_size', 11)
        model = LoanClassifier(input_size=input_size)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, model_info
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def evaluate_model_on_dataset(model, dataset_path, scaler_path=None):
    """Evaluate model on a dataset"""
    try:
        # Load dataset
        X_test, y_test = load_dataset(dataset_path, scaler_path)
        
        print(f"Data types - X: {type(X_test)}, y: {type(y_test)}")
        
        # Convert to proper format
        if hasattr(X_test, 'values'):
            X_test_array = X_test.values
        else:
            X_test_array = np.array(X_test)
            
        if hasattr(y_test, 'values'):
            y_test_array = y_test.values
        else:
            y_test_array = np.array(y_test)
        
        # Ensure y is 1D
        if y_test_array.ndim > 1:
            y_test_array = y_test_array.flatten()
        
        print(f"Array shapes - X: {X_test_array.shape}, y: {y_test_array.shape}")
        
        # Convert to tensors
        X_test_tensor = torch.FloatTensor(X_test_array)
        y_test_tensor = torch.FloatTensor(y_test_array)
        
        print(f"Tensor shapes - X: {X_test_tensor.shape}, y: {y_test_tensor.shape}")
        
        # Evaluate
        with torch.no_grad():
            outputs = model(X_test_tensor)
            predictions = (outputs > 0.5).float().squeeze()
            
            # Ensure same dimensions
            if predictions.dim() > 1:
                predictions = predictions.squeeze()
            if y_test_tensor.dim() > 1:
                y_test_tensor = y_test_tensor.squeeze()
            
            print(f"Final shapes - predictions: {predictions.shape}, y_test: {y_test_tensor.shape}")
            
            # Calculate metrics
            accuracy = (predictions == y_test_tensor).float().mean().item()
            loss = nn.BCELoss()(outputs.squeeze(), y_test_tensor).item()
            
            # Additional metrics
            true_positives = ((predictions == 1) & (y_test_tensor == 1)).sum().item()
            false_positives = ((predictions == 1) & (y_test_tensor == 0)).sum().item()
            true_negatives = ((predictions == 0) & (y_test_tensor == 0)).sum().item()
            false_negatives = ((predictions == 0) & (y_test_tensor == 1)).sum().item()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'accuracy': accuracy,
                'loss': loss,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives,
                'total_samples': len(y_test_tensor)
            }
            
    except Exception as e:
        print(f"Error evaluating dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_and_test_model_internal(model_path, info_path, test_data_path, scaler_path):
    """Internal function to load and test model"""
    try:
        # Load model
        model, model_info = load_model_from_files(model_path, info_path, scaler_path)
        if model is None:
            return None
        
        # Evaluate on test data
        metrics = evaluate_model_on_dataset(model, test_data_path, scaler_path)
        return metrics
        
    except Exception as e:
        print(f"Error in load_and_test_model_internal: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dp_models():
    """Test all DP models that were created"""
    
    print("="*70)
    print("TESTING DIFFERENTIAL PRIVACY MODELS")
    print("="*70)
    
    # Look for DP models
    saved_models_dir = Path("saved_models")
    if not saved_models_dir.exists():
        print("❌ No saved_models directory found!")
        return
    
    # Find DP models
    dp_models = list(saved_models_dir.glob("*dp*.pth"))
    dp_info_files = list(saved_models_dir.glob("*dp*_info.pkl"))
    
    if not dp_models:
        print("❌ No DP models found!")
        print("Make sure you ran the DP federated learning successfully.")
        return
    
    print(f"✅ Found {len(dp_models)} DP models:")
    for model in dp_models:
        print(f"  📁 {model.name}")
    
    # Test dataset configuration
    TEST_DATASETS = [
        ("Data/Test/TestData.csv", "Main Test Dataset"),
        ("Data/Test/client_1_test.csv", "Client 1 Test"),
        ("Data/Test/client_2_test.csv", "Client 2 Test"),
        ("Data/Test/client_3_test.csv", "Client 3 Test"),
    ]
    
    # Filter existing datasets
    existing_datasets = []
    for path, name in TEST_DATASETS:
        if os.path.exists(path):
            existing_datasets.append((path, name))
        else:
            print(f"⚠️  Skipping {name} - file not found: {path}")
    
    if not existing_datasets:
        print("❌ No test datasets found!")
        return
    
    print(f"\n✅ Found {len(existing_datasets)} test datasets")
    
    # Test each DP model
    scaler_path = "scalers/global_scaler.pkl"
    
    for model_path in dp_models:
        # Find corresponding info file
        model_name = model_path.stem
        info_path = saved_models_dir / f"{model_name}_info.pkl"
        
        if not info_path.exists():
            print(f"⚠️  Info file not found for {model_name}, skipping...")
            continue
        
        print(f"\n{'='*50}")
        print(f"TESTING MODEL: {model_name}")
        print(f"{'='*50}")
        
        # Load model
        model, model_info = load_model_from_files(str(model_path), str(info_path), scaler_path)
        
        if model is None:
            print(f"❌ Failed to load model {model_name}")
            continue
        
        # Print model info
        if model_info:
            print(f"Model Info:")
            print(f"  Input Size: {model_info.get('input_size', 'Unknown')}")
            print(f"  Total Parameters: {model_info.get('total_parameters', 'Unknown')}")
            print(f"  Differential Privacy: {model_info.get('differential_privacy', 'Unknown')}")
            if 'server_dp_epsilon' in model_info:
                print(f"  Server DP ε: {model_info['server_dp_epsilon']}")
                print(f"  Server DP δ: {model_info['server_dp_delta']}")
        
        # Test on each dataset
        for dataset_path, dataset_name in existing_datasets:
            print(f"\n--- Testing on {dataset_name} ---")
            
            try:
                # Use internal function to test
                result = load_and_test_model_internal(
                    str(model_path), 
                    str(info_path), 
                    dataset_path, 
                    scaler_path
                )
                
                if result:
                    print(f"✅ {dataset_name}: Accuracy = {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
                    print(f"   Precision: {result['precision']:.4f}")
                    print(f"   Recall: {result['recall']:.4f}")
                    print(f"   F1 Score: {result['f1_score']:.4f}")
                    print(f"   Loss: {result['loss']:.4f}")
                else:
                    print(f"❌ Failed to test {dataset_name}")
                    
            except Exception as e:
                print(f"❌ Error testing {dataset_name}: {e}")
    
    # Test final DP model specifically
    final_model = saved_models_dir / "dp_global_model_dp_final.pth"
    final_info = saved_models_dir / "dp_global_model_dp_final_info.pkl"
    
    if final_model.exists() and final_info.exists():
        print(f"\n{'='*70}")
        print("DETAILED ANALYSIS OF FINAL DP MODEL")
        print(f"{'='*70}")
        
        try:
            model, model_info = load_model_from_files(str(final_model), str(final_info), scaler_path)
            
            if model and existing_datasets:
                # Test on main dataset
                main_test = next((path for path, name in existing_datasets if "TestData" in name), existing_datasets[0][0])
                
                print(f"Testing final model on: {main_test}")
                metrics = evaluate_model_on_dataset(model, main_test, scaler_path)
                
                if metrics:
                    print(f"\n📊 Final DP Model Performance:")
                    print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
                    print(f"   Precision: {metrics['precision']:.4f}")
                    print(f"   Recall: {metrics['recall']:.4f}")
                    print(f"   F1 Score: {metrics['f1_score']:.4f}")
                    print(f"   Loss: {metrics['loss']:.4f}")
                    print(f"\n📈 Confusion Matrix:")
                    print(f"   True Positives: {metrics['true_positives']}")
                    print(f"   False Positives: {metrics['false_positives']}")
                    print(f"   True Negatives: {metrics['true_negatives']}")
                    print(f"   False Negatives: {metrics['false_negatives']}")
                    print(f"   Total Samples: {metrics['total_samples']}")
                
        except Exception as e:
            print(f"❌ Error in detailed analysis: {e}")

def check_privacy_analysis():
    """Check privacy analysis results"""
    
    print(f"\n{'='*50}")
    print("PRIVACY ANALYSIS")
    print(f"{'='*50}")
    
    privacy_file = Path("saved_models/dp_global_model_privacy_analysis.pkl")
    
    if privacy_file.exists():
        try:
            with open(privacy_file, 'rb') as f:
                privacy_data = pickle.load(f)
            
            print(f"📊 Privacy Budget Summary:")
            print(f"   Total Training Rounds: {privacy_data['total_rounds']}")
            print(f"   Final Accuracy: {privacy_data['final_accuracy']:.4f}")
            print(f"   Server DP Config: ε={privacy_data['server_dp_config']['epsilon']}, δ={privacy_data['server_dp_config']['delta']}")
            
            print(f"\n📈 Privacy Spent Per Round:")
            for round_data in privacy_data['privacy_metrics_per_round']:
                print(f"   Round {round_data['round']}: "
                      f"Avg ε={round_data['avg_client_privacy_spent']:.4f}, "
                      f"Max ε={round_data['max_client_privacy_spent']:.4f}, "
                      f"Clients: {round_data['num_clients']}")
            
        except Exception as e:
            print(f"❌ Error reading privacy analysis: {e}")
    else:
        print("❌ Privacy analysis file not found")

def check_saved_files():
    """Check what files were actually saved"""
    
    print(f"{'='*50}")
    print("CHECKING SAVED FILES")
    print(f"{'='*50}")
    
    saved_models_dir = Path("saved_models")
    if saved_models_dir.exists():
        all_files = list(saved_models_dir.glob("*"))
        print(f"📁 Files in saved_models directory ({len(all_files)} files):")
        for file in all_files:
            size = file.stat().st_size / 1024  # KB
            print(f"   {file.name} ({size:.1f} KB)")
    else:
        print("❌ No saved_models directory found")
    
    scalers_dir = Path("scalers")
    if scalers_dir.exists():
        scaler_files = list(scalers_dir.glob("*"))
        print(f"\n📁 Files in scalers directory ({len(scaler_files)} files):")
        for file in scaler_files:
            size = file.stat().st_size / 1024  # KB
            print(f"   {file.name} ({size:.1f} KB)")
    else:
        print("\n❌ No scalers directory found")

def create_performance_summary():
    """Create a summary of all model performances"""
    
    print(f"\n{'='*50}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    
    saved_models_dir = Path("saved_models")
    if not saved_models_dir.exists():
        print("❌ No saved_models directory found!")
        return
    
    # Find all models (DP and non-DP)
    all_models = list(saved_models_dir.glob("*.pth"))
    
    if not all_models:
        print("❌ No models found!")
        return
    
    print(f"📊 Found {len(all_models)} models total")
    
    # Categorize models
    dp_models = [m for m in all_models if "dp" in m.name.lower()]
    regular_models = [m for m in all_models if "dp" not in m.name.lower()]
    
    print(f"   DP Models: {len(dp_models)}")
    print(f"   Regular Models: {len(regular_models)}")
    
    # Test data
    test_data = "Data/Test/TestData.csv"
    if not os.path.exists(test_data):
        test_data = "Data/Test/client_1_test.csv"
    
    if not os.path.exists(test_data):
        print("❌ No test data found for performance comparison")
        return
    
    print(f"   Using test data: {test_data}")
    
    scaler_path = "scalers/global_scaler.pkl"
    
    print(f"\n📈 Model Performance Comparison:")
    print(f"{'Model Name':<40} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 80)
    
    for model_path in all_models:
        model_name = model_path.stem
        info_path = saved_models_dir / f"{model_name}_info.pkl"
        
        if not info_path.exists():
            continue
        
        try:
            result = load_and_test_model_internal(
                str(model_path), 
                str(info_path), 
                test_data, 
                scaler_path
            )
            
            if result:
                print(f"{model_name:<40} {result['accuracy']:<10.4f} {result['precision']:<10.4f} {result['recall']:<10.4f} {result['f1_score']:<10.4f}")
            else:
                print(f"{model_name:<40} {'FAILED':<10} {'FAILED':<10} {'FAILED':<10} {'FAILED':<10}")
                
        except Exception as e:
            print(f"{model_name:<40} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")

if __name__ == "__main__":
    check_saved_files()
    test_dp_models()
    check_privacy_analysis()
    create_performance_summary()
    
    print(f"\n{'='*70}")
    print("TESTING COMPLETED!")
    print("Check the results above to see how your DP models performed.")
    print(f"{'='*70}")