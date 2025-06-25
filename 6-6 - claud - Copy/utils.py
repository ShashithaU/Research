import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import pickle
import os

def save_scaler(scaler, path):
    """Save the fitted scaler for later use"""
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)

def load_scaler(path):
    """Load a previously fitted scaler"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_dataset(path, scaler_path=None, fit_scaler=False, target_column="Personal Loan"):
    """
    Load dataset with proper preprocessing
    
    Args:
        path: Path to CSV file
        scaler_path: Path to save/load scaler
        fit_scaler: Whether to fit a new scaler (only for first client or global preprocessing)
        target_column: Name of the target column
    """
    print(f"Loading dataset from: {path}")
    
    # Load data
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Handle scaling
    if scaler_path and os.path.exists(scaler_path) and not fit_scaler:
        # Load existing scaler
        print(f"Loading existing scaler from: {scaler_path}")
        scaler = load_scaler(scaler_path)
        X_scaled = scaler.transform(X)
    else:
        # Fit new scaler
        print("Fitting new scaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler if path provided
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            save_scaler(scaler, scaler_path)
            print(f"Scaler saved to: {scaler_path}")
    
    # Convert to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    
    print(f"Final tensor shapes - X: {X_tensor.shape}, y: {y_tensor.shape}")
    
    return X_tensor, y_tensor

def create_global_scaler(train_paths, scaler_path, target_column="Personal Loan"):
    """
    Create a global scaler from multiple training datasets
    This should be run before federated learning starts
    
    Args:
        train_paths: List of paths to training CSV files
        scaler_path: Path to save the global scaler
        target_column: Name of the target column
    """
    print("Creating global scaler from all training data...")
    
    all_data = []
    for path in train_paths:
        df = pd.read_csv(path)
        X = df.drop(columns=[target_column])
        all_data.append(X)
    
    # Combine all training data
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Combined training data shape: {combined_data.shape}")
    
    # Fit scaler on combined data
    scaler = StandardScaler()
    scaler.fit(combined_data)
    
    # Save scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    save_scaler(scaler, scaler_path)
    
    print(f"Global scaler saved to: {scaler_path}")
    return scaler

# Example usage for global preprocessing
if __name__ == "__main__":
    # Create global scaler (run this once before starting federated learning)
    train_paths = [
        "Data/Train/client_1.csv",
        "Data/Train/client_2.csv", 
        "Data/Train/client_3.csv"
    ]
    
    try:
        create_global_scaler(train_paths, "scalers/global_scaler.pkl")
        print("Global scaler created successfully!")
    except Exception as e:
        print(f"Error creating global scaler: {e}")
        print("Make sure all training data files exist.")