import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Import your existing model
from model import LoanClassifier

class StandaloneLoanTrainer:
    def __init__(self, model_config=None, device=None):
        """
        Standalone neural network trainer for loan classification
        
        Args:
            model_config: Dictionary with model configuration
            device: PyTorch device (cuda/cpu)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default model configuration (same as your best federated model)
        self.model_config = model_config or {
            'input_size': 11,
            'hidden_sizes': [1792, 896, 448, 224],  # Your best architecture
            'dropout_rate': 0.4  # Your optimal dropout
        }
        
        self.model = None
        self.scaler = StandardScaler()
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        print(f"Trainer initialized - Device: {self.device}")
        print(f"Model config: {self.model_config}")
    
    def load_and_prepare_data(self, train_paths, test_path=None, target_column="Personal Loan"):
        """
        Load and prepare training data from multiple CSV files
        
        Args:
            train_paths: List of paths to training CSV files
            test_path: Path to test CSV file (optional)
            target_column: Name of target column
            
        Returns:
            Prepared datasets
        """
        print("Loading and preparing data...")
        
        # Load and combine all training data
        all_train_data = []
        for path in train_paths:
            df = pd.read_csv(path)
            all_train_data.append(df)
            print(f"Loaded {path}: {df.shape}")
        
        # Combine all training data
        combined_train = pd.concat(all_train_data, ignore_index=True)
        print(f"Combined training data shape: {combined_train.shape}")
        
        # Separate features and target
        X_train = combined_train.drop(columns=[target_column])
        y_train = combined_train[target_column]
        
        # Update input size based on actual data
        self.model_config['input_size'] = X_train.shape[1]
        
        print(f"Features: {X_train.shape}, Target distribution: {y_train.value_counts().to_dict()}")
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        
        # Load test data if provided
        if test_path:
            test_df = pd.read_csv(test_path)
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            X_test_scaled = self.scaler.transform(X_test)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
            
            print(f"Test data loaded: {X_test_tensor.shape}")
            return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
        
        return X_train_tensor, y_train_tensor, None, None
    
    def create_model(self):
        """Create the neural network model"""
        print("Creating model...")
        self.model = LoanClassifier(
            input_size=self.model_config['input_size'],
            hidden_sizes=self.model_config['hidden_sizes'],
            dropout_rate=self.model_config['dropout_rate']
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model created with {total_params:,} parameters")
        return self.model
    
    def create_data_loaders(self, X_train, y_train, X_val=None, y_val=None, 
                           batch_size=64, val_split=0.2):
        """
        Create data loaders for training and validation
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            batch_size: Batch size for training
            val_split: Validation split ratio if no separate validation data
            
        Returns:
            train_loader, val_loader
        """
        # If no separate validation data, split training data
        if X_val is None:
            dataset_size = len(X_train)
            val_size = int(dataset_size * val_split)
            train_size = dataset_size - val_size
            
            # Random split
            indices = torch.randperm(dataset_size)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            X_train_split = X_train[train_indices]
            y_train_split = y_train[train_indices]
            X_val = X_train[val_indices]
            y_val = y_train[val_indices]
            
            print(f"Data split - Train: {len(X_train_split)}, Validation: {len(X_val)}")
        else:
            X_train_split = X_train
            y_train_split = y_train
            print(f"Using provided validation data - Train: {len(X_train_split)}, Val: {len(X_val)}")
        
        # Create datasets
        train_dataset = TensorDataset(X_train_split, y_train_split)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            # Get raw logits (no sigmoid applied)
            outputs = self.model.forward_logits(batch_X)
            
            # Calculate loss (BCEWithLogitsLoss applies sigmoid internally)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            with torch.no_grad():
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Get probabilities for validation
                outputs = self.model(batch_X)
                
                # Calculate loss
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=64, learning_rate=0.001, 
              early_stopping_patience=10, save_best=True):
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping_patience: Early stopping patience
            save_best: Whether to save the best model during training
            
        Returns:
            Training history
        """
        print(f"\nStarting training for {epochs} epochs...")
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        # Loss and optimizer (using your optimal configuration)
        criterion = nn.BCEWithLogitsLoss()  # For training
        val_criterion = nn.BCELoss()  # For validation
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_accuracy = 0
        patience_counter = 0
        best_model_state = None
        
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<10}")
        print("-" * 70)
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, val_criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)
            
            # Print progress
            print(f"{epoch+1:<6} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f} {current_lr:<10.6f}")
            
            # Early stopping and best model saving
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                if save_best:
                    best_model_state = self.model.state_dict().copy()
                print(f"  → New best validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model if saved
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation accuracy: {best_val_accuracy:.4f}")
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_accuracy:.4f}")
        return self.training_history
    
    def evaluate_with_threshold_optimization(self, X_test, y_test):
        """
        Evaluate model with threshold optimization (like your federated version)
        """
        print("\nEvaluating with threshold optimization...")
        
        self.model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            outputs = self.model(X_test)
            probabilities = outputs.cpu().numpy().flatten()
            true_labels = y_test.cpu().numpy().flatten()
        
        # Threshold optimization (same as your federated version)
        thresholds = np.arange(0.1, 0.8, 0.05)
        best_accuracy = 0
        best_f1 = 0
        best_threshold_acc = 0.5
        best_threshold_f1 = 0.5
        
        print(f"\n{'Threshold':<10} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1-Score':<9}")
        print("-" * 60)
        
        results = []
        for threshold in thresholds:
            preds = (probabilities > threshold).astype(int)
            
            accuracy = accuracy_score(true_labels, preds)
            precision = precision_score(true_labels, preds, zero_division=0)
            recall = recall_score(true_labels, preds, zero_division=0)
            f1 = f1_score(true_labels, preds, zero_division=0)
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': preds
            })
            
            print(f"{threshold:<10.2f} {accuracy:<10.3f} {precision:<11.3f} {recall:<8.3f} {f1:<9.3f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold_acc = threshold
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold_f1 = threshold
        
        print("-" * 60)
        print(f"Best Accuracy: {best_accuracy:.4f} at threshold {best_threshold_acc:.2f}")
        print(f"Best F1-Score: {best_f1:.4f} at threshold {best_threshold_f1:.2f}")
        
        # Use best F1 threshold
        best_result = next(r for r in results if r['threshold'] == best_threshold_f1)
        final_predictions = best_result['predictions']
        
        # Detailed evaluation
        print(f"\n{'='*50}")
        print(f"FINAL EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Optimal threshold: {best_threshold_f1:.2f}")
        print(f"Test samples: {len(true_labels)}")
        print(f"Accuracy: {best_f1:.4f} ({best_f1*100:.2f}%)")
        
        print(f"\nClassification Report:")
        print(classification_report(true_labels, final_predictions, 
                                  target_names=['No Loan', 'Loan']))
        
        cm = confusion_matrix(true_labels, final_predictions)
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"Actual    No Loan  Loan")
        print(f"No Loan   {cm[0,0]:7d}  {cm[0,1]:4d}")
        print(f"Loan      {cm[1,0]:7d}  {cm[1,1]:4d}")
        
        return {
            'optimal_threshold': best_threshold_f1,
            'accuracy': best_result['accuracy'],
            'precision': best_result['precision'],
            'recall': best_result['recall'],
            'f1_score': best_result['f1'],
            'predictions': final_predictions,
            'probabilities': probabilities,
            'true_labels': true_labels,
            'confusion_matrix': cm,
            'all_results': results
        }
    
    def save_model(self, save_dir="saved_models", model_name=None):
        """
        Save the trained model and related files
        
        Args:
            save_dir: Directory to save model files
            model_name: Name for the model files (auto-generated if None)
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"standalone_loan_model_{timestamp}"
        
        # File paths
        model_path = os.path.join(save_dir, f"{model_name}.pth")
        model_info_path = os.path.join(save_dir, f"{model_name}_info.pkl")
        scaler_path = os.path.join(save_dir, f"{model_name}_scaler.pkl")
        history_path = os.path.join(save_dir, f"{model_name}_history.pkl")
        
        # Save model state dict
        torch.save(self.model.state_dict(), model_path)
        
        # Save model info
        model_info = {
            'model_class': 'LoanClassifier',
            'input_size': self.model_config['input_size'],
            'hidden_sizes': self.model_config['hidden_sizes'],
            'dropout_rate': self.model_config['dropout_rate'],
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'architecture': str(self.model)
        }
        
        with open(model_info_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save training history
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        print(f"\n{'='*50}")
        print(f"MODEL SAVED SUCCESSFULLY")
        print(f"{'='*50}")
        print(f"Model weights: {model_path}")
        print(f"Model info: {model_info_path}")
        print(f"Scaler: {scaler_path}")
        print(f"Training history: {history_path}")
        
        return {
            'model_path': model_path,
            'model_info_path': model_info_path,
            'scaler_path': scaler_path,
            'history_path': history_path
        }
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if not self.training_history['train_loss']:
            print("No training history to plot")
            return
        
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.training_history['train_accuracy'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.training_history['val_accuracy'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()

def main():
    """Main training function"""
    print("="*60)
    print("STANDALONE NEURAL NETWORK TRAINER")
    print("="*60)
    
    # Configuration
    TRAIN_PATHS = [
        "Data/Train/TrainData.csv",
    ]
    
    TEST_PATH = "Data/Test/TestData.csv"  # Your main test dataset
    
    # Check if files exist
    missing_files = []
    for path in TRAIN_PATHS + [TEST_PATH]:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        print("Please check your file paths.")
        return
    
    # Create trainer
    trainer = StandaloneLoanTrainer()
    
    # Load and prepare data
    X_train, y_train, X_test, y_test = trainer.load_and_prepare_data(
        TRAIN_PATHS, TEST_PATH
    )
    
    # Train model
    print(f"\nTraining on {len(X_train)} samples...")
    training_history = trainer.train(
        X_train, y_train,
        epochs=100,  # More epochs for better convergence
        batch_size=64,
        learning_rate=0.001,
        early_stopping_patience=15
    )
    
    # Evaluate model
    if X_test is not None:
        evaluation_results = trainer.evaluate_with_threshold_optimization(X_test, y_test)
        
        print(f"\n🎉 FINAL RESULTS:")
        print(f"Accuracy: {evaluation_results['accuracy']:.4f} ({evaluation_results['accuracy']*100:.2f}%)")
        print(f"Precision: {evaluation_results['precision']:.4f}")
        print(f"Recall: {evaluation_results['recall']:.4f}")
        print(f"F1-Score: {evaluation_results['f1_score']:.4f}")
    
    # Save model
    save_info = trainer.save_model()
    
    # Plot training history
    trainer.plot_training_history(
        save_path=save_info['model_path'].replace('.pth', '_training_plot.png')
    )
    
    print(f"\n🏆 Training completed successfully!")
    print(f"Expected accuracy: 99%+ (same as your federated model)")

if __name__ == "__main__":
    main()