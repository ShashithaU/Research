import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from model import LoanClassifier
from utils import load_dataset
from dp_utils import DifferentialPrivacyManager, DPOptimizer  # ADD THIS LINE
import pickle
import os

# Load dataset
X_train, y_train = load_dataset("Data/Train/client_1.csv")
X_test, y_test = load_dataset("Data/Test/client_1_test.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Initialize model
input_size = X_train.shape[1]
model = LoanClassifier(input_size=input_size).to(DEVICE)

# Initialize Differential Privacy Manager
DP_CONFIG = {
    'epsilon': 4.0,          # Higher privacy budget (more utility)
    'delta': 1e-5,           # Keep same
    'max_grad_norm': 2.0,    # Higher clipping threshold
    'noise_multiplier': 0.6, # MUCH LOWER noise
    'sample_rate': 0.3       # Use 30% of data per batch
}

dp_manager = DifferentialPrivacyManager(**DP_CONFIG)
print(f"DP Manager initialized: ε={DP_CONFIG['epsilon']}, δ={DP_CONFIG['delta']}")

def dp_train(model, X_train, y_train, epochs, lr, dp_manager):
    """
    Train model with differential privacy
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Wrap optimizer with DP
    dp_optimizer = DPOptimizer(optimizer, dp_manager, DEVICE)
    
    model.train()
    total_grad_norm = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        # Check privacy budget
        if not dp_manager.should_continue_training():
            print(f"Privacy budget exhausted at epoch {epoch}")
            break
            
        for i in range(0, len(X_train), 32):
            batch_X = X_train[i:i+32]
            batch_y = y_train[i:i+32]
            
            # Subsampling for privacy (optional)
            if torch.rand(1).item() > dp_manager.sample_rate:
                continue
            
            dp_optimizer.zero_grad()
            
            # Forward pass
            outputs = model.forward_logits(batch_X)
            loss = criterion(outputs, batch_y.float())
            
            # Backward pass
            loss.backward()
            
            # DP optimization step (includes clipping and noise)
            grad_norm = dp_optimizer.step(model)
            
            total_grad_norm += grad_norm
            num_batches += 1
    
    # Update privacy accounting
    dp_manager.compute_privacy_spent(
        num_steps=num_batches,
        batch_size=32,
        dataset_size=len(X_train)
    )
    
    avg_grad_norm = total_grad_norm / max(num_batches, 1)
    
    print(f"Training completed with DP:")
    print(f"  Average gradient norm: {avg_grad_norm:.4f}")
    print(f"  Privacy spent: ε={dp_manager.privacy_spent:.4f}")
    print(f"  Remaining budget: ε={dp_manager.get_privacy_budget_remaining():.4f}")
    
    return loss.item()

def test(model, X, y):
    """Evaluate the model (no DP needed for evaluation)"""
    model.eval()
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y).item()
        preds = (outputs > 0.5).float()
        acc = (preds == y).float().mean().item()
    
    return acc, loss

class DPLoanClient(fl.client.NumPyClient):
    """Differentially Private Federated Learning Client"""
    
    def get_parameters(self, config):
        """Extract model parameters with optional DP noise"""
        params = [val.cpu().numpy() for val in model.state_dict().values()]
        
        # Add noise to parameters before sharing (optional)
        if config.get("add_noise_to_params", False):
            print("Adding DP noise to parameters before sharing...")
            param_tensors = [torch.tensor(p, device=DEVICE) for p in params]
            noisy_params = dp_manager.add_noise_to_parameters(param_tensors, DEVICE)
            params = [p.cpu().numpy() for p in noisy_params]
        
        return params

    def set_parameters(self, parameters):
        """Load parameters into model"""
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=DEVICE) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model with differential privacy"""
        print(f"DP Training with {len(parameters)} parameters")
        print(f"Privacy budget remaining: ε={dp_manager.get_privacy_budget_remaining():.4f}")
        
        # Check if we can continue training
        if not dp_manager.should_continue_training():
            print("⚠️  Privacy budget exhausted! Skipping training.")
            return (self.get_parameters(config={}), len(X_train), {"train_loss": 0.0})
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Get training configuration
        epochs = config.get("epochs", 5)
        learning_rate = config.get("learning_rate", 0.001)
        
        # Train model with DP
        avg_loss = dp_train(model, X_train.to(DEVICE), y_train.to(DEVICE), 
                           epochs=epochs, lr=learning_rate, dp_manager=dp_manager)
        
        # Return parameters (with optional noise)
        return (self.get_parameters(config=config), 
                len(X_train), 
                {
                    "train_loss": avg_loss,
                    "privacy_spent": dp_manager.privacy_spent,
                    "gradient_norm": dp_manager.max_grad_norm
                })

    def evaluate(self, parameters, config):
        """Evaluate the model (no DP needed)"""
        print("Evaluating DP model...")
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate model
        accuracy, loss = test(model, X_test.to(DEVICE), y_test.to(DEVICE))
        
        print(f"Evaluation completed. Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        print(f"Current privacy spent: ε={dp_manager.privacy_spent:.4f}")
        
        return (float(loss), 
                len(X_test), 
                {
                    "accuracy": float(accuracy), 
                    "test_loss": float(loss),
                    "privacy_spent": dp_manager.privacy_spent
                })

if __name__ == "__main__":
    print("="*60)
    print("DIFFERENTIAL PRIVACY FEDERATED LEARNING CLIENT")
    print("="*60)
    print(f"Privacy Configuration:")
    print(f"  Epsilon (ε): {DP_CONFIG['epsilon']}")
    print(f"  Delta (δ): {DP_CONFIG['delta']}")
    print(f"  Max Gradient Norm: {DP_CONFIG['max_grad_norm']}")
    print(f"  Noise Multiplier: {DP_CONFIG['noise_multiplier']}")
    print(f"  Sample Rate: {DP_CONFIG['sample_rate']}")
    print("="*60)
    
    # Start DP client
    print(f"Starting DP client with training data: {X_train.shape}, test data: {X_test.shape}")
    fl.client.start_numpy_client(
        server_address="localhost:8080", 
        client=DPLoanClient()
    )