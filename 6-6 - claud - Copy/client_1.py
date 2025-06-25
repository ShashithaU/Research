import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from model import LoanClassifier
from utils import load_dataset
import pickle
import os

# Load dataset
X_train, y_train = load_dataset("Data/Train/client_1.csv")
X_test, y_test = load_dataset("Data/Test/client_1_test.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Initialize model
input_size = X_train.shape[1]  # Dynamic input size
model = LoanClassifier(input_size=input_size).to(DEVICE)

def train(model, X_train, y_train, epochs, lr):
    # Use BCEWithLogitsLoss (applies sigmoid internally)
    criterion = nn.BCEWithLogitsLoss()  # Changed from BCELoss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_train), 32):  # Batch processing
            batch_X = X_train[i:i+32]
            batch_y = y_train[i:i+32]
            
            optimizer.zero_grad()
            
            # Get raw logits (no sigmoid applied)
            outputs = model.forward_logits(batch_X)  # Use forward_logits
            
            # Calculate loss (BCEWithLogitsLoss applies sigmoid internally)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()
    
    return loss.item()

def test(model, X, y):
    """Evaluate the model and return accuracy and loss"""
    model.eval()
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y).item()
        preds = (outputs > 0.5).float()
        acc = (preds == y).float().mean().item()
    
    return acc, loss

class LoanClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        """Extract model parameters as numpy arrays"""
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):
        """Load parameters into model"""
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=DEVICE) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model on local data"""
        print(f"Training with {len(parameters)} parameters")
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Get training configuration
        epochs = config.get("epochs", 5)
        learning_rate = config.get("learning_rate", 0.001)
        
        # Train model
        avg_loss = train(model, X_train.to(DEVICE), y_train.to(DEVICE), 
                        epochs=epochs, lr=learning_rate)
        
        print(f"Training completed. Average loss: {avg_loss:.4f}")
        
        # Return updated parameters, number of examples, and metrics
        return (self.get_parameters(config={}), 
                len(X_train), 
                {"train_loss": avg_loss})

    def evaluate(self, parameters, config):
        """Evaluate the model on local test data"""
        print("Evaluating model...")
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate model
        accuracy, loss = test(model, X_test.to(DEVICE), y_test.to(DEVICE))
        
        print(f"Evaluation completed. Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        
        # Return loss (for server aggregation), number of examples, and metrics
        return (float(loss), 
                len(X_test), 
                {"accuracy": float(accuracy), "test_loss": float(loss)})

if __name__ == "__main__":
    # Start client
    print(f"Starting client with training data: {X_train.shape}, test data: {X_test.shape}")
    fl.client.start_numpy_client(
        server_address="localhost:8080", 
        client=LoanClient()
    )