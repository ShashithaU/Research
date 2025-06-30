import flwr as fl
import torch
from model import LoanClassifier
from utils import load_dataset
from he_client import HEClient
import sys

# Import the original train/test functions from one of the client files
# (Assuming they are identical across clients)
from client_1 import train, test

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a client ID (1, 2, or 3)")
        sys.exit(1)
        
    client_id = sys.argv[1]
    print(f"Starting HE client {client_id}")

    # Load dataset for the specified client
    X_train, y_train = load_dataset(f"Data/Train/client_{client_id}.csv")
    X_test, y_test = load_dataset(f"Data/Test/client_{client_id}_test.csv")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    input_size = X_train.shape[1]
    model = LoanClassifier(input_size=input_size).to(DEVICE)

    # Start client
    fl.client.start_numpy_client(
        server_address="localhost:8080", 
        client=HEClient(model, X_train, y_train, X_test, y_test, train, test)
    )