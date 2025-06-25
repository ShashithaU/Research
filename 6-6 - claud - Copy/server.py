import flwr as fl
import torch
import os
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, Parameters
from model import LoanClassifier
import pickle
import numpy as np

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics using weighted average"""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    aggregated_accuracy = sum(accuracies) / sum(examples)
    
    print(f"Round accuracy: {aggregated_accuracy:.4f}")
    return {"accuracy": aggregated_accuracy}

def fit_config(server_round: int):
    """Return training configuration dict for each round"""
    config = {
        "server_round": server_round,
        "epochs": 5 if server_round < 3 else 3,  # Reduce epochs in later rounds
        "learning_rate": 0.001,
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round"""
    return {"server_round": server_round}

def save_global_model(parameters: Parameters, model_path: str, model_info_path: str, 
                     input_size: int = 11):
    """
    Save the global model parameters to disk
    """
    try:
        print(f"\n=== DEBUGGING MODEL SAVE ===")
        print(f"Attempting to save model to: {model_path}")
        print(f"Model info to: {model_info_path}")
        print(f"Input size: {input_size}")
        print(f"Parameters type: {type(parameters)}")
        print(f"Parameters tensors length: {len(parameters.tensors)}")
        
        # Create model instance
        print("Creating model instance...")
        model = LoanClassifier(input_size=input_size)
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Use Flower's built-in conversion function - this handles shapes automatically
        print("Converting parameters using Flower's built-in function...")
        from flwr.common import parameters_to_ndarrays
        
        # Convert Flower parameters to numpy arrays
        arrays = parameters_to_ndarrays(parameters)
        print(f"Converted {len(arrays)} parameter arrays")
        
        # Create state dict by zipping with model parameter names
        print("Creating state dict...")
        params_dict = zip(model.state_dict().keys(), arrays)
        state_dict = {k: torch.from_numpy(v).float() for k, v in params_dict}
        
        # Print shapes for debugging
        for name, tensor in state_dict.items():
            print(f"Parameter {name}: {tensor.shape}")
        
        print("Loading state dict into model...")
        model.load_state_dict(state_dict, strict=True)
        
        # Create directory if it doesn't exist
        print(f"Creating directory: {os.path.dirname(model_path)}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model state dict
        print(f"Saving model to: {model_path}")
        torch.save(model.state_dict(), model_path)
        
        # Save model architecture info
        print(f"Saving model info to: {model_info_path}")
        model_info = {
            'input_size': input_size,
            'model_class': 'LoanClassifier',
            'architecture': str(model),
            'total_parameters': sum(p.numel() for p in model.parameters())
        }
        
        with open(model_info_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"✅ Global model saved to: {model_path}")
        print(f"✅ Model info saved to: {model_info_path}")
        print("=== MODEL SAVE COMPLETE ===\n")
        return True
        
    except Exception as e:
        print(f"❌ Error saving global model: {e}")
        import traceback
        traceback.print_exc()  # Print full error for debugging
        print("=== MODEL SAVE FAILED ===\n")
        return False

class ModelSavingStrategy(fl.server.strategy.FedAvg):
    """Custom strategy that saves the global model after training"""
    
    def __init__(self, model_save_path: str, input_size: int = 11, **kwargs):
        super().__init__(**kwargs)
        self.model_save_path = model_save_path
        self.input_size = input_size
        self.round_accuracies = []
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results and save model if it's the last round"""
        
        # Call parent method to get aggregated metrics
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated_result is not None:
            loss, metrics = aggregated_result
            if "accuracy" in metrics:
                self.round_accuracies.append(metrics["accuracy"])
                print(f"Round {server_round} - Accuracy: {metrics['accuracy']:.4f}")
        
        # Save model after each round (optional - you can modify this)
        if hasattr(self, '_current_parameters') and self._current_parameters is not None:
            model_path = f"{self.model_save_path}_round_{server_round}.pth"
            info_path = f"{self.model_save_path}_round_{server_round}_info.pkl"
            save_global_model(self._current_parameters, model_path, info_path, self.input_size)
        
        return aggregated_result
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate fit results and store current parameters"""
        
        # Call parent method
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_result is not None:
            parameters, metrics = aggregated_result
            # Store current parameters for saving
            self._current_parameters = parameters
            
            # Save final model after last round
            if server_round == 5:  # Assuming 5 rounds total
                final_model_path = f"{self.model_save_path}_final.pth"
                final_info_path = f"{self.model_save_path}_final_info.pkl"
                save_global_model(parameters, final_model_path, final_info_path, self.input_size)
                
                # Save training history
                history_path = f"{self.model_save_path}_training_history.pkl"
                # Fix: Ensure directory exists before saving history
                os.makedirs(os.path.dirname(history_path), exist_ok=True)
                history = {
                    'round_accuracies': self.round_accuracies,
                    'total_rounds': server_round
                }
                with open(history_path, 'wb') as f:
                    pickle.dump(history, f)
                print(f"Training history saved to: {history_path}")
        
        return aggregated_result

if __name__ == "__main__":
    # Configuration
    INPUT_SIZE = 11  # Update this based on your actual feature count
    MODEL_SAVE_PATH = "saved_models/global_model"
    
    # Create custom strategy with model saving
    strategy = ModelSavingStrategy(
        model_save_path=MODEL_SAVE_PATH,
        input_size=INPUT_SIZE,
        
        # Fraction of clients to sample for training/evaluation
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        
        # Minimum number of clients for training/evaluation
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        
        # Functions to configure training/evaluation
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        
        # Function to aggregate evaluation metrics
        evaluate_metrics_aggregation_fn=weighted_average,
        
        # Initial parameters (optional)
        initial_parameters=None,
    )

    print("Starting Flower server with model saving...")
    
    # Start server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
    
    print("Server completed! Global model has been saved.")