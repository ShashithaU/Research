import flwr as fl
import torch
import os
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import Metrics, Parameters, Scalar
from flwr.server.strategy import FedAvg
from model import LoanClassifier
from dp_utils import DifferentialPrivacyManager, create_dp_noise
import pickle

class DPFedAvgStrategy(FedAvg):
    """
    Federated Averaging with Differential Privacy
    """
    
    def __init__(self, 
                 model_save_path: str,
                 input_size: int = 11,
                 server_dp_epsilon: float = 1.0,
                 server_dp_delta: float = 1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.model_save_path = model_save_path
        self.input_size = input_size
        self.round_accuracies = []
        self.privacy_metrics = []
        
        # Server-side DP for aggregation
        self.server_dp_manager = DifferentialPrivacyManager(
            epsilon=server_dp_epsilon,
            delta=server_dp_delta,
            max_grad_norm=1.0,
            noise_multiplier=0.5  # Less noise at server level
        )
        
        print(f"DP Server Strategy initialized:")
        print(f"  Server DP: ε={server_dp_epsilon}, δ={server_dp_delta}")
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate model parameters with differential privacy"""
        
        print(f"\n=== Round {server_round} Aggregation ===")
        
        # Call parent aggregation first
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_result is None:
            return None
            
        parameters, metrics = aggregated_result
        
        # Add server-side DP noise to aggregated parameters
        if len(results) > 0:
            print("Adding server-side DP noise to aggregated parameters...")
            
            # Convert parameters to tensors
            from flwr.common import parameters_to_ndarrays
            param_arrays = parameters_to_ndarrays(parameters)
            
            # Add noise to each parameter
            noisy_arrays = []
            total_noise_added = 0.0
            
            for i, param_array in enumerate(param_arrays):
                # Calculate sensitivity based on number of clients
                sensitivity = 2.0 / len(results)  # L2 sensitivity for averaging
                
                # Create DP noise
                noise = create_dp_noise(
                    shape=param_array.shape,
                    sensitivity=sensitivity,
                    epsilon=self.server_dp_manager.epsilon / len(param_arrays),  # Split budget
                    device=torch.device('cpu')
                ).numpy()
                
                # Add noise
                noisy_param = param_array + noise
                noisy_arrays.append(noisy_param)
                
                noise_magnitude = np.linalg.norm(noise)
                total_noise_added += noise_magnitude
                
                print(f"  Parameter {i}: shape={param_array.shape}, "
                      f"noise_magnitude={noise_magnitude:.6f}")
            
            print(f"Total noise magnitude added: {total_noise_added:.6f}")
            
            # Convert back to Flower parameters
            from flwr.common import ndarrays_to_parameters
            noisy_parameters = ndarrays_to_parameters(noisy_arrays)
            
            # Store current parameters for saving
            self._current_parameters = noisy_parameters
            
            # Collect privacy metrics from clients
            client_privacy_spent = []
            for _, fit_res in results:
                if "privacy_spent" in fit_res.metrics:
                    client_privacy_spent.append(fit_res.metrics["privacy_spent"])
            
            if client_privacy_spent:
                avg_client_privacy = np.mean(client_privacy_spent)
                max_client_privacy = np.max(client_privacy_spent)
                
                privacy_info = {
                    'round': server_round,
                    'avg_client_privacy_spent': avg_client_privacy,
                    'max_client_privacy_spent': max_client_privacy,
                    'server_noise_added': total_noise_added,
                    'num_clients': len(results)
                }
                
                self.privacy_metrics.append(privacy_info)
                
                print(f"Privacy Summary:")
                print(f"  Average client privacy spent: ε={avg_client_privacy:.4f}")
                print(f"  Maximum client privacy spent: ε={max_client_privacy:.4f}")
                print(f"  Server noise magnitude: {total_noise_added:.6f}")
            
            # Save model after each round
            if server_round % 2 == 0 or server_round == 5:  # Save every 2 rounds and final
                model_path = f"{self.model_save_path}_dp_round_{server_round}.pth"
                info_path = f"{self.model_save_path}_dp_round_{server_round}_info.pkl"
                self._save_dp_model(noisy_parameters, model_path, info_path, server_round)
            
            return noisy_parameters, metrics
        
        return aggregated_result
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results and track privacy"""
        
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated_result is not None:
            loss, metrics = aggregated_result
            if "accuracy" in metrics:
                self.round_accuracies.append(metrics["accuracy"])
                print(f"Round {server_round} - Accuracy: {metrics['accuracy']:.4f}")
        
        return aggregated_result
    
    def _save_dp_model(self, parameters: Parameters, model_path: str, 
                      info_path: str, server_round: int):
        """Save DP model with privacy information"""
        
        try:
            print(f"Saving DP model for round {server_round}...")
            
            # Create model and load parameters
            model = LoanClassifier(input_size=self.input_size)
            
            from flwr.common import parameters_to_ndarrays
            arrays = parameters_to_ndarrays(parameters)
            
            params_dict = zip(model.state_dict().keys(), arrays)
            state_dict = {k: torch.from_numpy(v).float() for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            
            # Create directory
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model
            torch.save(model.state_dict(), model_path)
            
            # Save model info with privacy details
            model_info = {
                'input_size': self.input_size,
                'model_class': 'LoanClassifier',
                'architecture': str(model),
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'server_round': server_round,
                'differential_privacy': True,
                'server_dp_epsilon': self.server_dp_manager.epsilon,
                'server_dp_delta': self.server_dp_manager.delta,
                'privacy_metrics_history': self.privacy_metrics.copy(),
                'round_accuracies': self.round_accuracies.copy()
            }
            
            with open(info_path, 'wb') as f:
                pickle.dump(model_info, f)
            
            print(f"✅ DP model saved: {model_path}")
            print(f"✅ DP info saved: {info_path}")
            
            # Save final model
            if server_round == 5:
                final_model_path = f"{self.model_save_path}_dp_final.pth"
                final_info_path = f"{self.model_save_path}_dp_final_info.pkl"
                
                torch.save(model.state_dict(), final_model_path)
                with open(final_info_path, 'wb') as f:
                    pickle.dump(model_info, f)
                
                # Save privacy analysis
                privacy_analysis_path = f"{self.model_save_path}_privacy_analysis.pkl"
                privacy_analysis = {
                    'total_rounds': server_round,
                    'privacy_metrics_per_round': self.privacy_metrics,
                    'final_accuracy': self.round_accuracies[-1] if self.round_accuracies else 0.0,
                    'server_dp_config': {
                        'epsilon': self.server_dp_manager.epsilon,
                        'delta': self.server_dp_manager.delta
                    }
                }
                
                with open(privacy_analysis_path, 'wb') as f:
                    pickle.dump(privacy_analysis, f)
                
                print(f"✅ Final DP model saved: {final_model_path}")
                print(f"✅ Privacy analysis saved: {privacy_analysis_path}")
            
        except Exception as e:
            print(f"❌ Error saving DP model: {e}")
            import traceback
            traceback.print_exc()

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics with privacy information"""
    
    # Standard weighted average for accuracy
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    aggregated_accuracy = sum(accuracies) / sum(examples)
    
    # Aggregate privacy metrics
    privacy_spent_values = []
    for num_examples, m in metrics:
        if "privacy_spent" in m:
            privacy_spent_values.append(m["privacy_spent"])
    
    result = {"accuracy": aggregated_accuracy}
    
    if privacy_spent_values:
        result["avg_privacy_spent"] = float(np.mean(privacy_spent_values))
        result["max_privacy_spent"] = float(np.max(privacy_spent_values))
    
    print(f"Round accuracy: {aggregated_accuracy:.4f}")
    if privacy_spent_values:
        print(f"Average privacy spent: ε={result['avg_privacy_spent']:.4f}")
        print(f"Maximum privacy spent: ε={result['max_privacy_spent']:.4f}")
    
    return result

def fit_config(server_round: int):
    """Configuration for training rounds"""
    config = {
        "server_round": server_round,
        "epochs": 3 if server_round < 3 else 2,  # Fewer epochs for DP
        "learning_rate": 0.001,
        "add_noise_to_params": True,  # Add noise to parameters
    }
    return config

def evaluate_config(server_round: int):
    """Configuration for evaluation rounds"""
    return {"server_round": server_round}

if __name__ == "__main__":
    print("="*70)
    print("DIFFERENTIAL PRIVACY FEDERATED LEARNING SERVER")
    print("="*70)
    
    # Configuration
    INPUT_SIZE = 11
    MODEL_SAVE_PATH = "saved_models/dp_global_model"
    
    # DP Configuration
    SERVER_DP_CONFIG = {
        'server_dp_epsilon': 1.0,  # Server-side privacy budget
        'server_dp_delta': 1e-5,
    }
    
    print(f"Configuration:")
    print(f"  Input size: {INPUT_SIZE}")
    print(f"  Server DP epsilon: {SERVER_DP_CONFIG['server_dp_epsilon']}")
    print(f"  Server DP delta: {SERVER_DP_CONFIG['server_dp_delta']}")
    print(f"  Model save path: {MODEL_SAVE_PATH}")
    print("="*70)
    
    # Create DP strategy
    strategy = DPFedAvgStrategy(
        model_save_path=MODEL_SAVE_PATH,
        input_size=INPUT_SIZE,
        **SERVER_DP_CONFIG,
        
        # Standard FL configuration
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        
        # Configuration functions
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        
        initial_parameters=None,
    )
    
    print("Starting DP Federated Learning Server...")
    
    # Start server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
    
    print("="*70)
    print("DP FEDERATED LEARNING COMPLETED!")
    print("Models with differential privacy have been saved.")
    print("="*70)