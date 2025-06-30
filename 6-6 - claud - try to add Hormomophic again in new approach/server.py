import flwr as fl
import torch
import os
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import Metrics, Parameters, ndarrays_to_parameters, FitRes, Scalar
from flwr.server.client_proxy import ClientProxy
from model import LoanClassifier
import pickle
import numpy as np
import tenseal as ts
import pickle
import numpy as np
import tenseal as ts

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


class HEFedAvg(fl.server.strategy.FedAvg):
    """A FedAvg strategy that handles homomorphic encryption."""
    
    def __init__(self, *args, **kwargs):
        # Extract num_rounds and remove it from kwargs to avoid passing it to FedAvg
        self.num_rounds = kwargs.pop("num_rounds")
        super().__init__(*args, **kwargs)
        
        # Create TenSEAL context
        self.tenseal_context = ts.context(
        ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.tenseal_context.generate_galois_keys()
        self.tenseal_context.global_scale = 2**40
        
        # We send the context (including public key) to clients
        # The server keeps the secret key for final decryption
        self.context_bytes = self.tenseal_context.serialize(save_secret_key=False)


    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[ClientProxy, FitRes]]:
        """Configure the next round of training and send HE context."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Add the TenSEAL context to the config
        config["tenseal_context"] = self.context_bytes
        
        # Send PLAINTEXT parameters to the client
        fit_ins = fl.common.FitIns(parameters, config)
        
        # Sample clients
        clients = client_manager.sample(
            num_clients=self.min_fit_clients, min_num_clients=self.min_fit_clients
        )
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager):
        """Configure the next round of evaluation."""
        if self.on_evaluate_config_fn is None:
            return []
        
        config = self.on_evaluate_config_fn(server_round)
        
        # Send PLAINTEXT parameters for evaluation
        eval_ins = fl.common.EvaluateIns(parameters, config)
        
        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients, min_num_clients=self.min_evaluate_clients
        )
        # Corrected bug: was returning fit_ins, should be eval_ins
        return [(client, eval_ins) for client in clients]


    # This duplicate method should be removed.
    # def configure_evaluate(...):
    

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]):
        """Configure the next round of evaluation."""
        if self.on_evaluate_config_fn is None:
            return []
        
        config = self.on_evaluate_config_fn(server_round)
        
        # Encrypt parameters for evaluation
        ndarrays = fl.common.parameters_to_ndarrays(parameters)
        encrypted_params = self.encrypt_parameters(ndarrays)
        # Corrected class name from EvalIns to EvaluateIns
        eval_ins = fl.common.EvaluateIns(fl.common.Parameters(tensors=[encrypted_params], tensor_type="bytes"), config)
        
        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients, min_num_clients=self.min_evaluate_clients
        )
        return [(client, eval_ins) for client in clients]
    

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]):
        """Aggregate encrypted results from clients."""
        if not results:
            return None, {}
            
        # The results are encrypted, deserialize them
        deserialized_updates = []
        for _, fit_res in results:
            # 1. Unpickle the list of serialized vectors
            serialized_vectors = pickle.loads(fit_res.parameters.tensors[0])
            # 2. Load each CKKSVector from its serialized representation
            update = [ts.CKKSVector.load(self.tenseal_context, ser_vec) for ser_vec in serialized_vectors]
            deserialized_updates.append(update)


        # 3. Sum the encrypted model updates homomorphically (layer by layer)
        aggregated_updates = [sum(layer) for layer in zip(*deserialized_updates)]
        
        # 4. Average the aggregated updates
        # TenSEAL doesn't support in-place division, so we multiply by the inverse
        for i in range(len(aggregated_updates)):
            aggregated_updates[i] *= (1 / len(results))
        
        # 5. Decrypt the new global model
        decrypted_ndarrays = [np.array(vec.decrypt(), dtype=np.float32) for vec in aggregated_updates]

        # Reshape parameters to original model shapes
        model = LoanClassifier(input_size=11) # Assuming input size
        keys = model.state_dict().keys()
        original_shapes = [model.state_dict()[k].shape for k in keys]
        
        reshaped_params = []
        for param, shape in zip(decrypted_ndarrays, original_shapes):
            reshaped_params.append(param.reshape(shape))

        # ... (rest of the HEFedAvg class methods) ...
        final_parameters = fl.common.ndarrays_to_parameters(reshaped_params)
        
        # Save the decrypted model at the end of all rounds
        if server_round == self.num_rounds:
            print("Server: Saving final decrypted model...")
            save_global_model(final_parameters, "saved_models/global_model_final_he.pth", "saved_models/global_model_final_he_info.pkl")

        return final_parameters, {}


    def encrypt_parameters(self, parameters: fl.common.NDArrays) -> bytes:
        """Helper to encrypt parameters."""
        encrypted_vectors = [ts.ckks_vector(self.tenseal_context, p.flatten()) for p in parameters]
        # Serialize the list of encrypted vectors using pickle
        return pickle.dumps([vec.serialize() for vec in encrypted_vectors])

    def decrypt_parameters(self, encrypted_parameters: bytes) -> fl.common.NDArrays:
        """Helper to decrypt parameters."""
        # Deserialize the list of bytes using pickle
        serialized_vectors = pickle.loads(encrypted_parameters)
        decrypted = []
        for ser_vec in serialized_vectors:
            enc_v = ts.CKKSVector.load(self.tenseal_context, ser_vec)
            decrypted.append(np.array(enc_v.decrypt(), dtype=np.float32))
        return decrypted
# This block should be de-indented to be outside the HEFedAvg class
if __name__ == "__main__":
    # Configuration
    INPUT_SIZE = 11  # Update this based on your actual feature count
    MODEL_SAVE_PATH = "saved_models/global_model"
    NUM_ROUNDS = 5
    
    # Create HE strategy
    strategy = HEFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        num_rounds=NUM_ROUNDS, # Pass num_rounds to strategy
    )

    print("Starting Flower server with Homomorphic Encryption...")
    
    # Start server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    
    print("Server completed! HE Global model has been saved.")