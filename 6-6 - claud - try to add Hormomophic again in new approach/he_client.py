import flwr as fl
import torch
import tenseal as ts
from collections import OrderedDict
import numpy as np
import pickle
from flwr.common import Code, FitIns, FitRes, GetParametersIns, GetParametersRes, Status, EvaluateIns, EvaluateRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays

# Global variable to hold the TenSEAL context
CTX = None

def encrypt_parameters(context: ts.Context, parameters: fl.common.NDArrays) -> bytes:
    """Encrypt model parameters using TenSEAL."""
    encrypted_vectors = [ts.ckks_vector(context, param.flatten()) for param in parameters]
    return pickle.dumps([vec.serialize() for vec in encrypted_vectors])

def decrypt_parameters(context: ts.Context, encrypted_parameters: bytes) -> fl.common.NDArrays:
    """Decrypt model parameters using TenSEAL."""
    # This function is no longer needed on the client
    pass

class HEClient(fl.client.Client):
    """A Flower client that uses homomorphic encryption."""
    
    def __init__(self, model, x_train, y_train, x_test, y_test, train_fn, test_fn):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_fn = train_fn
        self.test_fn = test_fn
        self.device = next(model.parameters()).device

    def _get_parameters_ndarrays(self):
        """Get model parameters as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def _set_parameters_from_ndarrays(self, parameters):
        """Set model parameters from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return the current local model parameters."""
        ndarrays = self._get_parameters_ndarrays()
        parameters = fl.common.ndarrays_to_parameters(ndarrays)
        return GetParametersRes(status=Status(code=Code.OK, message="Success"), parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        """Train the model, encrypt the updates, and return them."""
        global CTX
        
        if CTX is None:
            print("Client: Receiving TenSEAL context from server.")
            context_bytes = ins.config["tenseal_context"]
            CTX = ts.context_from(context_bytes)
        
        # The server sends plaintext parameters, so no decryption is needed.
        # Convert parameters to NumPy arrays and set the model
        ndarrays = parameters_to_ndarrays(ins.parameters)
        self._set_parameters_from_ndarrays(ndarrays)
        
        # Train model
        self.train_fn(self.model, self.x_train.to(self.device), self.y_train.to(self.device), 
                      epochs=ins.config["epochs"], lr=ins.config["learning_rate"])
        
        # Get and ENCRYPT updated parameters
        updated_params_ndarrays = self._get_parameters_ndarrays()
        encrypted_updated_params = encrypt_parameters(CTX, updated_params_ndarrays)
        
        # Create FitRes with the encrypted parameters
        params_proto = Parameters(tensors=[encrypted_updated_params], tensor_type="bytes")
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=params_proto,
            num_examples=len(self.x_train),
            metrics={}
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the model on the local test set."""
        # Evaluation uses plaintext models, no HE context needed here.
        
        # Convert parameters to NumPy arrays and set the model
        ndarrays = parameters_to_ndarrays(ins.parameters)
        self._set_parameters_from_ndarrays(ndarrays)
        
        # Evaluate model
        accuracy, loss = self.test_fn(self.model, self.x_test.to(self.device), self.y_test.to(self.device))
        
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(self.x_test),
            metrics={"accuracy": float(accuracy)}
        )