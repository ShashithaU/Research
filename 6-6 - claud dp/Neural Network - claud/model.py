import torch
import torch.nn as nn

class LoanClassifier(nn.Module):
    def __init__(self, input_size=11, hidden_sizes=[1792, 896, 448, 224], dropout_rate=0.4):
        """
        Neural network for loan classification
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super(LoanClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer (NO SIGMOID!)
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward_logits(self, x):
        """Forward pass returning raw logits (for training)"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.model(x)
    
    def forward(self, x):
        """Forward pass returning probabilities (for inference)"""
        logits = self.forward_logits(x)
        return torch.sigmoid(logits)
    
    def get_model_info(self):
        """Return information about the model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": str(self.model)
        }

# Alternative more complex model for better performance
class AdvancedLoanClassifier(nn.Module):
    def __init__(self, input_size=11, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        """
        Advanced neural network with batch normalization and skip connections
        """
        super(AdvancedLoanClassifier, self).__init__()
        
        self.input_size = input_size
        
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_bn = nn.BatchNorm1d(hidden_sizes[0])
        
        # Hidden layers with batch normalization
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        
        # Dropout and activation
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU networks"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Input layer
        x = self.input_layer(x)
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.input_bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Hidden layers
        for hidden_layer, batch_norm in zip(self.hidden_layers, self.batch_norms):
            residual = x  # Skip connection
            x = hidden_layer(x)
            
            if x.size(0) > 1:  # Only apply batch norm if batch size > 1
                x = batch_norm(x)
            
            x = self.relu(x)
            x = self.dropout(x)
            
            # Add residual connection if dimensions match
            if residual.shape[-1] == x.shape[-1]:
                x = x + residual
        
        # Output layer
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        return x

# Test the models
if __name__ == "__main__":
    # Test basic model
    print("Testing LoanClassifier...")
    model = LoanClassifier(input_size=11)
    print("Model info:", model.get_model_info())
    
    # Test with dummy data
    dummy_input = torch.randn(32, 11)  # Batch of 32 samples
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}, Output shape: {output.shape}")
    
    # Test advanced model
    print("\nTesting AdvancedLoanClassifier...")
    advanced_model = AdvancedLoanClassifier(input_size=11)
    output_advanced = advanced_model(dummy_input)
    print(f"Advanced model output shape: {output_advanced.shape}")
    
    print("\nModels created successfully!")