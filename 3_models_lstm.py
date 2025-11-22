"""
Bidirectional LSTM model for BLE indoor localization.

Architecture:
- Input: Temporal RSSI sequences (timesteps x beacons)
- 2 Bidirectional LSTM layers (128, 64 units)
- Dropout regularization (0.3, 0.5)
- Fully connected output layer
- Softmax activation for location classification
"""

import torch
import torch.nn as nn


class BiLSTMLocalization(nn.Module):
    """Bidirectional LSTM for indoor localization."""
    
    def __init__(self, input_size=10, hidden_size1=128, hidden_size2=64, 
                 num_classes=10, dropout=0.3):
        """
        Args:
            input_size: Number of BLE beacons
            hidden_size1: First LSTM layer hidden units
            hidden_size2: Second LSTM layer hidden units
            num_classes: Number of location classes
            dropout: Dropout rate
        """
        super(BiLSTMLocalization, self).__init__()
        
        # First Bidirectional LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.bn1 = nn.BatchNorm1d(hidden_size1 * 2)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second Bidirectional LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_size1 * 2,  # *2 for bidirectional
            hidden_size=hidden_size2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.bn2 = nn.BatchNorm1d(hidden_size2 * 2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size2 * 2, 64)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.bn1(lstm1_out[:, -1, :])  # Take last timestep
        lstm1_out = self.dropout1(lstm1_out)
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out.unsqueeze(1))
        lstm2_out = self.bn2(lstm2_out[:, -1, :])
        lstm2_out = self.dropout2(lstm2_out)
        
        # Fully connected layers
        out = self.fc1(lstm2_out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        
        return out


def load_model(checkpoint_path, device='cpu'):
    """Load a trained model from checkpoint."""
    model = BiLSTMLocalization()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


if __name__ == "__main__":
    # Test model instantiation
    model = BiLSTMLocalization(input_size=10, num_classes=10)
    
    # Test forward pass
    batch_size = 32
    sequence_length = 10
    input_size = 10
    
    x = torch.randn(batch_size, sequence_length, input_size)
    output = model(x)
    
    print(f"Model instantiated successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
