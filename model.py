from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import math

## trainCNNlstm
class TransformerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, d_model=32, nhead=2):
        super(TransformerLSTM, self).__init__()
        # Transformer encoder
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        #在定义 TransformerEncoderLayer 时设置 batch_first=True
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        # LSTM
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input x shape: (batch_size, seq_length, input_dim)
        # Embedding
        x = self.embedding(x)  # (batch_size, seq_length, d_model)
        # Positional encoding
        x = self.pos_encoder(x)
        # Transformer encoder
        x = x.permute(1, 0, 2)  # Transformer expects (seq_length, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_length, d_model)
        # LSTM
        _, (h_n, _) = self.lstm(x)
        # Use the last hidden state
        out = self.fc(h_n[-1])
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x



