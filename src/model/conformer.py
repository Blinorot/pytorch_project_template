import torch
from torch import nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, 
            groups=d_model, padding=(kernel_size - 1) // 2
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        x = self.layer_norm(x)
        x = x.transpose(1, 2) # (B, D, T)
        
        # GLU activation
        x = self.pointwise_conv1(x) # (B, 2D, T)
        x = F.glu(x, dim=1) # (B, D, T)
        
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = torch.nn.functional.silu(x)
        
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        return x.transpose(1, 2) # (B, T, D)

class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, dropout=dropout)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mha_norm = nn.LayerNorm(d_model)
        self.conv = ConformerConvModule(d_model, kernel_size=kernel_size, dropout=dropout)
        self.ff2 = FeedForwardModule(d_model, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D)
        x = x + 0.5 * self.ff1(x)
        
        residual = x
        x = self.mha_norm(x)
        x, _ = self.mha(x, x, x)
        x = x + residual
        
        x = x + self.conv(x)
        
        x = x + 0.5 * self.ff2(x)
        
        x = self.final_norm(x)
        return x

class Conformer(nn.Module):
    def __init__(self, n_feats, n_tokens, d_model=256, n_heads=4, n_layers=12, kernel_size=31, dropout=0.1, **kwargs):
        super().__init__()
        
        self.subsampling = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.input_proj = nn.Linear(d_model * (n_feats // 4), d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, kernel_size, dropout) 
            for _ in range(n_layers)
        ])
        
        self.fc = nn.Linear(d_model, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        # (B, F, T) -> (B, 1, F, T)
        x = spectrogram.unsqueeze(1)
        x = self.subsampling(x) # (B, D, F/4, T/4)
        
        B, D, F, T = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() # (B, T, D, F)
        x = x.view(B, T, D * F)
        
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
            
        output = self.fc(x)
        log_probs = F.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        l1 = (input_lengths + 1) // 2
        l2 = (l1 + 1) // 2
        return l2

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

