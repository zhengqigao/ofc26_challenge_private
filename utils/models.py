import torch 
import torch.nn as nn

class BasicFNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        
    def forward(self, x):
        return self.layers(x)
    
# utils/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """Residual 1D conv block: [B,C,T] -> [B,C,T]."""
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.dropout(h)
        h = self.conv2(h)
        return self.act(x + h)


class BasicCNN(nn.Module):
    """
    Drop-in replacement for your current BasicFNN:
      input:  [B, 4 + 95 + 95]  (global + spectra + wss)
      output: [B, 95]          (gain spectrum)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_bins: int = 95,
        global_dim: int = 4,
        global_hidden: int = 64,
        global_out: int = 64,
        conv_channels: int = 64,
        conv_layers: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert output_dim == n_bins, f"Expected output_dim=={n_bins}, got {output_dim}"
        assert input_dim == global_dim + 2 * n_bins, (
            f"Expected input_dim=={global_dim}+2*{n_bins}={global_dim + 2*n_bins}, got {input_dim}"
        )

        self.n_bins = n_bins
        self.global_dim = global_dim

        # global features -> embedding g
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, global_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden, global_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden, global_out),
        )

        # per-bin features: [spectra_i, wss_i] + global embedding
        self.in_proj = nn.Linear(2 + global_out, conv_channels)

        self.blocks = nn.ModuleList([
            ResidualConvBlock(conv_channels, kernel_size=kernel_size, dropout=dropout)
            for _ in range(conv_layers)
        ])

        # per-bin regression head
        self.y_head = nn.Conv1d(conv_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        x: [B, input_dim]
        layout: [global(4), spectra(95), wss(95)]
        """
        B = x.shape[0]
        g = x[:, :self.global_dim]                              # [B, 4]
        spectra = x[:, self.global_dim:self.global_dim+self.n_bins]          # [B, 95]
        wss = x[:, self.global_dim+self.n_bins:self.global_dim+2*self.n_bins] # [B, 95]

        # global embedding
        g_emb = self.global_mlp(g)                              # [B, global_out]

        # build per-bin sequence
        x_seq = torch.stack([spectra, wss], dim=-1)              # [B, 95, 2]
        g_rep = g_emb[:, None, :].expand(B, self.n_bins, g_emb.shape[-1])   # [B, 95, global_out]
        h = torch.cat([x_seq, g_rep], dim=-1)                    # [B, 95, 2+global_out]

        # project -> conv trunk
        h = self.in_proj(h)                                     # [B, 95, C]
        h = h.transpose(1, 2).contiguous()                      # [B, C, 95]

        for blk in self.blocks:
            h = blk(h)

        y_hat = self.y_head(h).squeeze(1)                       # [B, 95]
        return y_hat
