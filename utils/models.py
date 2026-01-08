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
        x[:,4::2] = x[:,4::2] / 100 # normalization 
        return self.layers(x)

class CustomFNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # input is [batch, 194]
        # the 0,1,2,3 columns are continuous values, system-level parameters
        # then 4-th column is continus value, and 5-th column is a latent control variable (either 0 or 1) associated with the 4-th column
        # then 6-th column is continus value, and 7-th column is a latent control variable (either 0 or 1) associated with the 6-th column
        # ...192-th column is continus value, and 193-th column is a latent control variable (either 0 or 1) associated with the 192-th column

import torch
import torch.nn as nn


class LinearGateNet(nn.Module):
    """
    Input:  [B, 194]s
      - x[:,:4] are global continuous
      - x[:,4:] are pairs (v_i, c_i) with c_i in {0,1}
        v_i at even offset, c_i at odd offset

    Output: [B, out_dim]  (e.g., 95)
    """

    def __init__(
        self,
        out_dim: int = 1,
        global_dim: int = 4,
        global_hidden_list: list[int] = [4, 4],
        token_hidden_list: list[int] = [1, 2, 4],
        head_hidden_list: list[int] = [16,32,64,32,16,8,1],
    ):
        super().__init__()
        
        
        self.global_dim = global_dim
        assert global_hidden_list[0] == global_dim
        assert token_hidden_list[0] == 1
        # for global features 4 -> global_hidden_list[-1]
        layer_list = []
        for i in range(len(global_hidden_list)-1):
            layer_list.append(nn.Linear(global_hidden_list[i], global_hidden_list[i+1]))
            layer_list.append(nn.ReLU())
        self.g_net = nn.Sequential(*layer_list)

        # for token features 1 -> token_hidden_list[-1]
        layer_list = []
        for i in range(len(token_hidden_list)-1):
            layer_list.append(nn.Linear(token_hidden_list[i], token_hidden_list[i+1]))
            layer_list.append(nn.ReLU())
        self.v_proj = nn.Sequential(*layer_list)
        
        v_out_dim = token_hidden_list[-1]
        global_out_dim = global_hidden_list[-1]
        
        # c_i -> gate scalar in (0,1); learnable but tiny
        self.c_gate = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )
        # learnable default embedding when gate is "off"
        self.off_token = nn.Parameter(torch.zeros(v_out_dim))

        # ---- Head: concat (global_out_dim, v_out_dim) -> out_dim
        layer_list = [nn.Linear(global_out_dim + v_out_dim, head_hidden_list[0]), nn.ReLU()]
        for i in range(len(head_hidden_list)-1):
            layer_list.append(nn.Linear(head_hidden_list[i], head_hidden_list[i+1]))
            layer_list.append(nn.ReLU())
        self.head = nn.Sequential(*layer_list)

    def forward(self, x):

        # global
        g = x[:, :self.global_dim]           # [B,4]
        g_feat = self.g_net(g)          # [B,global_out_dim]

        # pairs
        v = (x[:, self.global_dim::2] / 100).unsqueeze(-1)         # [B,N,1]
        c = x[:, self.global_dim+1::2].unsqueeze(-1)       # [B,N,1] (being either 0 or 1)

        # print(f"v.shape: {v.shape}, v")
        # print(f"v: {v}")
        # print(f"c.shape: {c.shape}, c")
        # print(f"c: {c}")
        e_v = self.v_proj(v)                 # [B,N,v_out_dim]

        gate = self.c_gate(c)                # [B,N,1]

        off = self.off_token.view(1, 1, -1)  # [1,1,v_out_dim]

        # gated token: when c≈0, use off_token; when c≈1, use transformed v (+ shift)
        tokens = e_v * gate + off * (1.0 - gate)  # [B,N,v_out_dim]

        z = torch.cat([g_feat.unsqueeze(1).repeat(1, tokens.shape[1], 1), tokens], dim=-1)  # [B, N, global_out_dim+v_out_dim]
        y = self.head(z)                         # [B,N,1]

        return y.squeeze(-1)
