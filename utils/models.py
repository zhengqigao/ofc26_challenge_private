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
            nn.Linear(128, output_dim),
        )
        
    def forward(self, x):
        x = x.clone()
        x[:,4::2] = x[:,4::2] / 100 # normalization 
        return self.layers(x)

class ComplicatedFNN(nn.Module):
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
            nn.Linear(128, output_dim),
        )
        
    def forward(self, x):
        x[:,4::2] = x[:,4::2] / 100 # normalization 
        return self.layers(x)

class GatedBasicFNN(nn.Module):
    def __init__(self, input_dim = 95, output_dim = 95):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        
        self.proj = nn.Sequential(
            nn.Linear(99, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        
    def forward(self, x):
        x = x.clone()
        x[:,4::2] = x[:,4::2] / 100 # normalization
        v = torch.cat([x[:,0:4], x[:,4::2]], dim=-1) # shape [batch, 99]
        c = x[:,5::2] # control variable, shape [batch, 95]
        g = self.gate(c) # shape [batch, 128]
        v = self.proj(v) # shape [batch, 128]
        o = v + g
        result = self.layers(o) # shape [batch, 1]
        return result


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
        layer_list = []
        in_dim = global_out_dim + v_out_dim
        for i, h_dim in enumerate(head_hidden_list):
            layer_list.append(nn.Linear(in_dim, h_dim))
            if i < len(head_hidden_list) - 1:
                layer_list.append(nn.ReLU())
            in_dim = h_dim
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


class ResidualFNN(nn.Module):
    """
    带残差连接的全连接网络，训练更稳定
    使用BatchNorm和Dropout防止过拟合
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 512, 256, 128, 64], dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 残差块
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] == hidden_dims[i+1]:
                # 相同维度，使用残差连接
                self.blocks.append(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dims[i], hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                ))
            else:
                # 不同维度，普通层
                self.blocks.append(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        x = x.clone()
        x[:, 4::2] = x[:, 4::2] / 100  # normalization
        x = self.input_layer(x)
        
        for i, block in enumerate(self.blocks):
            if len(block) == 6:  # 残差块
                residual = x
                x = block(x)
                x = x + residual
                x = nn.functional.relu(x)
            else:  # 普通层
                x = block(x)
        
        return self.output_layer(x)


class AttentionFNN(nn.Module):
    """
    使用自注意力机制捕捉通道间关系的模型
    适合捕捉相邻通道的空间相关性
    """
    def __init__(self, input_dim, output_dim, embed_dim=128, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.global_dim = 4
        self.n_channels = 95
        
        # 全局特征处理
        self.global_net = nn.Sequential(
            nn.Linear(self.global_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        
        # 通道特征处理（光谱值 + WSS）
        self.channel_proj = nn.Sequential(
            nn.Linear(2, embed_dim),  # [spectra, wss] -> embed_dim
            nn.LayerNorm(embed_dim)
        )
        
        # 自注意力层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = x.clone()
        x[:, 4::2] = x[:, 4::2] / 100  # normalization
        
        # 分离全局和通道特征
        global_feat = x[:, :self.global_dim]  # [B, 4]
        spectra = x[:, 4::2]  # [B, 95]
        wss = x[:, 5::2]  # [B, 95]
        
        # 全局特征嵌入
        g_emb = self.global_net(global_feat)  # [B, embed_dim]
        
        # 通道特征
        channel_feat = torch.stack([spectra, wss], dim=-1)  # [B, 95, 2]
        channel_emb = self.channel_proj(channel_feat)  # [B, 95, embed_dim]
        
        # 添加全局特征到每个通道
        g_emb_expanded = g_emb.unsqueeze(1).expand(-1, self.n_channels, -1)  # [B, 95, embed_dim]
        x_seq = channel_emb + g_emb_expanded  # [B, 95, embed_dim]
        
        # 自注意力
        x_attn = self.transformer(x_seq)  # [B, 95, embed_dim]
        
        # 输出
        output = self.output_head(x_attn).squeeze(-1)  # [B, 95]
        return output


class ChannelWiseFNN(nn.Module):
    """
    对每个通道独立处理，然后融合的模型
    适合处理通道间的独立性
    """
    def __init__(self, input_dim, output_dim, channel_hidden=32, global_hidden=64):
        super().__init__()
        self.global_dim = 4
        self.n_channels = 95
        
        # 全局特征处理
        self.global_net = nn.Sequential(
            nn.Linear(self.global_dim, global_hidden),
            nn.ReLU(),
            nn.Linear(global_hidden, global_hidden)
        )
        
        # 每个通道的独立处理网络（共享权重）
        self.channel_net = nn.Sequential(
            nn.Linear(2, channel_hidden),  # [spectra, wss]
            nn.ReLU(),
            nn.Linear(channel_hidden, channel_hidden),
            nn.ReLU(),
            nn.Linear(channel_hidden, channel_hidden)
        )
        
        # 融合层：全局 + 通道特征 -> 输出
        self.fusion_net = nn.Sequential(
            nn.Linear(global_hidden + channel_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = x.clone()
        x[:, 4::2] = x[:, 4::2] / 100  # normalization
        
        # 全局特征
        global_feat = x[:, :self.global_dim]  # [B, 4]
        g_emb = self.global_net(global_feat)  # [B, global_hidden]
        
        # 通道特征
        spectra = x[:, 4::2]  # [B, 95]
        wss = x[:, 5::2]  # [B, 95]
        channel_feat = torch.stack([spectra, wss], dim=-1)  # [B, 95, 2]
        
        # 对每个通道独立处理
        channel_emb = self.channel_net(channel_feat)  # [B, 95, channel_hidden]
        
        # 融合全局和通道特征
        g_emb_expanded = g_emb.unsqueeze(1).expand(-1, self.n_channels, -1)  # [B, 95, global_hidden]
        fused = torch.cat([g_emb_expanded, channel_emb], dim=-1)  # [B, 95, global_hidden + channel_hidden]
        
        # 输出
        output = self.fusion_net(fused).squeeze(-1)  # [B, 95]
        return output


class LightweightFNN(nn.Module):
    """
    轻量级模型，参数量少但有效
    适合小数据集，减少过拟合风险
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim),
        )
        
    def forward(self, x):
        x = x.clone()
        x[:, 4::2] = x[:, 4::2] / 100  # normalization
        return self.layers(x)


class HybridFNN(nn.Module):
    """
    混合模型：结合全局MLP和通道级处理
    先处理全局特征，再对每个通道独立预测
    """
    def __init__(self, input_dim, output_dim, global_hidden=128, channel_hidden=32):
        super().__init__()
        self.global_dim = 4
        self.n_channels = 95
        
        # 全局特征处理（影响所有通道）
        self.global_net = nn.Sequential(
            nn.Linear(self.global_dim, global_hidden),
            nn.ReLU(),
            nn.Linear(global_hidden, global_hidden),
            nn.ReLU(),
            nn.Linear(global_hidden, global_hidden)
        )
        
        # 通道特征投影
        self.channel_proj = nn.Sequential(
            nn.Linear(2, channel_hidden),  # [spectra, wss]
            nn.ReLU(),
            nn.Linear(channel_hidden, channel_hidden)
        )
        
        # 融合并输出
        self.output_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(global_hidden + channel_hidden, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(self.n_channels)
        ])
        
    def forward(self, x):
        x = x.clone()
        x[:, 4::2] = x[:, 4::2] / 100  # normalization
        
        # 全局特征
        global_feat = x[:, :self.global_dim]  # [B, 4]
        g_emb = self.global_net(global_feat)  # [B, global_hidden]
        
        # 通道特征
        spectra = x[:, 4::2]  # [B, 95]
        wss = x[:, 5::2]  # [B, 95]
        channel_feat = torch.stack([spectra, wss], dim=-1)  # [B, 95, 2]
        c_emb = self.channel_proj(channel_feat)  # [B, 95, channel_hidden]
        
        # 对每个通道独立预测
        outputs = []
        for i in range(self.n_channels):
            fused = torch.cat([g_emb, c_emb[:, i, :]], dim=-1)  # [B, global_hidden + channel_hidden]
            out = self.output_net[i](fused)  # [B, 1]
            outputs.append(out)
        
        return torch.cat(outputs, dim=-1)  # [B, 95]


class DeepResidualFNN(nn.Module):
    """
    深度残差网络，使用多个残差块
    适合复杂模式的学习
    """
    def __init__(self, input_dim, output_dim, base_dim=128, num_blocks=4, dropout=0.2):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, base_dim),
            nn.BatchNorm1d(base_dim),
            nn.ReLU()
        )
        
        # 残差块
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Linear(base_dim, base_dim),
                nn.BatchNorm1d(base_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(base_dim, base_dim),
                nn.BatchNorm1d(base_dim),
            ))
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.ReLU(),
            nn.Linear(base_dim // 2, output_dim)
        )
        
    def forward(self, x):
        x = x.clone()
        x[:, 4::2] = x[:, 4::2] / 100  # normalization
        x = self.input_proj(x)
        
        # 残差连接
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual
            x = nn.functional.relu(x)
        
        return self.output_layer(x)


class SpectralTransformer(nn.Module):
    """
    Channel-token transformer with global conditioning and residual-on-tilt output.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        num_channels=95,
        global_dim=4,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.25,
        ff_mult=2,
        spectra_noise_std=0.0,
        global_noise_std=0.0,
    ):
        super().__init__()
        inferred = (input_dim - global_dim) // 2
        if input_dim != global_dim + inferred * 2:
            raise ValueError(f"Invalid input_dim: {input_dim} for global_dim={global_dim}.")
        if inferred != num_channels:
            num_channels = inferred
        if output_dim != num_channels:
            raise ValueError(f"output_dim {output_dim} must match num_channels {num_channels}.")

        self.global_dim = global_dim
        self.num_channels = num_channels
        self.spectra_noise_std = spectra_noise_std
        self.global_noise_std = global_noise_std

        self.global_norm = nn.LayerNorm(global_dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        self.channel_proj = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pos_emb = nn.Parameter(torch.zeros(num_channels, embed_dim))
        nn.init.normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.residual_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        self.tilt_scale = nn.Parameter(torch.tensor(1.0))
        pos = torch.linspace(-0.5, 0.5, steps=num_channels)
        self.register_buffer("channel_pos", pos, persistent=False)

    def forward(self, x):
        x = x.clone()
        raw_global = x[:, :self.global_dim]
        base_global = raw_global
        spectra = x[:, self.global_dim::2] / 100
        wss = x[:, self.global_dim + 1::2]

        if self.training and self.spectra_noise_std > 0:
            spectra = spectra + torch.randn_like(spectra) * self.spectra_noise_std
        if self.training and self.global_noise_std > 0:
            raw_global = raw_global + torch.randn_like(raw_global) * self.global_noise_std

        g_emb = self.global_mlp(self.global_norm(raw_global))
        pos = self.channel_pos.unsqueeze(0).expand(x.size(0), -1)
        channel_feat = torch.stack([spectra, wss, pos], dim=-1)
        tokens = self.channel_proj(channel_feat)
        tokens = tokens + g_emb.unsqueeze(1) + self.pos_emb.unsqueeze(0)
        tokens = self.encoder(tokens)

        residual = self.residual_head(tokens).squeeze(-1)
        base = base_global[:, 0:1] + base_global[:, 1:2] * self.tilt_scale * self.channel_pos
        return base + residual


class SpectralCNN(nn.Module):
    """
    Lightweight 1D CNN over channels with global conditioning and residual-on-tilt output.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        num_channels=95,
        global_dim=4,
        hidden_channels=32,
        dropout=0.2,
        spectra_noise_std=0.0,
        global_noise_std=0.0,
    ):
        super().__init__()
        inferred = (input_dim - global_dim) // 2
        if input_dim != global_dim + inferred * 2:
            raise ValueError(f"Invalid input_dim: {input_dim} for global_dim={global_dim}.")
        if inferred != num_channels:
            num_channels = inferred
        if output_dim != num_channels:
            raise ValueError(f"output_dim {output_dim} must match num_channels {num_channels}.")

        self.global_dim = global_dim
        self.num_channels = num_channels
        self.spectra_noise_std = spectra_noise_std
        self.global_noise_std = global_noise_std

        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.conv_in = nn.Conv1d(3, hidden_channels, kernel_size=3, padding=1)
        self.conv_mid = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv_out = nn.Conv1d(hidden_channels, 1, kernel_size=1)

        self.tilt_scale = nn.Parameter(torch.tensor(1.0))
        pos = torch.linspace(-0.5, 0.5, steps=num_channels)
        self.register_buffer("channel_pos", pos, persistent=False)

    def forward(self, x):
        x = x.clone()
        raw_global = x[:, :self.global_dim]
        base_global = raw_global
        spectra = x[:, self.global_dim::2] / 100
        wss = x[:, self.global_dim + 1::2]

        if self.training and self.spectra_noise_std > 0:
            spectra = spectra + torch.randn_like(spectra) * self.spectra_noise_std
        if self.training and self.global_noise_std > 0:
            raw_global = raw_global + torch.randn_like(raw_global) * self.global_noise_std

        pos = self.channel_pos.to(x.device).unsqueeze(0).expand(x.size(0), -1)
        feat = torch.stack([spectra, wss, pos], dim=1)  # [B, 3, N]

        h = self.conv_in(feat)
        g = self.global_mlp(raw_global).unsqueeze(-1)
        h = h + g
        h = self.conv_mid(h)
        residual = self.conv_out(h).squeeze(1)

        base = base_global[:, 0:1] + base_global[:, 1:2] * self.tilt_scale * self.channel_pos
        return base + residual



class ImprovedSpectralCNN(nn.Module):
    """
    Lightweight 1D CNN over channels with global conditioning and residual-on-tilt output.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        num_channels=95,
        global_dim=4,
        hidden_channels=32,
        dropout=0.2,
        spectra_noise_std=0.0,
        global_noise_std=0.0,
    ):
        super().__init__()
        inferred = (input_dim - global_dim) // 2
        if input_dim != global_dim + inferred * 2:
            raise ValueError(f"Invalid input_dim: {input_dim} for global_dim={global_dim}.")
        if inferred != num_channels:
            num_channels = inferred
        if output_dim != num_channels:
            raise ValueError(f"output_dim {output_dim} must match num_channels {num_channels}.")

        self.global_dim = global_dim
        self.num_channels = num_channels
        self.spectra_noise_std = spectra_noise_std
        self.global_noise_std = global_noise_std

        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.conv_in = nn.Conv1d(3, hidden_channels, kernel_size=5, padding=2)
        self.conv_mid = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.conv_out = nn.Conv1d(hidden_channels, 1, kernel_size=1)

        self.tilt_scale = nn.Parameter(torch.tensor(1.0))
        pos = torch.linspace(-0.5, 0.5, steps=num_channels)
        self.register_buffer("channel_pos", pos, persistent=False)

    def forward(self, x):
        x = x.clone()
        raw_global = x[:, :self.global_dim]
        base_global = raw_global
        spectra = x[:, self.global_dim::2] / 100
        wss = x[:, self.global_dim + 1::2]

        if self.training and self.spectra_noise_std > 0:
            spectra = spectra + torch.randn_like(spectra) * self.spectra_noise_std
        if self.training and self.global_noise_std > 0:
            raw_global = raw_global + torch.randn_like(raw_global) * self.global_noise_std

        pos = self.channel_pos.to(x.device).unsqueeze(0).expand(x.size(0), -1)
        feat = torch.stack([spectra, wss, pos], dim=1)  # [B, 3, N]

        h = self.conv_in(feat)
        g = self.global_mlp(raw_global).unsqueeze(-1)
        h = h + g
        h = self.conv_mid(h)
        residual = self.conv_out(h).squeeze(1)

        base = base_global[:, 0:1] + base_global[:, 1:2] * self.tilt_scale * self.channel_pos
        return base + residual


class ImprovedDeepSpectralCNN(nn.Module):
    """
    Lightweight 1D CNN over channels with global conditioning and residual-on-tilt output.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        num_channels=95,
        global_dim=4,
        hidden_channels=32,
        dropout=0.2,
        spectra_noise_std=0.0,
        global_noise_std=0.0,
    ):
        super().__init__()
        inferred = (input_dim - global_dim) // 2
        if input_dim != global_dim + inferred * 2:
            raise ValueError(f"Invalid input_dim: {input_dim} for global_dim={global_dim}.")
        if inferred != num_channels:
            num_channels = inferred
        if output_dim != num_channels:
            raise ValueError(f"output_dim {output_dim} must match num_channels {num_channels}.")

        self.global_dim = global_dim
        self.num_channels = num_channels
        self.spectra_noise_std = spectra_noise_std
        self.global_noise_std = global_noise_std

        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.conv_in = nn.Sequential(
            nn.Conv1d(3, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        self.conv_mid = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.conv_out = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels // 2, 1, kernel_size=1),
        )

        self.tilt_scale = nn.Parameter(torch.tensor(1.0))
        pos = torch.linspace(-0.5, 0.5, steps=num_channels)
        self.register_buffer("channel_pos", pos, persistent=False)

    def forward(self, x):
        x = x.clone()
        raw_global = x[:, :self.global_dim]
        base_global = raw_global
        spectra = x[:, self.global_dim::2] / 100
        wss = x[:, self.global_dim + 1::2]

        if self.training and self.spectra_noise_std > 0:
            spectra = spectra + torch.randn_like(spectra) * self.spectra_noise_std
        if self.training and self.global_noise_std > 0:
            raw_global = raw_global + torch.randn_like(raw_global) * self.global_noise_std

        pos = self.channel_pos.to(x.device).unsqueeze(0).expand(x.size(0), -1)
        feat = torch.stack([spectra, wss, pos], dim=1)  # [B, 3, N]

        h = self.conv_in(feat)
        g = self.global_mlp(raw_global).unsqueeze(-1)
        h = h + g
        h = self.conv_mid(h)
        residual = self.conv_out(h).squeeze(1)

        base = base_global[:, 0:1] + base_global[:, 1:2] * self.tilt_scale * self.channel_pos
        return base + residual
    
    
class ImprovedEmbedDeepSpectralCNN(nn.Module):
    """
    Lightweight 1D CNN over channels with global conditioning and residual-on-tilt output.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        num_channels=95,
        global_dim=4,
        hidden_channels=32,
        hidden_embed_dim=4,
        dropout=0.2,
        spectra_noise_std=0.0,
        global_noise_std=0.0,
    ):
        super().__init__()
        inferred = (input_dim - global_dim) // 2
        if input_dim != global_dim + inferred * 2:
            raise ValueError(f"Invalid input_dim: {input_dim} for global_dim={global_dim}.")
        if inferred != num_channels:
            num_channels = inferred
        if output_dim != num_channels:
            raise ValueError(f"output_dim {output_dim} must match num_channels {num_channels}.")

        self.global_dim = global_dim
        self.num_channels = num_channels
        self.spectra_noise_std = spectra_noise_std
        self.global_noise_std = global_noise_std

        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.conv_in = nn.Conv1d(3, hidden_channels, kernel_size=3, padding=1)
        self.conv_mid = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.conv_out = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels // 2, 1, kernel_size=1),
        )

        self.wss_embed = nn.Embedding(2, hidden_embed_dim)
        self.tilt_scale = nn.Parameter(torch.tensor(1.0))

        pos = torch.linspace(-0.5, 0.5, steps=num_channels)
        self.register_buffer("channel_pos", pos, persistent=False)


    def forward(self, x):
        x = x.clone()
        raw_global = x[:, :self.global_dim]
        base_global = raw_global
        spectra = x[:, self.global_dim::2] / 100
        wss = x[:, self.global_dim + 1::2]

        wss_embed = self.wss_embed(wss.long()).transpose(1, 2)

        if self.training and self.spectra_noise_std > 0:
            spectra = spectra + torch.randn_like(spectra) * self.spectra_noise_std
        if self.training and self.global_noise_std > 0:
            raw_global = raw_global + torch.randn_like(raw_global) * self.global_noise_std

        pos = self.channel_pos.to(x.device).unsqueeze(0).expand(x.size(0), -1)
        feat = torch.cat([spectra.unsqueeze(1), wss_embed, pos.unsqueeze(1)], dim=1)  # [B, 2 + hidden_embed_dim, N]
        
        h = self.conv_in(feat) # [B, hidden_channels, N]
        g = self.global_mlp(raw_global).unsqueeze(-1)
        h = h + g # [B, hidden_channels, N]

        h = self.conv_mid(h) # [B, hidden_channels, N]
        residual = self.conv_out(h).squeeze(1) # [B, N]

        base = base_global[:, 0:1] + self.tilt_scale * base_global[:, 1:2].view(-1,1) * self.channel_pos.view(1,-1)
        return base + residual