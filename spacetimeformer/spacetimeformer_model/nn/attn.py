from math import sqrt, log

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..utils.masking import TriangularCausalMask, ProbMask
from performer_pytorch import FastAttention as _FastAttention

class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=False,
        scale=None,
        attention_dropout=0.1,
        use_toroidal_twist=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        self.use_toroidal_twist = use_toroidal_twist

        if self.use_toroidal_twist:
            self.twist_factor_row = nn.Parameter(torch.tensor(1.0))
            self.twist_factor_col = nn.Parameter(torch.tensor(1.0))

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # expand heads dimension
            scores.masked_fill_(attn_mask, -np.inf)

        if self.use_toroidal_twist:
            scores = self.wrap_attention_scores(scores)

        A = torch.nan_to_num(self.dropout(torch.softmax(scale * scores, dim=-1)))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if output_attn:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

    def wrap_attention_scores(self, scores):
        batch_size, num_heads, seq_length, _ = scores.size()
        wrap_around_scores = scores.clone()

        twist_factor_row = int(self.twist_factor_row.item())
        twist_factor_col = int(self.twist_factor_col.item())

        wrap_around_scores[:, :, 0, :] += scores[:, :, -1, :].roll(shifts=twist_factor_row, dims=-1)
        wrap_around_scores[:, :, :, 0] += scores[:, :, :, -1].roll(shifts=twist_factor_col, dims=-2)
        wrap_around_scores[:, :, :, -1] += scores[:, :, :, 0].roll(shifts=-twist_factor_col, dims=-2)
        wrap_around_scores[:, :, -1, :] += scores[:, :, 0, :].roll(shifts=-twist_factor_row, dims=-1)

        return wrap_around_scores

class ProbAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        use_toroidal_twist=False,
    ):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        self.use_toroidal_twist = use_toroidal_twist

        if self.use_toroidal_twist:
            self.twist_factor_row = nn.Parameter(torch.tensor(1.0))
            self.twist_factor_col = nn.Parameter(torch.tensor(1.0))

    # ... (rest of the ProbAttention class implementation)

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        # ... (existing implementation)

        if self.use_toroidal_twist:
            scores_top = self.wrap_attention_scores(scores_top)

        # ... (rest of the forward method)

    def wrap_attention_scores(self, scores):
        batch_size, num_heads, seq_length, _ = scores.size()
        wrap_around_scores = scores.clone()

        twist_factor_row = int(self.twist_factor_row.item())
        twist_factor_col = int(self.twist_factor_col.item())

        wrap_around_scores[:, :, 0, :] += scores[:, :, -1, :].roll(shifts=twist_factor_row, dims=-1)
        wrap_around_scores[:, :, :, 0] += scores[:, :, :, -1].roll(shifts=twist_factor_col, dims=-2)
        wrap_around_scores[:, :, :, -1] += scores[:, :, :, 0].roll(shifts=-twist_factor_col, dims=-2)
        wrap_around_scores[:, :, -1, :] += scores[:, :, 0, :].roll(shifts=-twist_factor_row, dims=-1)

        return wrap_around_scores

class PerformerAttention(_FastAttention):
    def __init__(
        self,
        mask_flag=False,
        dim_heads=None,
        ortho_scaling=0,
        feature_redraw_interval=1000,
        kernel="softmax",
        use_toroidal_twist=False,
    ):
        assert dim_heads is not None
        super().__init__(
            dim_heads=dim_heads,
            ortho_scaling=ortho_scaling,
            nb_features=max(100, int(dim_heads * log(dim_heads))),
            causal=mask_flag,
            generalized_attention=kernel == "relu",
            kernel_fn=nn.ReLU() if kernel == "relu" else "N/A",
        )
        self.redraw_interval = feature_redraw_interval
        self.register_buffer("calls_since_last_redraw", torch.tensor(0))
        self.use_toroidal_twist = use_toroidal_twist

        if self.use_toroidal_twist:
            self.twist_factor_row = nn.Parameter(torch.tensor(1.0))
            self.twist_factor_col = nn.Parameter(torch.tensor(1.0))

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        # ... (existing implementation)

        scores = super().forward(queries, keys, values)

        if self.use_toroidal_twist:
            scores = self.wrap_attention_scores(scores)

        return scores.transpose(1, 2), None

    def wrap_attention_scores(self, scores):
        batch_size, num_heads, seq_length, _ = scores.size()
        wrap_around_scores = scores.clone()

        twist_factor_row = int(self.twist_factor_row.item())
        twist_factor_col = int(self.twist_factor_col.item())

        wrap_around_scores[:, :, 0, :] += scores[:, :, -1, :].roll(shifts=twist_factor_row, dims=-1)
        wrap_around_scores[:, :, :, 0] += scores[:, :, :, -1].roll(shifts=twist_factor_col, dims=-2)
        wrap_around_scores[:, :, :, -1] += scores[:, :, :, 0].roll(shifts=-twist_factor_col, dims=-2)
        wrap_around_scores[:, :, -1, :] += scores[:, :, 0, :].roll(shifts=-twist_factor_row, dims=-1)

        return wrap_around_scores

class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_queries_keys,
        d_values,
        n_heads,
        dropout_qkv=0.0,
        mix=False,
        use_toroidal_twist=False,
    ):
        super(AttentionLayer, self).__init__()

        self.inner_attention = attention(use_toroidal_twist=use_toroidal_twist)
        self.query_projection = nn.Linear(d_model, d_queries_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_queries_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.n_heads = n_heads
        self.mix = mix

    # ... (rest of the AttentionLayer class implementation)
