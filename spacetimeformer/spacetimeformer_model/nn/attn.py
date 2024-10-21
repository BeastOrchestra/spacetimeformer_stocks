from math import sqrt, log

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=False,
        scale=None,
        attention_dropout=0.1,
        toroid_wrap=False,  # New argument for toroidal wrapping
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        self.toroid_wrap = toroid_wrap  # Store the toroidal wrap flag

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

        if self.toroid_wrap:  # Apply toroidal wrapping if specified
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

        # Twisted toroidal wrap-around for rows
        wrap_around_scores[:, :, 0, :] += scores[:, :, -1, :]
        wrap_around_scores[:, :, :, 0] += scores[:, :, :, -1]

        # Twisted toroidal wrap-around for columns
        wrap_around_scores[:, :, :, -1] += scores[:, :, :, 0]
        wrap_around_scores[:, :, -1, :] += scores[:, :, 0, :]

        return wrap_around_scores


class ProbAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        toroid_wrap=False,  # New argument for toroidal wrapping
    ):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        self.toroid_wrap = toroid_wrap  # Store the toroidal wrap flag

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        scores_top, index = self._prob_QK(queries, keys, sample_k=self.factor, n_top=L_Q)

        # add scale factor
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # Apply toroidal wrapping if specified
        if self.toroid_wrap:
            scores_top = self.wrap_attention_scores(scores_top)

        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask, output_attn=output_attn
        )

        return context.transpose(2, 1).contiguous(), attn

    def wrap_attention_scores(self, scores):
        batch_size, num_heads, seq_length, _ = scores.size()
        wrap_around_scores = scores.clone()

        # Twisted toroidal wrap-around for rows
        wrap_around_scores[:, :, 0, :] += scores[:, :, -1, :]
        wrap_around_scores[:, :, :, 0] += scores[:, :, :, -1]

        # Twisted toroidal wrap-around for columns
        wrap_around_scores[:, :, :, -1] += scores[:, :, :, 0]
        wrap_around_scores[:, :, -1, :] += scores[:, :, 0, :]

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
    ):
        super(AttentionLayer, self).__init__()

        self.inner_attention = attention()
        self.query_projection = nn.Linear(d_model, d_queries_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_queries_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.dropout_qkv(self.query_projection(queries)).view(B, L, H, -1)
        keys = self.dropout_qkv(self.key_projection(keys)).view(B, S, H, -1)
        values = self.dropout_qkv(self.value_projection(values)).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries=queries,
            keys=keys,
            values=values,
            attn_mask=attn_mask,
            output_attn=output_attn,
        )

        if output_attn and attn is None:
            onehot_values = (
                torch.eye(S).unsqueeze(0).repeat(B, 1, 1).unsqueeze(2).to(values.device)
            )
            with torch.no_grad():
                attn, _ = self.inner_attention(
                    queries=queries,
                    keys=keys,
                    values=onehot_values,
                    attn_mask=attn_mask,
                )
                attn = attn.transpose(2, 1)

        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        if not output_attn:
            assert attn is None

        out = self.out_projection(out)
        return out, attn
