from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

import spacetimeformer as stf


class Spacetimeformer_Forecaster(stf.Forecaster):
    def __init__(
        self,
        d_yc: int = 1,
        d_yt: int = 1,
        d_x: int = 4,
        max_seq_len: int = None,
        start_token_len: int = 64,
        attn_factor: int = 5,
        d_model: int = 200,
        d_queries_keys=50,
        d_values=50,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 2,
        d_ff: int = 800,
        dropout_emb: float = 0.1,
        dropout_qkv: float = 0.0,
        dropout_ff: float = 0.2,
        dropout_attn_out: float = 0.0,
        dropout_attn_matrix: float = 0.0,
        pos_emb_type: str = "abs",
        global_self_attn: str = "performer",
        local_self_attn: str = "performer",
        global_cross_attn: str = "performer",
        local_cross_attn: str = "performer",
        performer_kernel: str = "relu",
        embed_method: str = "spatio-temporal",
        performer_relu: bool = True,
        performer_redraw_interval: int = 1000,
        attn_time_windows: int = 1,
        use_shifted_time_windows: bool = True,
        activation: str = "gelu",
        norm: str = "batch",
        use_final_norm: bool = True,
        init_lr: float = 1e-10,
        base_lr: float = 5e-4,
        warmup_steps: float = 1000,
        decay_factor: float = 0.8,
        initial_downsample_convs: int = 0,
        intermediate_downsample_convs: int = 0,
        l2_coeff: float = 1e-3,
        loss: str = "mse",
        class_loss_imp: float = 1e-3,
        recon_loss_imp: float = 0,
        time_emb_dim: int = 6,
        null_value: float = None,
        pad_value: float = None,
        linear_window: int = 0,
        linear_shared_weights: bool = False,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        use_val: bool = True,
        use_time: bool = True,
        use_space: bool = True,
        use_given: bool = True,
        recon_mask_skip_all: float = 1.0,
        recon_mask_max_seq_len: int = 5,
        recon_mask_drop_seq: float = 0.1,
        recon_mask_drop_standard: float = 0.2,
        recon_mask_drop_full: float = 0.05,
        toroid_wrap: bool = False,  # Add the toroid_wrap argument
        verbose=True,
    ):
        super().__init__(
            d_x=d_x,
            d_yc=d_yc,
            d_yt=d_yt,
            l2_coeff=l2_coeff,
            loss=loss,
            linear_window=linear_window,
            use_revin=use_revin,
            use_seasonal_decomp=use_seasonal_decomp,
            linear_shared_weights=linear_shared_weights,
        )
        self.spacetimeformer = stf.spacetimeformer_model.nn.Spacetimeformer(
            d_yc=d_yc,
            d_yt=d_yt,
            d_x=d_x,
            start_token_len=start_token_len,
            attn_factor=attn_factor,
            d_model=d_model,
            d_queries_keys=d_queries_keys,
            d_values=d_values,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            initial_downsample_convs=initial_downsample_convs,
            intermediate_downsample_convs=intermediate_downsample_convs,
            dropout_emb=dropout_emb,
            dropout_attn_out=dropout_attn_out,
            dropout_attn_matrix=dropout_attn_matrix,
            dropout_qkv=dropout_qkv,
            dropout_ff=dropout_ff,
            pos_emb_type=pos_emb_type,
            global_self_attn=global_self_attn,
            local_self_attn=local_self_attn,
            global_cross_attn=global_cross_attn,
            local_cross_attn=local_cross_attn,
            activation=activation,
            device=self.device,
            norm=norm,
            use_final_norm=use_final_norm,
            embed_method=embed_method,
            performer_attn_kernel=performer_kernel,
            performer_redraw_interval=performer_redraw_interval,
            attn_time_windows=attn_time_windows,
            use_shifted_time_windows=use_shifted_time_windows,
            time_emb_dim=time_emb_dim,
            verbose=True,
            null_value=null_value,
            pad_value=pad_value,
            max_seq_len=max_seq_len,
            use_val=use_val,
            use_time=use_time,
            use_space=use_space,
            use_given=use_given,
            recon_mask_skip_all=recon_mask_skip_all,
            recon_mask_max_seq_len=recon_mask_max_seq_len,
            recon_mask_drop_seq=recon_mask_drop_seq,
            recon_mask_drop_standard=recon_mask_drop_standard,
            recon_mask_drop_full=recon_mask_drop_full,
        )
        self.toroid_wrap = toroid_wrap  # Store the toroid_wrap flag
        self.start_token_len = start_token_len
        self.init_lr = init_lr
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.embed_method = embed_method
        self.class_loss_imp = class_loss_imp
        self.recon_loss_imp = recon_loss_imp
        self.set_null_value(null_value)
        self.pad_value = pad_value
        self.save_hyperparameters()

        qprint = lambda _msg_: print(_msg_) if verbose else None
        qprint(f" *** Spacetimeformer (v1.5) Summary: *** ")
        qprint(f"\t\tModel Dim: {d_model}")
        qprint(f"\t\tFF Dim: {d_ff}")
        qprint(f"\t\tEnc Layers: {e_layers}")
        qprint(f"\t\tDec Layers: {d_layers}")
        qprint(f"\t\tEmbed Dropout: {dropout_emb}")
        qprint(f"\t\tFF Dropout: {dropout_ff}")
        qprint(f"\t\tAttn Out Dropout: {dropout_attn_out}")
        qprint(f"\t\tAttn Matrix Dropout: {dropout_attn_matrix}")
        qprint(f"\t\tQKV Dropout: {dropout_qkv}")
        qprint(f"\t\tL2 Coeff: {l2_coeff}")
        qprint(f"\t\tWarmup Steps: {warmup_steps}")
        qprint(f"\t\tNormalization Scheme: {norm}")
        qprint(f"\t\tAttention Time Windows: {attn_time_windows}")
        qprint(f"\t\tShifted Time Windows: {use_shifted_time_windows}")
        qprint(f"\t\tPosition Emb Type: {pos_emb_type}")
        qprint(f"\t\tRecon Loss Imp: {recon_loss_imp}")
        qprint(f" ***                                  *** ")

    @property
    def train_step_forward_kwargs(self):
        return {"output_attn": False}

    @property
    def eval_step_forward_kwargs(self):
        return {"output_attn": False}

    # ... (rest of the class remains unchanged)
