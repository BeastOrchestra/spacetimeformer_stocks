from argparse import ArgumentParser
import random
import sys
import warnings
import os
import uuid

import pytorch_lightning as pl
import torch

import spacetimeformer as stf
from TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
import csv
import pandas
import numpy as np
_MODELS = ["spacetimeformer", "mtgnn", "heuristic", "lstm", "lstnet", "linear", "s4"]

_DSETS = [
    "stocks",
    "exchange",
]


def create_model(config):
    x_dim, yc_dim, yt_dim = None, None, None
    if config.dset == "stocks":
        x_dim = 95
        yc_dim = 2 # Can reduce to specific features. i.e you could forecast only 'Close' (yc_dim=1)
        yt_dim = 2
    elif config.dset == "exchange":
        x_dim = 6
        yc_dim = 8
        yt_dim = 8

    assert x_dim is not None
    assert yc_dim is not None
    assert yt_dim is not None

    if config.model == "lstm":
        forecaster = stf.lstm_model.LSTM_Forecaster(
            # encoder
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            time_emb_dim=config.time_emb_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p,
            # training
            learning_rate=config.learning_rate,
            teacher_forcing_prob=config.teacher_forcing_start,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
        )

    elif config.model == "heuristic":
        forecaster = stf.heuristic_model.Heuristic_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            target_points=config.target_points,
            loss=config.loss,
            method=config.method,
        )
    elif config.model == "mtgnn":
        forecaster = stf.mtgnn_model.MTGNN_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            target_points=config.target_points,
            gcn_depth=config.gcn_depth,
            dropout_p=config.dropout_p,
            node_dim=config.node_dim,
            dilation_exponential=config.dilation_exponential,
            conv_channels=config.conv_channels,
            subgraph_size=config.subgraph_size,
            skip_channels=config.skip_channels,
            end_channels=config.end_channels,
            residual_channels=config.residual_channels,
            layers=config.layers,
            propalpha=config.propalpha,
            tanhalpha=config.tanhalpha,
            learning_rate=config.learning_rate,
            kernel_size=config.kernel_size,
            l2_coeff=config.l2_coeff,
            time_emb_dim=config.time_emb_dim,
            loss=config.loss,
            linear_window=config.linear_window,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
            use_revin=config.use_revin,
        )
    elif config.model == "lstnet":
        forecaster = stf.lstnet_model.LSTNet_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            hidRNN=config.hidRNN,
            hidCNN=config.hidCNN,
            hidSkip=config.hidSkip,
            CNN_kernel=config.CNN_kernel,
            skip=config.skip,
            dropout_p=config.dropout_p,
            output_fun=config.output_fun,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
        )
    elif config.model == "spacetimeformer":
        if hasattr(config, "context_points") and hasattr(config, "target_points"):
            max_seq_len = config.context_points + config.target_points
        elif hasattr(config, "max_len"):
            max_seq_len = config.max_len
        else:
            raise ValueError("Undefined max_seq_len")
        forecaster = stf.spacetimeformer_model.Spacetimeformer_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            max_seq_len=max_seq_len,
            start_token_len=config.start_token_len,
            attn_factor=config.attn_factor,
            d_model=config.d_model,
            d_queries_keys=config.d_qk,
            d_values=config.d_v,
            n_heads=config.n_heads,
            e_layers=config.enc_layers,
            d_layers=config.dec_layers,
            d_ff=config.d_ff,
            dropout_emb=config.dropout_emb,
            dropout_attn_out=config.dropout_attn_out,
            dropout_attn_matrix=config.dropout_attn_matrix,
            dropout_qkv=config.dropout_qkv,
            dropout_ff=config.dropout_ff,
            pos_emb_type=config.pos_emb_type,
            use_final_norm=not config.no_final_norm,
            global_self_attn=config.global_self_attn,
            local_self_attn=config.local_self_attn,
            global_cross_attn=config.global_cross_attn,
            local_cross_attn=config.local_cross_attn,
            performer_kernel=config.performer_kernel,
            performer_redraw_interval=config.performer_redraw_interval,
            attn_time_windows=config.attn_time_windows,
            use_shifted_time_windows=config.use_shifted_time_windows,
            norm=config.norm,
            activation=config.activation,
            init_lr=config.init_lr,
            base_lr=config.base_lr,
            warmup_steps=config.warmup_steps,
            decay_factor=config.decay_factor,
            initial_downsample_convs=config.initial_downsample_convs,
            intermediate_downsample_convs=config.intermediate_downsample_convs,
            embed_method=config.embed_method,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            class_loss_imp=config.class_loss_imp,
            recon_loss_imp=config.recon_loss_imp,
            time_emb_dim=config.time_emb_dim,
            null_value=config.null_value,
            pad_value=config.pad_value,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
            use_val=not config.no_val,
            use_time=not config.no_time,
            use_space=not config.no_space,
            use_given=not config.no_given,
            recon_mask_skip_all=config.recon_mask_skip_all,
            recon_mask_max_seq_len=config.recon_mask_max_seq_len,
            recon_mask_drop_seq=config.recon_mask_drop_seq,
            recon_mask_drop_standard=config.recon_mask_drop_standard,
            recon_mask_drop_full=config.recon_mask_drop_full,
        )
    elif config.model == "linear":
        forecaster = stf.linear_model.Linear_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            linear_shared_weights=config.linear_shared_weights,
            use_revin=config.use_revin,
            use_seasonal_decomp=config.use_seasonal_decomp,
        )
    elif config.model == "s4":
        forecaster = stf.s4_model.S4_Forecaster(
            context_points=config.context_points,
            target_points=config.target_points,
            d_state=config.d_state,
            d_model=config.d_model,
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            layers=config.layers,
            time_emb_dim=config.time_emb_dim,
            channels=config.channels,
            dropout_p=config.dropout_p,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            linear_shared_weights=config.linear_shared_weights,
            use_revin=config.use_revin,
            use_seasonal_decomp=config.use_seasonal_decomp,
        )

    return forecaster

def create_parser():
    model = sys.argv[1]
    dset = sys.argv[2]

    # Throw error now before we get confusing parser issues
    assert (
        model in _MODELS
    ), f"Unrecognized model (`{model}`). Options include: {_MODELS}"
    assert dset in _DSETS, f"Unrecognized dset (`{dset}`). Options include: {_DSETS}"

    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dset")

    if dset == "stocks":
        parser.add_argument("--train_data_path", type=str, default="spacetimeformer/data/train",
                            help="Path to the training data for the 'stocks' dataset")
        parser.add_argument("--test_data_path", type=str, default="spacetimeformer/data/test",
                            help="Path to the test data for the 'stocks' dataset")
        parser.add_argument("--oos_data_path", type=str, default="spacetimeformer/data/oos",
                            help="Path to the out-of-sample data for the 'stocks' dataset")
        parser.add_argument("--context_points", type=int, required=True, help="Number of context points")
        parser.add_argument("--target_points", type=int, required=True, help="Number of target points to predict")
        parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    stf.data.DataModule.add_cli(parser)


    if model == "lstm":
        stf.lstm_model.LSTM_Forecaster.add_cli(parser)
        stf.callbacks.TeacherForcingAnnealCallback.add_cli(parser)
    elif model == "lstnet":
        stf.lstnet_model.LSTNet_Forecaster.add_cli(parser)
    elif model == "mtgnn":
        stf.mtgnn_model.MTGNN_Forecaster.add_cli(parser)
    elif model == "heuristic":
        stf.heuristic_model.Heuristic_Forecaster.add_cli(parser)
    elif model == "spacetimeformer":
        stf.spacetimeformer_model.Spacetimeformer_Forecaster.add_cli(parser)
    elif model == "linear":
        stf.linear_model.Linear_Forecaster.add_cli(parser)
    elif model == "s4":
        stf.s4_model.S4_Forecaster.add_cli(parser)
    stf.callbacks.TimeMaskedLossCallback.add_cli(parser)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_samples", type=int, default=8)
    parser.add_argument("--attn_plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--no_earlystopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--trials", type=int, default=1, help="How many consecutive trials to run"
    )

    if len(sys.argv) > 3 and sys.argv[3] == "-h":
        parser.print_help()
        sys.exit(0)

    return parser

# folder:
# ./spacetimeformer_stocks/data/STF_LOG_DIR