from argparse import ArgumentParser
import random
import sys
import warnings
import os
import uuid

import pytorch_lightning as pl
import torch

import spacetimeformer as stf
from TimeSeriesDataset_ContextOnly import TimeSeriesDataset_ContextOnly
from torch.utils.data import DataLoader
import csv
import pandas as pd
import numpy as np
import datetime

_MODELS = ["spacetimeformer"]

_DSETS = [
    "stocks",
]

def create_model(config):
    x_dim, yc_dim, yt_dim = None, None, None
    if config.dset == "stocks":
        x_dim = 95
        yc_dim = 2
        yt_dim = 2

    assert x_dim is not None
    assert yc_dim is not None
    assert yt_dim is not None

    if config.model == "spacetimeformer":
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
            start_token_len=getattr(config, 'start_token_len', 0),
            attn_factor=getattr(config, 'attn_factor', 1),
            d_model=config.d_model,
            d_queries_keys=getattr(config, 'd_qk', config.d_model),
            d_values=getattr(config, 'd_v', config.d_model),
            n_heads=config.n_heads,
            e_layers=config.enc_layers,
            d_layers=config.dec_layers,
            d_ff=config.d_ff,
            dropout_emb=config.dropout_emb,
            dropout_attn_out=config.dropout_attn_out,
            dropout_attn_matrix=config.dropout_attn_matrix,
            dropout_qkv=config.dropout_qkv,
            dropout_ff=config.dropout_ff,
            pos_emb_type=getattr(config, 'pos_emb_type', 'abs'),
            use_final_norm=not getattr(config, 'no_final_norm', False),
            global_self_attn=config.global_self_attn,
            local_self_attn=config.local_self_attn,
            global_cross_attn=config.global_cross_attn,
            local_cross_attn=config.local_cross_attn,
            performer_kernel=getattr(config, 'performer_kernel', 'softmax'),
            performer_redraw_interval=getattr(config, 'performer_redraw_interval', 1000),
            attn_time_windows=getattr(config, 'attn_time_windows', None),
            use_shifted_time_windows=getattr(config, 'use_shifted_time_windows', False),
            norm=getattr(config, 'norm', 'layer'),
            activation=config.activation,
            init_lr=getattr(config, 'init_lr', 1e-4),
            base_lr=getattr(config, 'base_lr', 1e-3),
            warmup_steps=getattr(config, 'warmup_steps', 2000),
            decay_factor=config.decay_factor,
            initial_downsample_convs=getattr(config, 'initial_downsample_convs', 0),
            intermediate_downsample_convs=getattr(config, 'intermediate_downsample_convs', 0),
            embed_method=config.embed_method,
            l2_coeff=getattr(config, 'l2_coeff', 0),
            loss=config.loss,
            class_loss_imp=getattr(config, 'class_loss_imp', 1.0),
            recon_loss_imp=getattr(config, 'recon_loss_imp', 1.0),
            time_emb_dim=config.time_emb_dim,
            null_value=getattr(config, 'null_value', None),
            pad_value=getattr(config, 'pad_value', None),
            linear_window=getattr(config, 'linear_window', None),
            use_revin=getattr(config, 'use_revin', False),
            linear_shared_weights=getattr(config, 'linear_shared_weights', False),
            use_seasonal_decomp=getattr(config, 'use_seasonal_decomp', False),
            use_val=not getattr(config, 'no_val', False),
            use_time=not getattr(config, 'no_time', False),
            use_space=not getattr(config, 'no_space', False),
            use_given=not getattr(config, 'no_given', False),
            recon_mask_skip_all=getattr(config, 'recon_mask_skip_all', False),
            recon_mask_max_seq_len=getattr(config, 'recon_mask_max_seq_len', None),
            recon_mask_drop_seq=getattr(config, 'recon_mask_drop_seq', 0.0),
            recon_mask_drop_standard=getattr(config, 'recon_mask_drop_standard', 0.0),
            recon_mask_drop_full=getattr(config, 'recon_mask_drop_full', 0.0),
            use_toroidal_twist=getattr(config, 'use_toroidal_twist', False),
        )
    return forecaster

def create_parser():
    model = sys.argv[1]
    dset = sys.argv[2]

    assert model in _MODELS, f"Unrecognized model (`{model}`). Options include: {_MODELS}"
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

    if model == "spacetimeformer":
        stf.spacetimeformer_model.Spacetimeformer_Forecaster.add_cli(parser)
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
    parser.add_argument("--trials", type=int, default=1, help="How many consecutive trials to run")
    parser.add_argument("--use_toroidal_twist", action="store_true", help="Enable toroidal twist attention")
    parser.add_argument("--null_value", type=float, default=None, help="Value to use for null entries")
    parser.add_argument("--pad_value", type=float, default=None, help="Value to use for padding")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")

    if len(sys.argv) > 3 and sys.argv[3] == "-h":
        parser.print_help()
        sys.exit(0)

    return parser

# ... (rest of the code remains the same)

def main(args):
    # ... (previous code remains the same)

    # Model Training and Evaluation
    forecaster = create_model(args)
    forecaster = forecaster.to(device)  # Move the model to the specified device

    output_path = "/Users/alecjeffery/Documents/Playgrounds/Python/largeModels/HighAccuracy_Oct13th.pth"
    
    # Load the weights into the model
    state_dict = torch.load(output_path, map_location=torch.device('cpu'))
    
    # Check if the model was trained with toroidal twist attention
    if args.use_toroidal_twist and 'inner_attention.twist_factor_row' not in state_dict:
        print("Warning: The loaded model does not contain toroidal twist parameters. Initializing them randomly.")
        # Initialize the twist factors if they're not in the state dict
        forecaster.load_state_dict(state_dict, strict=False)
    elif not args.use_toroidal_twist and 'inner_attention.twist_factor_row' in state_dict:
        print("Warning: The loaded model contains toroidal twist parameters, but they are not being used.")
        # Remove the twist factors from the state dict
        state_dict = {k: v for k, v in state_dict.items() if 'twist_factor' not in k}
        forecaster.load_state_dict(state_dict, strict=False)
    else:
        forecaster.load_state_dict(state_dict)

    # ... (rest of the code remains the same)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
