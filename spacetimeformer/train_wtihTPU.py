from argparse import ArgumentParser
import random
import sys
import warnings
import os
import uuid
import multiprocessing
cpuCount = multiprocessing.cpu_count()

# imports pytorch
import torch
# imports the torch_xla package
import torch_xla.core.xla_model as xm

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.overrides import has_torch_function_variadic, handle_torch_function
import warnings

import spacetimeformer as stf
from TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
import csv
import pandas
import numpy as np
_MODELS = ["spacetimeformer", "mtgnn", "heuristic", "lstm", "lstnet", "linear", "s4"]

_DSETS = [
    "asos",
    "metr-la",
    "pems-bay",
    "stocks",
    "exchange",
    "precip",
    "toy2",
    "solar_energy",
    "syn",
    "mnist",
    "cifar",
    "copy",
    "cont_copy",
    "m4",
    "wiki",
    "ettm1",
    "weather",
    "monash",
    "hangzhou",
    "traffic",
]


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

    elif dset == "precip":
        stf.data.precip.GeoDset.add_cli(parser)
        stf.data.precip.CONUS_Precip.add_cli(parser)
    elif dset == "metr-la" or dset == "pems-bay":
        stf.data.metr_la.METR_LA_Data.add_cli(parser)
    elif dset == "syn":
        stf.data.synthetic.SyntheticData.add_cli(parser)
        stf.data.CSVTorchDset.add_cli(parser)
    elif dset == "mnist":
        stf.data.image_completion.MNISTDset.add_cli(parser)
    elif dset == "cifar":
        stf.data.image_completion.CIFARDset.add_cli(parser)
    elif dset == "copy":
        stf.data.copy_task.CopyTaskDset.add_cli(parser)
    elif dset == "cont_copy":
        stf.data.cont_copy_task.ContCopyTaskDset.add_cli(parser)
    elif dset == "m4":
        stf.data.m4.M4TorchDset.add_cli(parser)
    elif dset == "wiki":
        stf.data.wiki.WikipediaTorchDset.add_cli(parser)
    elif dset == "monash":
        stf.data.monash.MonashDset.add_cli(parser)
    elif dset == "hangzhou":
        stf.data.metro.MetroData.add_cli(parser)
    else:
        stf.data.CSVTimeSeries.add_cli(parser)
        stf.data.CSVTorchDset.add_cli(parser)
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


def create_model(config):
    x_dim, yc_dim, yt_dim = None, None, None
    if config.dset == "metr-la":
        x_dim = 2
        yc_dim = 207
        yt_dim = 207
    elif config.dset == "stocks":
        x_dim = 95
        yc_dim = 2 # Can reduce to specific features. i.e you could forecast only 'Close' (yc_dim=1)
        yt_dim = 2
    elif config.dset == "exchange":
        x_dim = 6
        yc_dim = 8
        yt_dim = 8
    elif config.dset == "toy2":
        x_dim = 6
        yc_dim = 20
        yt_dim = 20
    elif config.dset == "syn":
        x_dim = 5
        yc_dim = 20
        yt_dim = 20
    elif config.dset == "mnist":
        x_dim = 1
        yc_dim = 28
        yt_dim = 28
    elif config.dset == "cifar":
        x_dim = 1
        yc_dim = 3
        yt_dim = 3
    elif config.dset == "copy" or config.dset == "cont_copy":
        x_dim = 1
        yc_dim = config.copy_vars
        yt_dim = config.copy_vars
    elif config.dset == "m4":
        x_dim = 4
        yc_dim = 1
        yt_dim = 1
    elif config.dset == "wiki":
        x_dim = 2
        yc_dim = 1
        yt_dim = 1
    elif config.dset == "monash":
        x_dim = 4
        yc_dim = 1
        yt_dim = 1
    elif config.dset == "ettm1":
        x_dim = 4
        yc_dim = 7
        yt_dim = 7
    elif config.dset == "weather":
        x_dim = 3
        yc_dim = 21
        yt_dim = 21
    elif config.dset == "hangzhou":
        x_dim = 4
        yc_dim = 160
        yt_dim = 160
    elif config.dset == "traffic":
        x_dim = 2
        yc_dim = 862
        yt_dim = 862
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

# def weighted_mse_loss(input: Tensor, target: Tensor) -> Tensor:
#     # ... [previous code] ...
# this doesnt work in batch
#     total_weighted_squared_diffs = 0
#     for i in range(input.shape[1]):
#         expanded_input, expanded_target = torch.broadcast_tensors(input, target)
#         n = expanded_input.shape[0]
#         weights = torch.arange(1, n + 1, dtype=torch.float32, device=expanded_input.device)**2
#         weights /= weights.sum()
#         print(weights)
#         # Reshape weights to make them broadcastable
#         # Adding a dimension so that weights shape becomes [256, 1]
#         weights = weights.unsqueeze(1)

#         # Apply weights to squared differences
#         weighted_squared_diffs = weights * (expanded_input[:,i] - expanded_target[:,i])**2
#         total_weighted_squared_diffs += weighted_squared_diffs.sum()

#     return total_weighted_squared_diffs / (input.shape[1] * input.shape[0])


def create_dset(config):
    INV_SCALER = lambda x: x
    SCALER = lambda x: x
    NULL_VAL = None
    PLOT_VAR_IDXS = None
    PLOT_VAR_NAMES = None
    PAD_VAL = None

    if config.dset == "metr-la" or config.dset == "pems-bay":
        if config.dset == "pems-bay":
            assert (
                "pems_bay" in config.data_path
            ), "Make sure to switch to the pems-bay file!"
        data = stf.data.metr_la.METR_LA_Data(config.data_path)
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.metr_la.METR_LA_Torch,
            dataset_kwargs={"data": data},
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = data.inverse_scale
        SCALER = data.scale
        NULL_VAL = 0.0

    elif config.dset == "hangzhou":
        data = stf.data.metro.MetroData(config.data_path)
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.metro.MetroTorch,
            dataset_kwargs={"data": data},
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = data.inverse_scale
        SCALER = data.scale
        NULL_VAL = 0.0

    elif config.dset == "precip":
        dset = stf.data.precip.GeoDset(dset_dir=config.dset_dir, var="precip")
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.precip.CONUS_Precip,
            dataset_kwargs={
                "dset": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        NULL_VAL = -1.0
    elif config.dset == "syn":
        dset = stf.data.synthetic.SyntheticData(config.data_path)
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
    elif config.dset in ["mnist", "cifar"]:
        if config.dset == "mnist":
            config.target_points = 28 - config.context_points
            datasetCls = stf.data.image_completion.MNISTDset
            PLOT_VAR_IDXS = [18, 24]
            PLOT_VAR_NAMES = ["18th row", "24th row"]
        else:
            config.target_points = 32 * 32 - config.context_points
            datasetCls = stf.data.image_completion.CIFARDset
            PLOT_VAR_IDXS = [0]
            PLOT_VAR_NAMES = ["Reds"]
        DATA_MODULE = stf.data.DataModule(
            datasetCls=datasetCls,
            dataset_kwargs={"context_points": config.context_points},
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
    elif config.dset == "copy":
        # set these manually in case the model needs them
        config.context_points = config.copy_length + int(
            config.copy_include_lags
        )  # seq + lags
        config.target_points = config.copy_length
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.copy_task.CopyTaskDset,
            dataset_kwargs={
                "length": config.copy_length,
                "copy_vars": config.copy_vars,
                "lags": config.copy_lags,
                "mask_prob": config.copy_mask_prob,
                "include_lags": config.copy_include_lags,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
    elif config.dset == "cont_copy":
        # set these manually in case the model needs them
        config.context_points = config.copy_length + int(
            config.copy_include_lags
        )  # seq + lags
        config.target_points = config.copy_length
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.cont_copy_task.ContCopyTaskDset,
            dataset_kwargs={
                "length": config.copy_length,
                "copy_vars": config.copy_vars,
                "lags": config.copy_lags,
                "include_lags": config.copy_include_lags,
                "magnitude_matters": config.copy_mag_matters,
                "freq_shift": config.copy_freq_shift,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
    elif config.dset == "m4":
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.m4.M4TorchDset,
            dataset_kwargs={
                "data_path": config.data_path,
                "resolutions": args.resolutions,
                "max_len": args.max_len,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            collate_fn=stf.data.m4.pad_m4_collate,
            overfit=args.overfit,
        )
        NULL_VAL = -1.0
        PAD_VAL = -1.0

    elif config.dset == "wiki":
        DATA_MODULE = stf.data.DataModule(
            stf.data.wiki.WikipediaTorchDset,
            dataset_kwargs={
                "data_path": config.data_path,
                "forecast_duration": args.forecast_duration,
                "max_len": args.max_len,
            },
            batch_size=args.batch_size,
            workers=args.workers,
            collate_fn=stf.data.wiki.pad_wiki_collate,
            overfit=args.overfit,
        )
        NULL_VAL = -1.0
        PAD_VAL = -1.0
        SCALER = stf.data.wiki.WikipediaTorchDset.scale
        INV_SCALER = stf.data.wiki.WikipediaTorchDset.inverse_scale
    elif config.dset == "monash":
        root_dir = config.root_dir
        DATA_MODULE = stf.data.monash.monash_dloader.make_monash_dmodule(
            root_dir=root_dir,
            max_len=config.max_len,
            include=config.include,
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=config.overfit,
        )
        NULL_VAL = -64.0
        PAD_VAL = -64.0
    elif config.dset == "ettm1":
        target_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        dset = stf.data.CSVTimeSeries(
            data_path=config.data_path,
            target_cols=target_cols,
            ignore_cols=[],
            val_split=4.0 / 20,  # from informer
            test_split=4.0 / 20,  # from informer
            time_col_name="date",
            time_features=["month", "day", "weekday", "hour"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
        # PAD_VAL = -32.0
        PLOT_VAR_NAMES = target_cols
        PLOT_VAR_IDXS = [i for i in range(len(target_cols))]
    elif config.dset == "weather":
        data_path = config.data_path
        dset = stf.data.CSVTimeSeries(
            data_path=config.data_path,
            target_cols=[],
            ignore_cols=[],
            # paper says 7:1:2 split
            val_split=1.0 / 10,
            test_split=2.0 / 10,
            time_col_name="date",
            time_features=["day", "hour", "minute"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
        PLOT_VAR_NAMES = ["OT", "p (mbar)", "raining (s)"]
        PLOT_VAR_IDXS = [20, 0, 15]
    else:
        time_col_name = "Datetime"
        data_path = config.data_path
        time_features = ["year", "month", "day", "weekday", "hour", "minute"]
        if config.dset == "asos":
            if data_path == "auto":
                data_path = "./data/temperature-v1.csv"
            target_cols = ["ABI", "AMA", "ACT", "ALB", "JFK", "LGA"]
        elif config.dset == "solar_energy":
            if data_path == "auto":
                data_path = "./data/solar_AL_converted.csv"
            target_cols = [str(i) for i in range(137)]
        elif "toy" in config.dset:
            if data_path == "auto":
                if config.dset == "toy2":
                    data_path = "./data/toy_dset2.csv"
                else:
                    raise ValueError(f"Unrecognized toy dataset {config.dset}")
            target_cols = [f"D{i}" for i in range(1, 21)]
        elif config.dset == "exchange":
            if data_path == "auto":
                data_path = "./data/exchange_rate_converted.csv"
            target_cols = [
                "Australia",
                "United Kingdom",
                "Canada",
                "Switzerland",
                "China",
                "Japan",
                "New Zealand",
                "Singapore",
            ]

        elif config.dset == "stocks":
            config.phase = "train"
            if config.phase == "train":
                data_path = 'spacetimeformer/data/train' # train
            elif config.phase == "test":
                data_path = 'spacetimeformer/data/test' # test
            else:
                data_path = 'spacetimeformer/data/oos'  # out-of-sample or validation

            data_module = TimeSeriesDataset(data_folder=data_path, context_length=config.context_points, forecast_length=config.target_points)
            data_loader = DataLoader(data_module, batch_size=config.batch_size, shuffle=True if config.phase == "train" else False, num_workers=4)
            target_cols = ['open', 'high', 'low', 'Close', 'vclose', 'vopen', 'vhigh', 'vlow',
                           'VIX', 'SPY', 'TNX', 'rsi14', 'rsi9', 'rsi24', 'MACD5355macddiff',
                           'MACD5355macddiffslope', 'MACD5355macd', 'MACD5355macdslope',
                           'MACD5355macdsig', 'MACD5355macdsigslope', 'MACD12269macddiff',
                           'MACD12269macddiffslope', 'MACD12269macd', 'MACD12269macdslope',
                           'MACD12269macdsig', 'MACD12269macdsigslope', 'lowTail', 'highTail',
                           'openTail', 'IntradayBar', 'IntradayRange', 'CloseOverSMA5',
                           'CloseOverSMA10', 'CloseOverSMA12', 'CloseOverSMA20', 'CloseOverSMA30',
                           'CloseOverSMA65', 'CloseOverSMA50', 'CloseOverSMA100',
                           'CloseOverSMA200', 'VolOverSMA5', 'VolOverSMA10', 'VolOverSMA12',
                           'VolOverSMA20', 'VolOverSMA30', 'VolOverSMA65', 'VolOverSMA50',
                           'VolOverSMA100', 'VolOverSMA200', 'Ret1day', 'Ret4day', 'Ret8day',
                           'Ret12day', 'Ret24day', 'Ret72day', 'Ret240day', 'RSC', 'bands_l',
                           'bands_u', 'ADX', 'cloudA', 'cloudB', 'closeVsIchA', 'closeVsIchB',
                           'IchAvIchB', 'CondVol_1', 'CondVol_4', 'CondVol_8', 'CondVol_12',
                           'CondVol_24', 'CondVol_72', 'CondVol_240', 'CV1vCV4', 'CV4vCV8',
                           'CV8vCV12', 'CV12vCV24', 'CV8vCV24', 'CV24vCV240', 'RSC_VIX',
                           'RSC_VIX_IV', 'RSC_VIX_real', 'RSC_VIX_IV_real', 'RSC_IV_gar',
                           'close_spy_corr22', 'close_tnx_corr22', 'vclose_VIX_corr22',
                           'garch_IV_corr22', 'close_spy_corr65', 'close_tnx_corr65',
                           'vclose_VIX_corr65', 'garch_IV_corr65', 'close_spy_corr252',
                           'close_tnx_corr252', 'vclose_VIX_corr252', 'garch_IV_corr252']


        elif config.dset == "traffic":
            if data_path == "auto":
                data_path = "./data/traffic.csv"
            target_cols = [f"Lane {i}" for i in range(862)]
            time_col_name = "FakeTime"
            time_features = ["month", "day"]

        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols="all",
            time_col_name=time_col_name,
            time_features=time_features,
            val_split=0.2,
            test_split=0.2,
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    if config.dset =='stocks':

        return (
            data_loader,
            INV_SCALER,
            SCALER,
            NULL_VAL,
            PLOT_VAR_IDXS,
            PLOT_VAR_NAMES,
            PAD_VAL,
        )
    else:
        return (
            DATA_MODULE,
            INV_SCALER,
            SCALER,
            NULL_VAL,
            PLOT_VAR_IDXS,
            PLOT_VAR_NAMES,
            PAD_VAL,
        )
# data_loader

def create_callbacks(config, save_dir):
    filename = f"{config.run_name}_" + str(uuid.uuid1()).split("-")[0]
    model_ckpt_dir = os.path.join(save_dir, filename)
    config.model_ckpt_dir = model_ckpt_dir
    saving = pl.callbacks.ModelCheckpoint(
        dirpath=model_ckpt_dir,
        monitor="val/loss",
        mode="min",
        filename=f"{config.run_name}" + "{epoch:02d}",
        save_top_k=1,
        auto_insert_metric_name=True,
    )
    callbacks = [saving]

    if not config.no_earlystopping:
        callbacks.append(
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="val/loss",
                patience=config.patience,
            )
        )

    if config.wandb:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    if config.model == "lstm":
        callbacks.append(
            stf.callbacks.TeacherForcingAnnealCallback(
                start=config.teacher_forcing_start,
                end=config.teacher_forcing_end,
                steps=config.teacher_forcing_anneal_steps,
            )
        )
    if config.time_mask_loss:
        callbacks.append(
            stf.callbacks.TimeMaskedLossCallback(
                start=config.time_mask_start,
                end=config.time_mask_end,
                steps=config.time_mask_anneal_steps,
            )
        )
    return callbacks

def append_to_csv(y_t, predictions, csv_file='predictions.csv'):
    # Reshape tensors
    y_t = y_t.reshape(-1, 2)
    predictions = predictions.reshape(-1, 2)
    # Concatenate along the second dimension
    combined = torch.cat((y_t, predictions), dim=1)
    # Convert to Pandas DataFrame
    df = pandas.DataFrame(combined.cpu().numpy(), columns=['y_t_1', 'y_t_2', 'pred_1', 'pred_2'])
    # Append to CSV
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)


def main(args):
    # Initialization and Setup
    log_dir = os.getenv("STF_LOG_DIR", "./data/STF_LOG_DIR")
    args.use_gpu = True
    # device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    device = xm.xla_device()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if args.wandb:
        import wandb
        project = os.getenv("STF_WANDB_PROJ")
        entity = os.getenv("STF_WANDB_ACCT")
        experiment = wandb.init(project=project, entity=entity, config=args, dir=log_dir, reinit=True)
        config = wandb.config
        wandb.run.name = args.run_name
        wandb.run.save()
        logger = pl.loggers.WandbLogger(experiment=experiment, save_dir=log_dir)

    # Data Preparation
    if args.dset == "stocks":
        # Custom DataLoader for 'stocks'
        args.null_value = None # NULL_VAL
        args.pad_value = None

        train_loader = DataLoader(TimeSeriesDataset(data_folder='spacetimeformer/data/train', context_length=args.context_points, forecast_length=args.target_points),batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(TimeSeriesDataset(data_folder='spacetimeformer/data/test', context_length=args.context_points, forecast_length=args.target_points), batch_size=args.batch_size, shuffle=False, num_workers=4)
        oos_loader = DataLoader(TimeSeriesDataset(data_folder='spacetimeformer/data/oos', context_length=args.context_points, forecast_length=args.target_points),batch_size=args.batch_size, shuffle=False, num_workers=4)

        # data_loader, inv_scaler, scaler, null_val, plot_var_idxs, plot_var_names, pad_val = create_dset(args)
    else:
        # Standard DataModule for other datasets
        data_module, inv_scaler, scaler, null_val, plot_var_idxs, plot_var_names, pad_val = create_dset(args)
        args.null_value = None
        args.pad_value = pad_val

    # Model Training and Evaluation
    forecaster = create_model(args)
    forecaster = forecaster.to(device)  # Move the model to the specified device

    if args.dset == "stocks":
        # Custom Training Loop for 'stocks'
        optimizer = torch.optim.Adam(forecaster.parameters(), lr=args.learning_rate)
        loss_function = torch.nn.MSELoss()  # Assuming MSE loss for regression
        for epoch in range(args.epochs):
            forecaster.train()  # Set the model to training mode
            total_train_loss = 0
            for batch_idx, (context, forecast) in enumerate(train_loader):
                # Unpack  batch into x_c, y_c, x_t, y_t
                # original
                x_c = context[:, :-args.target_points, :]  # Context features
                y_c = context[:, :-args.target_points, [3, 4]]  # Context targets
                x_t = context[:, -args.target_points:, :]  # Target features
                y_t = forecast[:, :args.target_points, [3, 4]] # Target targets
                # Experimental
                # x_c = context[:, :30, :]
                # y_c = context[:, 30:, [3, 4]]
                # x_t = forecast[:, :30, :]
                # y_t = forecast[:, 30:, [3, 4]]
                # Move data to the appropriate device
                x_c, y_c, x_t, y_t = x_c.to(device), y_c.to(device), x_t.to(device), y_t.to(device)

                optimizer.zero_grad()
                model_output = forecaster(x_c, y_c, x_t, y_t)
                # Extract predictions from model output
                predictions = model_output[0] if isinstance(model_output, tuple) else model_output
                # Calculate loss
                loss = loss_function(predictions, y_t)
                # loss = weighted_mse_loss(predictions, y_t)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            average_train_loss = total_train_loss / len(train_loader)

            # Test Phase
            forecaster.eval()
            total_test_loss = 0
            with torch.no_grad():
                for context, forecast in test_loader:
                    # Original Below
                    x_c = context[:, :-args.target_points, :]
                    y_c = context[:, :-args.target_points, [3, 4]]
                    x_t = context[:, -args.target_points:, :]
                    y_t = forecast[:, :args.target_points, [3, 4]]

                    # # Experimental
                    # x_c = context[:, :30, :]
                    # y_c = context[:, 30:, [3, 4]]
                    # x_t = forecast[:, :30, :]
                    # y_t = forecast[:, 30:, [3, 4]]
                    x_c, y_c, x_t, y_t = x_c.to(device), y_c.to(device), x_t.to(device), y_t.to(device)
                    
                    model_output = forecaster(x_c, y_c, x_t, y_t)
                    predictions = model_output[0] if isinstance(model_output, tuple) else model_output
                    test_loss = loss_function(predictions, y_t)
                    # test_loss = weighted_mse_loss(predictions, y_t)
                    total_test_loss += test_loss.item()

            average_test_loss = total_test_loss / len(test_loader)
            # print(f"Epoch {epoch}, Test Loss: {average_test_loss}")

            # Out-of-Sample Phase
            total_oos_loss = 0

            oos_results = []

            # Out-of-Sample Phase
            total_oos_loss = 0
            forecaster.eval()
            with torch.no_grad():
                for context, forecast in oos_loader:
                    # Original
                    x_c = context[:, :-args.target_points, :]
                    y_c = context[:, :-args.target_points, [3, 4]]
                    x_t = context[:, -args.target_points:, :]
                    y_t = forecast[:, :args.target_points, [3, 4]]
                    # experimental
                    # x_c = context[:, :30, :]
                    # y_c = context[:, 30:, [3, 4]]
                    # x_t = forecast[:, :30, :]
                    # y_t = forecast[:, 30:, [3, 4]]
                    x_c, y_c, x_t, y_t = x_c.to(device), y_c.to(device), x_t.to(device), y_t.to(device)
                    
                    model_output = forecaster(x_c, y_c, x_t, y_t)
                    predictions = model_output[0] if isinstance(model_output, tuple) else model_output

                    oos_loss = loss_function(predictions, y_t)
                    # oos_loss = weighted_mse_loss(predictions, y_t)
                    total_oos_loss += oos_loss.item()

                    # Store results for each batch
                    actual = y_t.cpu().numpy()
                    pred = predictions.cpu().numpy()
                    oos_results.extend(zip(actual.reshape(-1, 2), pred.reshape(-1, 2)))
            average_oos_loss = total_oos_loss / len(oos_loader)

            # Save OOS results to CSV at the end of each epoch
            with open(f'predictions_epoch_{epoch}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Actual_price', 'Actual_volatility', 'Predicted_price', 'Predicted_volatility'])  # Header
                for actual, predicted in oos_results:
                    writer.writerow([*actual, *predicted])

            print(f"Epoch {epoch}, "
                f"Training Loss: {average_train_loss}, "
                f"Test Loss: {average_test_loss}, "
                f"Out-of-Sample Loss: {average_oos_loss}")

        # Save a checkpoint after each epoch
            checkpoint_path = os.path.join(log_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(forecaster.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    else:
        # Standard Training with PyTorch Lightning for other datasets
        forecaster.set_inv_scaler(inv_scaler)
        forecaster.set_scaler(scaler)
        forecaster.set_null_value(null_val)

        # Callbacks and Trainer Configuration
        callbacks = create_callbacks(args, save_dir=log_dir)
        trainer = pl.Trainer(
            gpus=args.gpus,
            callbacks=callbacks,
            logger=logger if args.wandb else None,
            # ... additional trainer configurations ...
        )

        # Fitting the model
        trainer.fit(forecaster, datamodule=data_module)

        # Testing the model
        trainer.test(datamodule=data_module, ckpt_path="best")

    # WANDB Experiment Finish (if applicable)
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)


# if __name__ == "__main__":
#     # CLI
#     parser = create_parser()
#     args = parser.parse_args()

#     for trial in range(args.trials):
#         main(args)