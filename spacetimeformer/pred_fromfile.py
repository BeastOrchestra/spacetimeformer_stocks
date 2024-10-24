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

# import gdown



_MODELS = ["spacetimeformer"]

_DSETS = [
    "stocks",
]

def create_model(config):
    x_dim, yc_dim, yt_dim = None, None, None
    if config.dset == "stocks":
        x_dim = 95
        yc_dim = 2 # Can reduce to specific features. i.e you could forecast only 'Close' (yc_dim=1)
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
    parser.add_argument(
        "--trials", type=int, default=1, help="How many consecutive trials to run"
    )

    if len(sys.argv) > 3 and sys.argv[3] == "-h":
        parser.print_help()
        sys.exit(0)

    return parser


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

    else:
        time_col_name = "Datetime"
        data_path = config.data_path
        time_features = ["year", "month", "day", "weekday", "hour", "minute"]
        if config.dset == "stocks":
            config.phase = "predict"
            data_path = 'spacetimeformer/data/oos'  # out-of-sample or validation
            dataset = TimeSeriesDataset_ContextOnly(folder_name=data_path, file_name='data.csv', context_length=config.context_points)
            dataloader = DataLoader(dataset, batch_size=1000, shuffle=False) # 1000 so that you get all of the oos dset

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
            dataloader,
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

# Notes for improving this section. Analysis indicates that the delta of the oos_predictions
# is proportional to the return on the position. Larger delta implies larger return (pos & neg)
# Therefore you want to sort based on the "a" dataframe for the largest and smallest changes
# You will then estimate future ending location by looking at the last value in "a" and correcting w/ mu & sig

# def formatOutput(tops):  # Original
#     # Only look at the top X
#     dir=os.getcwd()
#     print('current Dir:', dir)
#     a = pd.read_csv('oos_predictions.csv', index_col=0)
#     # b = pd.read_csv('TixMuSig.csv',index_col=1)
#     b = pd.read_csv('./spacetimeformer/data/TixMuSig.csv',index_col=1)


#     col = ['Close_'+str(i) for i in range(1,11)]
#     Vcol = ['Volatility_'+str(i) for i in range(1,11)]
#     for i in a.index:
#         a.loc[i][col] = a.loc[i][col]*b.loc[i].closesig + b.loc[i].closemu
#         a.loc[i][Vcol] = a.loc[i][Vcol]*b.loc[i].volsig + b.loc[i].volmu

#     current_date = datetime.datetime.now()
#     formatted_date = f"{current_date.month}_{current_date.day}_{current_date.year}"
#     a.to_csv('oos_predictions_'+formatted_date+'.csv')

#     a['Price_PrctDelta'] = 100*(a['Close_10']-a['Close_1'])/a['Close_1']
#     a['Volatility_PrctDelta'] = 100*(a['Volatility_10']-a['Volatility_1'])/a['Volatility_1']

#     PossibleLongCalls = a[(a['Price_PrctDelta'] > 0) & (a['Volatility_PrctDelta'] > 0)]
#     PossibleLongPuts = a[(a['Price_PrctDelta'] < 0) & (a['Volatility_PrctDelta'] > 0)]

#     PossibleLongs = a[(a['Price_PrctDelta'] > 0)]
#     PossibleShorts = a[(a['Price_PrctDelta'] < 0)]

#     VolPump = a[(a['Volatility_PrctDelta'] > 0)]
#     VolDump = a[(a['Volatility_PrctDelta'] < 0)]

#     Calls=PossibleLongCalls[['Price_PrctDelta','Volatility_PrctDelta']].sort_values(by='Price_PrctDelta',ascending=False)
#     Puts=PossibleLongPuts[['Price_PrctDelta','Volatility_PrctDelta']].sort_values(by='Price_PrctDelta',ascending=True)

#     Longs=PossibleLongs[['Price_PrctDelta','Volatility_PrctDelta']].sort_values(by='Price_PrctDelta',ascending=False)
#     Shorts=PossibleShorts[['Price_PrctDelta','Volatility_PrctDelta']].sort_values(by='Price_PrctDelta',ascending=True)

#     LongVol=VolPump[['Price_PrctDelta','Volatility_PrctDelta']].sort_values(by='Volatility_PrctDelta',ascending=False)
#     ShortVol=VolDump[['Price_PrctDelta','Volatility_PrctDelta']].sort_values(by='Volatility_PrctDelta',ascending=True)

#     print('Long: ',Longs[Longs['Price_PrctDelta'] > 2].Price_PrctDelta[:tops])
#     print('Short: ',Shorts[Shorts['Price_PrctDelta'] < -2].Price_PrctDelta[:tops])

#     print('Long Calls: ', Calls[ (Calls['Price_PrctDelta'] > 2) &(Calls['Volatility_PrctDelta'] > 5)].Price_PrctDelta[:tops])
#     print('Long Puts: ',Puts[ (Puts['Price_PrctDelta']< -2) & (Puts['Volatility_PrctDelta'] > 5)].Price_PrctDelta[:tops])

#     print('Long Volatility: ',LongVol[LongVol.Volatility_PrctDelta > 20].Volatility_PrctDelta[:tops])
#     print('Short Volatility: ',ShortVol[ShortVol.Volatility_PrctDelta < -20].Volatility_PrctDelta[:tops])
def formatOutput(tops): # Revised 
    # Only look at the top X

    a = pd.read_csv('oos_predictions.csv', index_col=0)
    b = pd.read_csv('./spacetimeformer/data/TixMuSig.csv',index_col=1)


    col = ['Close_'+str(i) for i in range(1,11)]
    Vcol = ['Volatility_'+str(i) for i in range(1,11)]
    for i in a.index:
        a.loc[i][col] = a.loc[i][col]*b.loc[i].closesig + b.loc[i].closemu
        a.loc[i][Vcol] = a.loc[i][Vcol]*b.loc[i].volsig + b.loc[i].volmu

    current_date = datetime.datetime.now()
    formatted_date = f"{current_date.month}_{current_date.day}_{current_date.year}"
    a.to_csv('oos_predictions_'+formatted_date+'.csv')

    a = pd.read_csv('oos_predictions.csv', index_col=0) # Use this to rank the predicted move

    a['Price_PrctDelta'] = a['Close_10']-a['Close_1']
    a['Volatility_PrctDelta'] = a['Volatility_10']-a['Volatility_1']
# Instead of > | < 0, use specific thresholds that have been identified during returns analysis
    PossibleLongCalls = a[(a['Price_PrctDelta'] > 0) & (a['Volatility_PrctDelta'] > 0)]
    PossibleLongPuts = a[(a['Price_PrctDelta'] < 0) & (a['Volatility_PrctDelta'] > 0)]

    PossibleLongs = a[(a['Price_PrctDelta'] > 0)]
    PossibleShorts = a[(a['Price_PrctDelta'] < 0)]

    VolPump = a[(a['Volatility_PrctDelta'] > 0)]
    VolDump = a[(a['Volatility_PrctDelta'] < 0)]

    Calls=PossibleLongCalls[['Price_PrctDelta','Volatility_PrctDelta']].sort_values(by='Price_PrctDelta',ascending=False)
    Puts=PossibleLongPuts[['Price_PrctDelta','Volatility_PrctDelta']].sort_values(by='Price_PrctDelta',ascending=True)

    Longs=PossibleLongs[['Price_PrctDelta','Volatility_PrctDelta']].sort_values(by='Price_PrctDelta',ascending=False)
    Shorts=PossibleShorts[['Price_PrctDelta','Volatility_PrctDelta']].sort_values(by='Price_PrctDelta',ascending=True)

    LongVol=VolPump[['Price_PrctDelta','Volatility_PrctDelta']].sort_values(by='Volatility_PrctDelta',ascending=False)
    ShortVol=VolDump[['Price_PrctDelta','Volatility_PrctDelta']].sort_values(by='Volatility_PrctDelta',ascending=True)
    eqThresh = .2
    print('Long: ',Longs[Longs['Price_PrctDelta'] > eqThresh].Price_PrctDelta[:tops])
    print('Short: ',Shorts[Shorts['Price_PrctDelta'] < -eqThresh].Price_PrctDelta[:tops])
    Shorts[Shorts['Price_PrctDelta'] < -eqThresh].Price_PrctDelta[:tops].to_csv('shorts.csv')
    optThresh = .5
    print('Long Calls: ', Calls[ (Calls['Price_PrctDelta'] > eqThresh) &(Calls['Volatility_PrctDelta'] > optThresh)].Price_PrctDelta[:tops])
    print('Long Puts: ',Puts[ (Puts['Price_PrctDelta']< -eqThresh) & (Puts['Volatility_PrctDelta'] > optThresh)].Price_PrctDelta[:tops])

    print('Long Volatility: ',LongVol[LongVol.Volatility_PrctDelta > optThresh].Volatility_PrctDelta[:tops])
    print('Short Volatility: ',ShortVol[ShortVol.Volatility_PrctDelta < -optThresh].Volatility_PrctDelta[:tops])

def main(args):
    # Initialization and Setup
    log_dir = os.getenv("STF_LOG_DIR", "./data/STF_LOG_DIR")
    args.use_gpu = False
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
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
        print('Making Predictions...')
        # Custom DataLoader for 'stocks'
        args.null_value = None # NULL_VAL
        args.pad_value = None
        # oos_loader = DataLoader(TimeSeriesDataset_ContextOnly(data_folder='spacetimeformer/data/oos', 
        #                                                       context_length=args.context_points, 
        #                                                       forecast_length=args.target_points),
        #                                                       batch_size=args.batch_size, 
        #                                                       shuffle=False, num_workers=4)
        folder='spacetimeformer/data/oos'
        xt_holder = []  # Initialize xt_holder as an empty list to hold tensors
        for filename in os.listdir(folder):
            if filename.endswith('.csv'):  # Check if the file ends with '.csv'
                filepath = os.path.join(folder, filename)
                dataset = TimeSeriesDataset_ContextOnly(folder_name=folder, file_name=filename, context_length=args.context_points)
                dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
                for batch_idx, context in enumerate(dataloader):
                    x_t = context[:, -args.context_points:, :]
                    xt_holder.append(x_t[-1,:,:])

        # Ensure torch.stack() is called outside the loop, after xt_holder has collected all tensors
        xt_holder = torch.stack(xt_holder, dim=0)
        print('Eval Dataset Shape: ', xt_holder.shape)

    # Model Training and Evaluation
    forecaster = create_model(args)
    forecaster = forecaster.to(device)  # Move the model to the specified device

# Current best
    # output_path = "/Users/alecjeffery/Documents/Playgrounds/Python/largeModels/mar29_2024_p57_v71.pth"
# 4/22
    # output_path = "/Users/alecjeffery/Documents/Playgrounds/Python/largeModels/V75P60_4_25.pth"
# Wrapped Model
    # output_path = "/Users/alecjeffery/Documents/Playgrounds/Python/largeModels/WrappedAttn_sV_81_LV_86_sp_67_lp_62_Jul9.pth"
# 5/23 *** Preferred
    # output_path = "/Users/alecjeffery/Documents/Playgrounds/Python/largeModels/72V_61P_5_23_2024.pth"
# 5/25
    # output_path = "/Users/alecjeffery/Documents/Playgrounds/Python/largeModels/V60P60_v-0.65-0.36p-0.1-0.24.pth"
# 10/13
    output_path = "/Users/alecjeffery/Documents/Playgrounds/Python/largeModels/HighAccuracy_Oct13th.pth"
    # Load the weights into the model
    # forecaster.load_state_dict(torch.load(output_path))
    forecaster.load_state_dict(torch.load(output_path, map_location=torch.device('cpu')))

    # stock_names = [i[:-4] for i in os.listdir(folder)]  # Extract stock names from filenames
    stock_names = [filename[:-4] for filename in os.listdir(folder) if filename.endswith('.csv')]# To deal with the .DS_Store file issue

    print('STOCK NAMED',stock_names)
    if args.dset == "stocks":
        forecaster.eval()
        with torch.no_grad():
            x_c = xt_holder[:, args.target_points:, :]
            y_c = xt_holder[:, args.target_points:, [3, 4]]
            x_t = xt_holder[:, -args.target_points:, :]  # Assuming x_t is used for prediction
            y_t = xt_holder[:, -args.target_points:, [3, 4]]

            x_c, y_c, x_t, y_t = x_c.to(device), y_c.to(device), x_t.to(device), y_t.to(device)
            model_output = forecaster(x_c, y_c, x_t, y_t)
            
            predictions = model_output[0] if isinstance(model_output, tuple) else model_output
            predictions = predictions.cpu().detach().numpy()  # Move to CPU and convert to numpy

            # Separate 'close' and 'volatility' values
            close_values = predictions[:, :, 0]  # Assuming 'close' values are the first in the last dimension
            volatility_values = predictions[:, :, 1]  # Assuming 'volatility' values are the second

            # Flatten 'close' and 'volatility' arrays
            close_flattened = close_values.reshape(predictions.shape[0], -1)
            volatility_flattened = volatility_values.reshape(predictions.shape[0], -1)

            # Concatenate the flattened 'close' and 'volatility' arrays horizontally
            predictions_flattened = np.hstack((close_flattened, volatility_flattened))

            # Create column names for the DataFrame
            close_columns = [f'Close_{i+1}' for i in range(close_values.shape[1])]
            volatility_columns = [f'Volatility_{i+1}' for i in range(volatility_values.shape[1])]
            column_names = close_columns + volatility_columns

            # Assuming each sample's predictions are now correctly ordered and flattened
            if len(predictions_flattened) == len(stock_names):
                # Create the DataFrame with the reshaped predictions
                predictions_df = pd.DataFrame(predictions_flattened, columns=column_names, index=stock_names)
                # Save to CSV with stock names as row indices, timestamped
                predictions_df.to_csv('oos_predictions.csv')
                formatOutput(tops=5)
                # current_date = datetime.datetime.now()
                # formatted_date = f"{current_date.month}_{current_date.day}_{current_date.year}"
                # predictions_df.to_csv('oos_predictions_'+formatted_date+'.csv')
            else:
                print("Mismatch between the number of predictions and the number of stock names.")

    # WANDB Experiment Finish (if applicable)
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
