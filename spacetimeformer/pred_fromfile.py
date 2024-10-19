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

    if len(sys.argv) > 3 and sys.argv[3] == "-h":
        parser.print_help()
        sys.exit(0)

    return parser

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
        
        # Use command-line arguments for d_model and d_ff
        d_model = getattr(config, 'd_model', 720)  # Default to 720 if not provided
        d_ff = getattr(config, 'd_ff', 2880)  # Default to 2880 if not provided
        d_qk = d_model // config.n_heads
        d_v = d_model // config.n_heads

        forecaster = stf.spacetimeformer_model.Spacetimeformer_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            max_seq_len=max_seq_len,
            start_token_len=getattr(config, 'start_token_len', 0),
            attn_factor=getattr(config, 'attn_factor', 1),
            d_model=d_model,
            d_queries_keys=d_qk,
            d_values=d_v,
            n_heads=config.n_heads,
            e_layers=config.enc_layers,
            d_layers=config.dec_layers,
            d_ff=d_ff,
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
        )
    return forecaster

def formatOutput(tops):
    a = pd.read_csv('oos_predictions.csv', index_col=0)
    b = pd.read_csv('./spacetimeformer/data/TixMuSig.csv', index_col=1)

    col = ['Close_'+str(i) for i in range(1,11)]
    Vcol = ['Volatility_'+str(i) for i in range(1,11)]
    for i in a.index:
        a.loc[i][col] = a.loc[i][col]*b.loc[i].closesig + b.loc[i].closemu
        a.loc[i][Vcol] = a.loc[i][Vcol]*b.loc[i].volsig + b.loc[i].volmu

    current_date = datetime.datetime.now()
    formatted_date = f"{current_date.month}_{current_date.day}_{current_date.year}"
    a.to_csv('oos_predictions_'+formatted_date+'.csv')

    a = pd.read_csv('oos_predictions.csv', index_col=0)

    a['Price_PrctDelta'] = a['Close_10']-a['Close_1']
    a['Volatility_PrctDelta'] = a['Volatility_10']-a['Volatility_1']

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
        args.null_value = None
        args.pad_value = None
        folder='spacetimeformer/data/oos'
        xt_holder = []
        for filename in os.listdir(folder):
            if filename.endswith('.csv'):
                filepath = os.path.join(folder, filename)
                dataset = TimeSeriesDataset_ContextOnly(folder_name=folder, file_name=filename, context_length=args.context_points)
                dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
                for batch_idx, context in enumerate(dataloader):
                    x_t = context[:, -args.context_points:, :]
                    xt_holder.append(x_t[-1,:,:])

        xt_holder = torch.stack(xt_holder, dim=0)
        print('Eval Dataset Shape: ', xt_holder.shape)

    # Model Training and Evaluation
    forecaster = create_model(args)
    forecaster = forecaster.to(device)

    output_path = "/Users/alecjeffery/Documents/Playgrounds/Python/largeModels/HighAccuracy_Oct13th.pth"
    
    state_dict = torch.load(output_path, map_location=torch.device('cpu'))
    forecaster.load_state_dict(state_dict)

    stock_names = [filename[:-4] for filename in os.listdir(folder) if filename.endswith('.csv')]

    print('STOCK NAMED',stock_names)
    if args.dset == "stocks":
        forecaster.eval()
        with torch.no_grad():
            x_c = xt_holder[:, args.target_points:, :]
            y_c = xt_holder[:, args.target_points:, [3, 4]]
            x_t = xt_holder[:, -args.target_points:, :]
            y_t = xt_holder[:, -args.target_points:, [3, 4]]

            x_c, y_c, x_t, y_t = x_c.to(device), y_c.to(device), x_t.to(device), y_t.to(device)
            model_output = forecaster(x_c, y_c, x_t, y_t)
            
            predictions = model_output[0] if isinstance(model_output, tuple) else model_output
            predictions = predictions.cpu().detach().numpy()

            close_values = predictions[:, :, 0]
            volatility_values = predictions[:, :, 1]

            close_flattened = close_values.reshape(predictions.shape[0], -1)
            volatility_flattened = volatility_values.reshape(predictions.shape[0], -1)

            predictions_flattened = np.hstack((close_flattened, volatility_flattened))

            close_columns = [f'Close_{i+1}' for i in range(close_values.shape[1])]
            volatility_columns = [f'Volatility_{i+1}' for i in range(volatility_values.shape[1])]
            column_names = close_columns + volatility_columns

            if len(predictions_flattened) == len(stock_names):
                predictions_df = pd.DataFrame(predictions_flattened, columns=column_names, index=stock_names)
                predictions_df.to_csv('oos_predictions.csv')
                formatOutput(tops=5)
            else:
                print("Mismatch between the number of predictions and the number of stock names.")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
