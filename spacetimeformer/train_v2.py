import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from spacetimeformer import Spacetimeformer_Forecaster
from spacetimeformer.data import DataModule

def main(args):
    # Data preparation
    data_module = DataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # Add other data module parameters as needed
    )

    # Model creation
    model = Spacetimeformer_Forecaster(
        d_x=args.d_x,
        d_yc=args.d_yc,
        d_yt=args.d_yt,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        e_layers=args.enc_layers,
        d_layers=args.dec_layers,
        dropout_emb=args.dropout_emb,
        dropout_attn_out=args.dropout_attn_out,
        dropout_attn_matrix=args.dropout_attn_matrix,
        dropout_qkv=args.dropout_qkv,
        dropout_ff=args.dropout_ff,
        activation=args.activation,
        time_emb_dim=args.time_emb_dim,
        loss=args.loss,
        learning_rate=args.learning_rate,
        decay_factor=args.decay_factor,
        # Add other model parameters as needed
    )

    # Training setup
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    logger = TensorBoardLogger("tb_logs", name="spacetimeformer")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        callbacks=[checkpoint_callback],
        logger=logger,
        # Add other trainer parameters as needed
    )

    # Start training
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spacetimeformer model")
    
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    
    # Model parameters
    parser.add_argument("--d_x", type=int, required=True, help="Dimension of input features")
    parser.add_argument("--d_yc", type=int, required=True, help="Dimension of observed outputs")
    parser.add_argument("--d_yt", type=int, required=True, help="Dimension of target outputs")
    parser.add_argument("--max_seq_len", type=int, required=True, help="Maximum sequence length")
    parser.add_argument("--d_model", type=int, default=720, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=2880, help="Feedforward dimension")
    parser.add_argument("--n_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--enc_layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--dec_layers", type=int, default=3, help="Number of decoder layers")
    parser.add_argument("--dropout_emb", type=float, default=0.15, help="Embedding dropout rate")
    parser.add_argument("--dropout_attn_out", type=float, default=0.15, help="Attention output dropout rate")
    parser.add_argument("--dropout_attn_matrix", type=float, default=0.15, help="Attention matrix dropout rate")
    parser.add_argument("--dropout_qkv", type=float, default=0.15, help="Query/Key/Value dropout rate")
    parser.add_argument("--dropout_ff", type=float, default=0.15, help="Feedforward dropout rate")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function")
    parser.add_argument("--time_emb_dim", type=int, default=6, help="Time embedding dimension")
    parser.add_argument("--loss", type=str, default="mse", help="Loss function")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--decay_factor", type=float, default=0.98, help="Learning rate decay factor")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")

    args = parser.parse_args()
    main(args)
