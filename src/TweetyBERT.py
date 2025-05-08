#!/usr/bin/env python3

import argparse
import os
import argparse
import torch

from experiment_manager import ExperimentRunner

def main(args):
    # Determine device priority: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    experiment_runner = ExperimentRunner(device=device)
    config = {
        "experiment_name": args.experiment_name,
        "continue_training": args.continue_training,
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "batch_size": args.batch_size,
        "d_transformer": args.d_transformer,
        "nhead_transformer": args.nhead_transformer,
        "num_freq_bins": args.num_freq_bins,
        "dropout": args.dropout,
        "dim_feedforward": args.dim_feedforward,
        "transformer_layers": args.transformer_layers,
        "m": args.m,
        "p": args.p,
        "alpha": args.alpha,
        "pos_enc_type": args.pos_enc_type,
        "pitch_shift": args.pitch_shift,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "context": args.context,
        "weight_decay": args.weight_decay,
        "early_stopping": args.early_stopping,
        "patience": args.patience,
        "trailing_avg_window": args.trailing_avg_window,
        "num_ground_truth_labels": args.num_ground_truth_labels
    }
    experiment_runner.run_experiment(config, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TweetyBERT experiment")
    parser.add_argument("--experiment_name", type=str, default="Default_Experiment")
    parser.add_argument("--continue_training", type=bool, default=False)
    parser.add_argument("--train_dir", type=str, default="temp/train_dir")
    parser.add_argument("--test_dir", type=str, default="temp/test_dir")
    parser.add_argument("--batch_size", type=int, default=42)
    parser.add_argument("--d_transformer", type=int, default=196)
    parser.add_argument("--nhead_transformer", type=int, default=4)
    parser.add_argument("--num_freq_bins", type=int, default=196)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dim_feedforward", type=int, default=768)
    parser.add_argument("--transformer_layers", type=int, default=4)
    parser.add_argument("--m", type=int, default=250)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--pos_enc_type", type=str, default="relative")
    parser.add_argument("--pitch_shift", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=float, default=3e4)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--context", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--early_stopping", type=bool, default=True)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--trailing_avg_window", type=int, default=200)
    parser.add_argument("--num_ground_truth_labels", type=int, default=50)

    args = parser.parse_args()
    main(args)
