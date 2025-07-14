import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR
import polars as pl
import numpy as np
import scipy
import os
import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
import pandas as pd
from torch.utils.data import DataLoader
from enformer_pytorch.data import GenomeIntervalDataset

from scooby.modeling import Scooby
from scooby.utils.utils import poisson_multinomial_torch, evaluate, fix_rev_comp_rna, read_backed, add_weight_decay, get_lora
from scooby.data import onTheFlyDataset
import scanpy as sc
import h5py
import yaml

def train(config):
    """
    Trains the scooby model.

    Args:
        config (dict): Dictionary containing the training configuration.
    """
    # Set up accelerator
    ddp_kwargs = DistributedDataParallelKwargs(static_graph=True)
    local_world_size = 1
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs], step_scheduler_with_optimizer=False)

    # Extract configuration parameters
    output_dir = config["output_dir"]
    run_name = config["run_name"]
    
    rna_plus = config["data"]["rna_plus_path"]
    rna_minus = config["data"]["rna_minus_path"]
    embedding_path = config["data"]["embedding_path"]
    neighbors_path = config["data"]["neighbors_path"]
    sequences_path = config["data"]["sequences_path"]
    genome_path = config["data"]["genome_path"]
    
    cell_emb_dim = config["model"]["cell_emb_dim"]
    num_tracks = config["model"]["num_tracks"]
    batch_size = config["training"]["batch_size"]
    lr = float(config["training"]["lr"])
    wd = float(config["training"]["wd"])
    clip_global_norm = config["training"]["clip_global_norm"]
    warmup_steps = config["training"]["warmup_steps"] * local_world_size
    num_epochs = config["training"]["num_epochs"] * local_world_size
    eval_every_n = config["training"]["eval_every_n"]
    total_weight = config["training"]["total_weight"]
    test_fold = config["data"]["test_fold"]
    val_fold = config["data"]["val_fold"]
    context_length = config["data"]["context_length"]
    shift_augs = config["data"]["shift_augs"]
    rc_aug = config["data"]["rc_aug"]

    device = accelerator.device

    # Load data
    adatas = {
        "rna_plus": read_backed(h5py.File(rna_plus), "fragment_single"),
        "rna_minus": read_backed(h5py.File(rna_minus), "fragment_single"),
    }

    # we have an option to train with targets pseudobulked across neighbors, but we train without neighbors, true single cell
    neighbors = scipy.sparse.load_npz(neighbors_path)
    embedding = pd.read_parquet(embedding_path)

    # cell weights can be used to put more weight on some cells or just ignore some cell types altogether
    # cell_weights = np.load(f"{data_path}scooby_training_data/cell_weights_no_normoblast.npy")

    # Calculate training steps
    num_steps = (45_000 * num_epochs) // (batch_size)
    accelerator.print("Will be training for ", num_steps)

    # Initialize model, optimizer, and scheduler
    scooby = Scooby.from_pretrained(
        config["model"]["pretrained_model"],
        cell_emb_dim=cell_emb_dim,
        embedding_dim=1920,
        n_tracks=num_tracks,
        return_center_bins_only=True,
        disable_cache=True,
        use_transform_borzoi_emb=False,
    )
    scooby = get_lora(scooby, train=True)
    parameters = add_weight_decay(scooby, lr = lr, weight_decay = wd)
    optimizer = torch.optim.AdamW(parameters)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.0000001, total_iters=warmup_steps, verbose=False)
    train_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.00, total_iters=num_steps - warmup_steps, verbose=False)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, train_scheduler], [warmup_steps])

    # Create datasets and dataloaders
    filter_train = lambda df: df.filter((pl.col("column_4") != f"fold{test_fold}") & (pl.col("column_4") != f"fold{val_fold}"))
    ds = GenomeIntervalDataset(
        bed_file=sequences_path,
        fasta_file=genome_path,
        filter_df_fn=filter_train,
        return_seq_indices=False,
        shift_augs=shift_augs,
        rc_aug=rc_aug,
        return_augs=True,
        context_length=context_length,
        chr_bed_to_fasta_map={},
    )

    filter_val = lambda df: df.filter((pl.col("column_4") == f"fold{val_fold}"))
    val_ds = GenomeIntervalDataset(
        bed_file=sequences_path,
        fasta_file=genome_path,
        filter_df_fn=filter_val,
        return_seq_indices=False,
        shift_augs=(0, 0),
        rc_aug=False,
        return_augs=True,
        context_length=context_length,
        chr_bed_to_fasta_map={},
    )

    accelerator.print(len(val_ds), val_fold, test_fold)

    otf_dataset = onTheFlyDataset(
        adatas['rna_plus'],
        adatas['rna_minus'],
        neighbors=neighbors,
        embedding=embedding,
        ds=ds,
        cell_sample_size=64,
        cell_weights=None,
        clip_soft=5,
    )
    val_dataset = onTheFlyDataset(
        adatas['rna_plus'],
        adatas['rna_minus'],
        neighbors=neighbors,
        embedding=embedding,
        ds=val_ds,
        cell_sample_size=32,
        cell_weights=None,
        clip_soft=5,
    )

    training_loader = DataLoader(otf_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # Prepare model, optimizer, scheduler, and dataloaders for distributed training
    scooby = nn.SyncBatchNorm.convert_sync_batchnorm(scooby)
    scooby, optimizer, scheduler, training_loader, val_loader = accelerator.prepare(
        scooby, optimizer, scheduler, training_loader, val_loader
    )

    # Initialize trackers
    accelerator.init_trackers("scooby", init_kwargs={"wandb": {"name": f"{run_name}"}})
    loss_fn = poisson_multinomial_torch

    # Training loop
    for epoch in range(num_epochs):
        for i, [inputs, rc_augs, targets, cell_emb_idx] in tqdm.tqdm(enumerate(training_loader)):
            inputs = inputs.permute(0, 2, 1).to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            for rc_aug_idx in rc_augs.nonzero():
                rc_aug_idx = rc_aug_idx[0]
                flipped_version = torch.flip(targets[rc_aug_idx].unsqueeze(0), (1, -3))
                targets[rc_aug_idx] = fix_rev_comp_rna(flipped_version)[0]
            optimizer.zero_grad()
            with torch.autocast("cuda"):
                outputs = scooby(inputs, cell_emb_idx)
                loss = loss_fn(outputs, targets, total_weight=total_weight)
                accelerator.log({"loss": loss})
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(scooby.parameters(), clip_global_norm)
            accelerator.log({"learning_rate": scheduler.get_last_lr()[0]})
            optimizer.step()
            scheduler.step()
            if i % eval_every_n == 0:
                evaluate(accelerator, scooby, val_loader, mode='rna') 
                scooby.train()
            if (i % 1000 == 0 and epoch != 0) or (i % 2000 == 0 and epoch == 0 and i != 0):
                accelerator.save_state(output_dir=f"{output_dir}/scooby_epoch_{epoch}_{i}_{run_name}")
    accelerator.save_state(output_dir=f"{output_dir}/scooby_final_{run_name}")
    accelerator.end_training()

if __name__ == "__main__":
    # Load configuration from YAML file
    with open("config_rna_only.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Train the model
    train(config)
