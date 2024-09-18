# TODO: improve usability

if __name__ == '__main__':
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    import numpy as np
    import torch.nn as nn
    from torch.optim.lr_scheduler import SequentialLR, LinearLR
    import polars as pl
    import scipy
    import os
    import tqdm
    from accelerate import Accelerator, DistributedDataParallelKwargs
    import pandas as pd
    from peft import LoraConfig
    from torch.utils.data import DataLoader
    from peft import get_peft_model
    from polya_project.data import GenomeIntervalDataset
    from modeling.scborzoi import ScBorzoi
    from utils.utils import poisson_multinomial_torch, evaluate, fix_rev_comp_multiome
    from data.scdata import onTheFlyMultiomeDataset
    import anndata as ad
    import scanpy as sc
    import h5py
    from anndata.experimental import read_elem, sparse_dataset

    ddp_kwargs = DistributedDataParallelKwargs(static_graph = True)
    local_world_size = 1

    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs], step_scheduler_with_optimizer = False)

    ### TODO
    output_dir = '' 
    run_name = "scooby"
    fasta_file = 'hg38/genome_human.fa'
    bed_file = 'hg38/sequences.bed'
    data_path = '/s/project/QNA/scborzoi/submission_data/'
    borzoi_weight_path = 'path_to_borzoi_weights' #converted using borzoi-pytorch
    cell_emb_dim = 14
    num_tracks = 3

    #Dataset params
    fold = 0
    context_length = 524288
    shift_augs = (-3, 3)
    test_fold = 3
    val_fold = 4
    rc_aug = True

    #Train params
    batch_size = 1
    lr = 1e-4
    wd = 1e-6
    clip_global_norm = 1.0
    warmup_steps = 1000* local_world_size
    num_epochs = 4*local_world_size
    eval_every_n = 256
    total_weight = 0.2
    device = accelerator.device
    
    min_value = torch.finfo(torch.float16).min
    max_value = torch.finfo(torch.float16).max
    
    def read_backed(group, key):
        return ad.AnnData(
            sparse_dataset(group["X"]),
            obsm={'fragment_single': sparse_dataset(group["obsm"][key])},
            **{
                k: read_elem(group[k]) if k in group else {}
                for k in ["layers", "obs", "var", "varm", "uns", "obsp", "varp"]
            }
        )
    
    adatas = {
        'rna_plus': read_backed(h5py.File(os.path.join(data_path, 'snapatac_merged_fixed_plus.h5ad')), 'fragment_single'),
        'rna_minus': read_backed(h5py.File(os.path.join(data_path, 'snapatac_merged_fixed_minus.h5ad')), 'fragment_single'),
        'atac': sc.read(os.path.join(data_path, 'snapatac_merged_fixed_atac.h5ad')),
        }
    
    neighbors = scipy.sparse.load_npz(f'{data_path}borzoi_training_data/no_neighbors.npz')
    embedding = pd.read_parquet(f'{data_path}borzoi_training_data/embedding_no_val_genes_new.pq')
    cell_weights = np.load(f'{data_path}borzoi_training_data/cell_weights_no_normoblast.npy')

    num_steps = (45_000 * num_epochs)//(batch_size)
    accelerator.print ("will be training for ", num_steps)

    csb = ScBorzoi(cell_emb_dim = cell_emb_dim, embedding_dim = 1920,n_tracks = num_tracks,return_center_bins_only = True, disable_cache = True, use_transform_borzoi_emb = True)
    old_weights = torch.load(borzoi_weight_path)
    csb.load_state_dict(old_weights, strict = False)

    config = LoraConfig(
        target_modules=r"(?!separable\d+).*conv_layer|.*to_q|.*to_v|transformer\.\d+\.1\.fn\.1|transformer\.\d+\.1\.fn\.4",
    )
    csb = get_peft_model(csb, config) # get LoRA model
    csb.print_trainable_parameters()

    for params in csb.parameters():
       params.requires_grad = False
    for params in csb.base_model.cell_state_to_conv.parameters():
       params.requires_grad = True
    for params in csb.base_model.transform_borzoi_emb.parameters():
       params.requires_grad = True
        
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 

    accelerator.print ("CSB, params: ", count_parameters(csb))

    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        high_lr = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name in skip_list:
                no_decay.append(param)
            elif "cell_state_to_conv" in name:
                high_lr.append(param)
                #accelerator.print ("setting to highlr", name)
            else:
                decay.append(param)
        return [{'params': high_lr, 'weight_decay': weight_decay, 'lr' : 4e-4}, {'params': no_decay, 'weight_decay': 0., 'lr' : lr}, {'params': decay, 'weight_decay': weight_decay, 'lr' : lr}]

    parameters = add_weight_decay(csb, weight_decay = wd)

    optimizer = torch.optim.AdamW(parameters)

    warmup_scheduler = LinearLR(optimizer, start_factor = 0.0000001, total_iters = warmup_steps, verbose = False)
    train_scheduler = LinearLR(optimizer, start_factor = 1.0, end_factor = 0.00, total_iters = num_steps - warmup_steps, verbose = False)

    scheduler = SequentialLR(optimizer, [warmup_scheduler, train_scheduler], [warmup_steps]) 

    filter_train = lambda df: df.filter((pl.col('column_4') != f'fold{test_fold}') & (pl.col('column_4') != f'fold{val_fold}'))

    ds = GenomeIntervalDataset(
        bed_file = bed_file,       
        fasta_file = fasta_file,                    
        filter_df_fn = filter_train,                       
        return_seq_indices = False,                          
        shift_augs = shift_augs,                              
        rc_aug = rc_aug,
        return_augs = True,
        context_length = context_length,
        chr_bed_to_fasta_map = {}
    )
    
    
    filter_val = lambda df: df.filter((pl.col('column_4') == f'fold{val_fold}') )
    val_ds = GenomeIntervalDataset(
        bed_file = bed_file,                       
        fasta_file = fasta_file,                        
        filter_df_fn = filter_val,                       
        return_seq_indices = False,                         
        shift_augs = (0,0),                              
        rc_aug = False,
        return_augs = True,
        context_length = context_length,
        chr_bed_to_fasta_map = {}
    )

    accelerator.print (len(val_ds), val_fold, test_fold)

    otf_dataset= onTheFlyMultiomeDataset( adatas=adatas,neighbors=neighbors, embedding=embedding, ds = ds, cell_sample_size = 64, cell_weights = None, normalize_atac = True, clip_soft = 5)
    val_dataset = onTheFlyMultiomeDataset( adatas=adatas,neighbors=neighbors, embedding=embedding, ds = val_ds, cell_sample_size = 32, cell_weights = None, normalize_atac = True, clip_soft = 5)

    training_loader = DataLoader(otf_dataset, batch_size=batch_size, shuffle = True, num_workers = 8)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle = False, num_workers = 1, pin_memory= True)


    csb = nn.SyncBatchNorm.convert_sync_batchnorm(csb) 

    csb, optimizer, scheduler, training_loader, val_loader = accelerator.prepare(
            csb, optimizer, scheduler, training_loader, val_loader
    )

    accelerator.init_trackers("scooby", init_kwargs={"wandb":{"name":f"{run_name}",}})
    loss_fn = poisson_multinomial_torch

    for epoch in range(40):
        for i, [inputs, rc_augs, targets, cell_emb_idx]  in tqdm.tqdm(enumerate(training_loader)):
            inputs = inputs.permute(0,2,1).to(device, non_blocking=True) 
            targets = targets.to(device, non_blocking=True)
            for rc_aug_idx in rc_augs.nonzero():
                rc_aug_idx = rc_aug_idx[0]
                flipped_version = torch.flip(targets[rc_aug_idx].unsqueeze(0),(1,-3))
                targets[rc_aug_idx] = fix_rev_comp_multiome(flipped_version)[0]
            optimizer.zero_grad()
            with torch.autocast("cuda"):        
                outputs = csb(inputs, cell_emb_idx)
                loss = loss_fn(outputs, targets, total_weight = total_weight)
                accelerator.log({"loss":loss})
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(csb.parameters(), clip_global_norm)
            accelerator.log({"learning_rate" : scheduler.get_last_lr()[0]})
            optimizer.step()
            scheduler.step()
            if i % eval_every_n == 0:
                evaluate(accelerator,csb,val_loader)
                csb.train()
            if (i % 1000 == 0 and epoch  != 0) or (i % 2000 == 0 and epoch  == 0 and i != 0):
                accelerator.save_state(output_dir= f'{output_dir}/csb_epoch_{epoch}_{i}_{run_name}')
    accelerator.save_state(output_dir= f'{output_dir}/csb_final_{run_name}')
    accelerator.end_training()
