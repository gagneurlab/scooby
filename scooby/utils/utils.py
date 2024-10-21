import scipy
import torch
import numpy as np
import tqdm
import torch.nn.functional as F
from scipy import stats
from matplotlib import pyplot as plt
import anndata as ad
from anndata.experimental import read_elem, sparse_dataset


def poisson_multinomial_torch(
    y_pred,
    y_true,
    total_weight: float = 0.2,
    epsilon: float = 1e-6,
    rescale: bool = False,
):
    """
    Calculates the Poisson-Multinomial loss.

    This loss function combines a Poisson loss term for the total count and a multinomial loss term for the 
    distribution across sequence positions.

    Args:
        y_pred (torch.Tensor): Predicted values (batch_size, seq_len).
        y_true (torch.Tensor): True values (batch_size, seq_len).
        total_weight (float, optional): Weight of the Poisson total term. Defaults to 0.2.
        epsilon (float, optional): Small value added to avoid log(0). Defaults to 1e-6.
        rescale (bool, optional): Whether to rescale the loss. Defaults to False.

    Returns:
        torch.Tensor: The mean Poisson-Multinomial loss.
    """
    seq_len = y_true.shape[1]

    # add epsilon to protect against tiny values
    y_true += epsilon
    y_pred += epsilon

    # sum across lengths
    s_true = y_true.sum(dim=1, keepdim=True)
    s_pred = y_pred.sum(dim=1, keepdim=True)

    # normalize to sum to one
    p_pred = y_pred / s_pred

    # total count poisson loss
    poisson_term = F.poisson_nll_loss(s_pred, s_true, log_input=False, eps=0, reduction="mean")  # B x T
    # print (poisson_term,poisson_term.shape)
    poisson_term /= seq_len
    # print (poisson_term)

    # multinomial loss
    pl_pred = torch.log(p_pred)  # B x L x T
    multinomial_dot = -torch.multiply(y_true, pl_pred)  # B x L x T
    multinomial_term = multinomial_dot.sum(dim=1)  # B x T
    multinomial_term /= seq_len
    # print (multinomial_term.mean(), poisson_term.mean())

    # normalize to scale of 1:1 term ratio
    loss_raw = multinomial_term + total_weight * poisson_term
    # print (loss_raw.shape)
    if rescale:
        loss_rescale = loss_raw * 2 / (1 + total_weight)
    else:
        loss_rescale = loss_raw

    return loss_rescale.mean()


def multinomial_torch(
    y_pred,
    y_true,
    total_weight: float = 0.2,
    epsilon: float = 1e-6,
    rescale: bool = False,
):
    """
    Calculates the Multinomial loss.

    This loss function measures the difference between the predicted and true distributions across sequence positions.

    Args:
        y_pred (torch.Tensor): Predicted values (batch_size, seq_len).
        y_true (torch.Tensor): True values (batch_size, seq_len).
        total_weight (float, optional): Not used in this function, but included for compatibility with other loss functions. Defaults to 0.2.
        epsilon (float, optional): Small value added to avoid log(0). Defaults to 1e-6.
        rescale (bool, optional): Whether to rescale the loss. Defaults to False.

    Returns:
        torch.Tensor: The mean Multinomial loss.
    """
    seq_len = y_true.shape[1]

    # add epsilon to protect against tiny values
    y_true += epsilon
    y_pred += epsilon

    # sum across lengths
    s_pred = y_pred.sum(dim=1, keepdim=True)

    # normalize to sum to one
    p_pred = y_pred / s_pred

    # multinomial loss
    pl_pred = torch.log(p_pred)  # B x L x T
    multinomial_dot = -torch.multiply(y_true, pl_pred)  # B x L x T
    multinomial_term = multinomial_dot.sum(dim=1)  # B x T
    multinomial_term /= seq_len
    # print (multinomial_term.mean(), poisson_term.mean())

    # normalize to scale of 1:1 term ratio
    loss_raw = multinomial_term
    # print (loss_raw.shape)
    if rescale:
        loss_rescale = loss_raw * 2 / (1 + total_weight)
    else:
        loss_rescale = loss_raw

    return loss_rescale.mean()


def poisson_torch(
    y_pred,
    y_true,
    total_weight: float = 0.2,
    epsilon: float = 1e-6,
    rescale: bool = False,
):
    """
    Calculates the Poisson loss.

    This loss function measures the difference between the predicted and true total counts.

    Args:
        y_pred (torch.Tensor): Predicted values (batch_size, seq_len).
        y_true (torch.Tensor): True values (batch_size, seq_len).
        total_weight (float, optional): Not used in this function, but included for compatibility with other loss functions. Defaults to 0.2.
        epsilon (float, optional): Small value added to avoid log(0). Defaults to 1e-6.
        rescale (bool, optional): Whether to rescale the loss. Defaults to False.

    Returns:
        torch.Tensor: The mean Poisson loss.
    """
    seq_len = y_true.shape[1]

    # add epsilon to protect against tiny values
    y_true += epsilon
    y_pred += epsilon

    # sum across lengths
    s_true = y_true.sum(dim=1, keepdim=True)
    s_pred = y_pred.sum(dim=1, keepdim=True)

    # total count poisson loss
    poisson_term = F.poisson_nll_loss(s_pred, s_true, log_input=False, eps=0, reduction="mean")  # B x T
    # print (poisson_term,poisson_term.shape)
    poisson_term /= seq_len
    # print (poisson_term)
    loss_raw = poisson_term
    # print (loss_raw.shape)
    if rescale:
        loss_rescale = loss_raw * 2 / (1 + total_weight)
    else:
        loss_rescale = loss_raw

    return loss_rescale.mean()


def fix_rev_comp_multiome(outputs_rev_comp):
    """
    Reverses the order of elements in the output tensor for the reverse complement sequence in the multiome model.

    This function rearranges the elements of the output tensor to match the correct order for the reverse 
    complement sequence, ensuring that the RNA and ATAC tracks are in the expected positions.

    Args:
        outputs_rev_comp (torch.Tensor): The output tensor for the reverse complement sequence.

    Returns:
        torch.Tensor: The rearranged output tensor.
    """
    num_pos = outputs_rev_comp.shape[2] // 3
    fix_indices_tensor = (torch.arange(0, num_pos * 3, step=3, dtype=int).repeat_interleave(3)) + torch.tensor(
        [1, 0, 2]
    ).repeat(num_pos)
    test_out = torch.empty_like(outputs_rev_comp)
    test_out[:, :, fix_indices_tensor] = outputs_rev_comp
    return test_out


def fix_rev_comp2(outputs_rev_comp):
    """
    Reverses the order of elements in the output tensor for the reverse complement sequence (for RNA-only predictions).

    This function rearranges the elements of the output tensor to match the correct order for the reverse 
    complement sequence for RNA only predictions.

    Args:
        outputs_rev_comp (torch.Tensor): The output tensor for the reverse complement sequence.

    Returns:
        torch.Tensor: The rearranged output tensor.
    """
    num_pos = outputs_rev_comp.shape[2]
    test = torch.arange(0, num_pos).unsqueeze(0)
    fix_indices_tensor = torch.zeros(1, num_pos)
    fix_indices_tensor[:, :num_pos:2] = test[:, 1:num_pos:2]
    fix_indices_tensor[:, 1:num_pos:2] = test[:, :num_pos:2]
    fix_indices_tensor = fix_indices_tensor.squeeze().int()
    test_out = torch.empty_like(outputs_rev_comp)
    test_out[:, :, fix_indices_tensor] = outputs_rev_comp
    return test_out


def evaluate(accelerator, csb, val_loader):
    """
    Evaluates the model on the validation set.

    This function performs inference on the validation set, calculates the Pearson correlation between the predicted
    and true profiles, and logs the results using the accelerator.

    Args:
        accelerator: The accelerator object (e.g., from Hugging Face Accelerate).
        csb (torch.nn.Module): The model.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
    """
    device = accelerator.device
    csb.eval()
    output_list, target_list, pearsons_per_track = [], [], []

    stop_idx = 1

    for i, [inputs, rc_augs, targets,_,  cell_emb_idx] in tqdm.tqdm(enumerate(val_loader)):
        if i < (stop_idx):
            continue
        if i == (stop_idx + 1):
            break
        inputs = inputs.permute(0, 2, 1).to(device, non_blocking=True)
        target_list.append(targets.to(device, non_blocking=True))
        with torch.no_grad():
            with torch.autocast("cuda"):
                output_list.append(csb(inputs, cell_emb_idx = cell_emb_idx).detach())
        break
    targets = torch.vstack(target_list).squeeze().numpy(force=True)  # [reindex].flatten(0,1).numpy(force =True)
    outputs = torch.vstack(output_list).squeeze().numpy(force=True)  # [reindex].flatten(0,1).numpy(force =True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Create a 1x2 subplot grid

    # accelerator.print (outputs.shape)
    for x in range(0, 96):
        pearsons_per_track.append(stats.pearsonr(outputs.T[x].flatten(), targets.T[x].flatten())[0])

    accelerator.log({"val_rnaseq_across_tracks_pearson_r": np.nanmean(pearsons_per_track)})
    accelerator.log({"val_pearson_r": stats.pearsonr(outputs.flatten(), targets.flatten())[0]})

    # Plot 'outputs' in the first subplot
    axes[0].imshow(outputs.T, vmax=1, aspect="auto")
    axes[0].set_title("Outputs")  # You can add a title if desired

    # Plot 'targets' in the second subplot
    axes[1].imshow(targets.T, vmax=1, aspect="auto")
    axes[1].set_title("Targets")  # You can add a title if desired
    # plt.show()
    accelerator.log({"val_sample_viz": fig})

    fig.clf()
    plt.close()



def undo_squashed_scale(x, clip_soft=384, track_transform=3 / 4):
    """
    Reverses the squashed scaling transformation applied to the output profiles.

    Args:
        x (torch.Tensor): The input tensor to be unsquashed.
        clip_soft (float, optional): The soft clipping value. Defaults to 384.
        track_transform (float, optional): The transformation factor. Defaults to 3/4.

    Returns:
        torch.Tensor: The unsquashed tensor.
    """
    x = x.clone()  # IMPORTANT BECAUSE OF IMPLACE OPERATIONS TO FOLLOW?

    # undo soft_clip
    unclip_mask = x > clip_soft
    # somewhat expensive to always index masked, so we only do it when it is necessary somewhere
    # worst case the following seems a bit slower, but in many cases there is no masking so we can skip
    if unclip_mask.any():
        x[unclip_mask] = (x[unclip_mask] - clip_soft) ** 2 + clip_soft

    # undo sqrt
    x = (x + 1) ** (1.0 / track_transform) - 1
    return x


def squashed_scale(x, clip_soft=384, clip=768, track_transform=3 / 4):
    """
    Applies the squashed scaling transformation to the input tensor.

    Args:
        x (torch.Tensor): The input tensor to be squashed.
        clip_soft (float, optional): The soft clipping value. Defaults to 384.
        clip (float, optional): The hard clipping value. Defaults to 768.
        track_transform (float, optional): The transformation factor. Defaults to 3/4.

    Returns:
        torch.Tensor: The squashed tensor.
    """
    x = x.clone()  # IMPORTANT BECAUSE OF IMPLACE OPERATIONS TO FOLLOW
    # undo soft_clip
    seq_cov = -1 + (1 + x) ** track_transform

    clip_mask = seq_cov > clip_soft
    seq_cov[clip_mask] = clip_soft - 1 + torch.sqrt(seq_cov[clip_mask] - clip_soft + 1)
    seq_cov = torch.clip(seq_cov, -clip, clip)
    return seq_cov


def get_gene_slice_and_strand(transcriptome, gene, position, span, sliced=True):
    """
    Retrieves the gene slice and strand information from the transcriptome.

    Args:
        transcriptome: The transcriptome object.
        gene (str): The name of the gene.
        position (int): The genomic position.
        span (int): The span of the genomic region.
        sliced (bool, optional): Whether to slice the output. Defaults to True.

    Returns:
        Tuple[torch.Tensor, str]: The gene slice and strand.
    """
    gene_slice = transcriptome.genes[gene].output_slice(
        position, 6144 * 32, 32, span=span, sliced=sliced
    )  # select right columns
    strand = transcriptome.genes[gene].strand
    return gene_slice, strand


def process_rna(outputs, strand, clip_soft, num_neighbors):
    """
    Processes the RNA output of the model.

    This function applies unsquashing and normalization to the RNA output.

    Args:
        outputs (torch.Tensor): The RNA output of the model.
        strand (str): The strand of the gene.
        clip_soft (float): The soft clipping value.
        num_neighbors (int): The number of neighbors.

    Returns:
        torch.Tensor: The processed RNA output.
    """
    num_pos = outputs.shape[-1]
    if strand == "+":
        return undo_squashed_scale(outputs[0, :, :num_pos:2], clip_soft=clip_soft) * (1 / num_neighbors)
    elif strand == "-":
        return undo_squashed_scale(outputs[0, :, 1:num_pos:2], clip_soft=clip_soft) * (1 / num_neighbors)


def process_atac(outputs, num_neighbors):
    """
    Processes the ATAC output of the model.

    This function applies normalization to the ATAC output.

    Args:
        outputs (torch.Tensor): The ATAC output of the model.
        num_neighbors (int): The number of neighbors.

    Returns:
        torch.Tensor: The processed ATAC output.
    """
    return outputs[0] * 20 * (1 / num_neighbors)


def get_outputs(csb, seqs, gene_slice, region_slice, predict, model_type, conv_weight, conv_bias):
    """
    Gets the outputs of the model for the given sequences, gene slice, and region slice.

    Args:
        csb (torch.nn.Module): The model.
        seqs (torch.Tensor): The input sequences.
        gene_slice (torch.Tensor): The bins to predict for RNA.
        region_slice (torch.Tensor): The bins to predict for ATAC.
        predict (callable): The prediction function.
        model_type (str): The type of the model.
        conv_weight (torch.Tensor): The convolutional weights.
        conv_bias (torch.Tensor): The convolutional biases.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The RNA and ATAC outputs.
    """
    seqs_rev_comp = torch.flip(seqs.permute(0, 2, 1), (-1, -2)).permute(0, 2, 1)
    if model_type == "multiome":
        outputs = predict(csb, seqs, seqs_rev_comp, conv_weight, conv_bias, bins_to_predict=None)
        outputs = outputs.float().detach()
        # rna
        outputs_rna = outputs[:, :, torch.tensor([1, 1, 0]).repeat(outputs.shape[2] // 3).bool()][:, gene_slice]
        # atac
        outputs_atac = outputs[:, :, torch.tensor([0, 0, 1]).repeat(outputs.shape[2] // 3).bool()]
        outputs_atac = [outputs_atac[:, region] for region in region_slice]
        return outputs_rna, outputs_atac
    elif model_type == "multiome_atac":
        outputs_atac = []
        for region in region_slice:
            outputs = predict(csb, seqs, seqs_rev_comp, conv_weight, conv_bias, bins_to_predict=region)
            outputs = outputs.float().detach()
            outputs_atac.append(outputs[:, :, torch.tensor([0, 0, 1]).repeat(outputs.shape[2] // 3).bool()])
        return None, outputs_atac
    elif model_type == "multiome_rna":
        outputs = predict(csb, seqs, seqs_rev_comp, conv_weight, conv_bias, bins_to_predict=gene_slice)
        outputs = outputs.float().detach()
        outputs_rna = outputs[:, :, torch.tensor([1, 1, 0]).repeat(outputs.shape[2] // 3).bool()]
        return outputs_rna, None
    elif model_type == "rna":
        outputs = predict(csb, seqs, seqs_rev_comp, conv_weight, conv_bias, bins_to_predict=gene_slice)
        outputs = outputs.float().detach()
        outputs_rna = outputs
        return outputs_rna, None


def get_pseudobulk_count_pred(
    csb, seqs, cell_emb_conv_weights_and_biases, gene_slice, strand, predict, clip_soft, model_type, num_neighbors=1
):
    """
    Calculates the predicted pseudobulk count for a given gene.

    This function predicts the gene expression for each cell in a cell type using the provided cell embeddings,
    sums the unsquashed predictions over all cells, and returns the total count.

    Args:
        csb (torch.nn.Module): The model.
        seqs (torch.Tensor): The input sequences.
        cell_emb_conv_weights_and_biases (List[Tuple[torch.Tensor, torch.Tensor]]): A list of tuples containing the convolutional weights and biases for each cell in the cell type.
        gene_slice (torch.Tensor): The bins to predict for RNA (e.g. exons).
        strand (str): The strand of the gene.
        predict (callable): The prediction function.
        clip_soft (float): The soft clipping value.
        model_type (str): The type of the model. Should be one of "multiome", "multiome_rna", "multiome_atac", or "rna".
        num_neighbors (int, optional): The number of neighbors. Defaults to 1.

    Returns:
        torch.Tensor: The predicted pseudobulk count for the gene.
    """
    seqs_rev_comp = torch.flip(seqs.permute(0, 2, 1), (-1, -2)).permute(0, 2, 1)
    stacked_outputs = []
    # go over embeddings for all cells of a cell type, sum the unsquashed predictions
    for conv_weight, conv_bias in cell_emb_conv_weights_and_biases:
        # get predictions for all cells of one cell type
        outputs = predict(csb, seqs, seqs_rev_comp, conv_weight, conv_bias, bins_to_predict=gene_slice)
        # get RNA:
        if "multiome" in model_type:
            outputs = outputs[:, :, torch.tensor([1, 1, 0]).repeat(outputs.shape[2] // 3).bool()]
        outputs = outputs.float().detach()
        #print (process_rna(outputs, strand, clip_soft, num_neighbors).shape)
        stacked_outputs.append(process_rna(outputs, strand, clip_soft, num_neighbors).sum())
        #print (stacked_outputs[-1].shape)
    return torch.stack(stacked_outputs, dim = 0)


def get_cell_count_pred(
    csb,
    seqs,
    gene_slice,
    strand,
    region_slice,
    predict,
    clip_soft,
    model_type,
    embeddings=None,
    conv_weight=None,
    conv_bias=None,
    num_neighbors=1,
    chunk_size=70000,
):
    """
    Calculates the predicted cell count for RNA, ATAC, or both.

    This function predicts the genomic profiles for each cell, aggregates them according to the model type, 
    and returns the aggregated counts.

    Args:
        csb (torch.nn.Module): The model.
        seqs (torch.Tensor): The input sequences.
        gene_slice (torch.Tensor): The bins to predict for RNA (e.g., exons).
        strand (str): The strand of the gene.
        region_slice (torch.Tensor): The bins to predict for ATAC (e.g., peaks).
        predict (callable): The prediction function.
        clip_soft (float): The soft clipping value.
        model_type (str): The type of the model. Should be one of "multiome", "multiome_rna", "multiome_atac", or "rna".
        embeddings (torch.Tensor, optional): The embeddings for all cells. Defaults to None.
        conv_weight (torch.Tensor, optional): The convolutional weights. Defaults to None.
        conv_bias (torch.Tensor, optional): The convolutional biases. Defaults to None.
        num_neighbors (int, optional): The number of neighbors. Defaults to 1.
        chunk_size (int, optional): The chunk size for splitting the embeddings. Defaults to 70000.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the predicted RNA and ATAC counts.
    """
    assert model_type in ["multiome", "multiome_rna", "multiome_atac", "rna"], "Invalid model_type"
    if model_type in ["multiome_rna", "rna"]:
        assert region_slice is None, "region_slice should be None for rna models"
    if model_type in ["multiome_atac"]:
        assert gene_slice is None, "gene_slice should be None for atac models"
    if gene_slice is None:
        gene_slice = np.arange(6144)
    if region_slice is None:
        region_slice = np.arange(6144)

    stacked_outputs_rna, stacked_outputs_atac = [], []
    # go over embeddings for all cells, sum the unsquashed predictions over the gene slices
    if embeddings is None:
        outputs_rna, outputs_atac = get_outputs(
            csb, seqs, gene_slice, region_slice, predict, model_type, conv_weight, conv_bias
        )

        if model_type in ["multiome_rna", "rna"]:
            stacked_outputs_rna.append(process_rna(outputs_rna, strand, clip_soft, num_neighbors))
        elif model_type in ["multiome_atac"]:
            stacked_outputs_atac.extend(
                [process_atac(outputs_atac_i, num_neighbors).sum(axis=0) for outputs_atac_i in outputs_atac]
            )
        else:
            stacked_outputs_rna.append(process_rna(outputs_rna, strand, clip_soft, num_neighbors))
            stacked_outputs_atac.extend(
                [process_atac(outputs_atac_i, num_neighbors).sum(axis=0) for outputs_atac_i in outputs_atac]
            )
        return {
            "rna": (torch.hstack(stacked_outputs_rna).sum(0).cpu() if len(stacked_outputs_rna) > 0 else None),
            "atac": (torch.stack(stacked_outputs_atac, axis=1).cpu() if len(stacked_outputs_atac) > 0 else None),
        }

    for embedding in torch.split(embeddings, split_size_or_sections=chunk_size, dim=0):
        conv_weight, conv_bias = csb.forward_cell_embs_only(embedding.unsqueeze(0))
        outputs_rna, outputs_atac = get_outputs(
            csb, seqs, gene_slice, region_slice, predict, model_type, conv_weight, conv_bias
        )
        if model_type in ["multiome_rna", "rna"]:
            stacked_outputs_rna.append(process_rna(outputs_rna, strand, clip_soft, num_neighbors))
        elif model_type in ["multiome_atac"]:
            stacked_outputs_atac.extend(
                torch.stack(
                    [process_atac(outputs_atac_i, num_neighbors).sum(axis=0) for outputs_atac_i in outputs_atac], axis=1
                )
            )
        else:
            stacked_outputs_rna.append(process_rna(outputs_rna, strand, clip_soft, num_neighbors))
            stacked_outputs_atac.extend(
                torch.stack(
                    [process_atac(outputs_atac_i, num_neighbors).sum(axis=0) for outputs_atac_i in outputs_atac], axis=1
                )
            )

    return {
        "rna": (torch.hstack(stacked_outputs_rna).sum(0).cpu() if len(stacked_outputs_rna) > 0 else None),
        "atac": (torch.stack(stacked_outputs_atac, axis=0).cpu() if len(stacked_outputs_atac) > 0 else None),
    }


def get_cell_profile_pred(csb, seqs, embeddings, predict, clip_soft, model_type, num_neighbors=1, chunk_size=70000):
    """
    Calculates the predicted profiles for each cell.

    This function predicts the genomic profiles for each cell, applies unsquashing and normalization, and returns the 
    processed profiles.

    Args:
        csb (torch.nn.Module): The model.
        seqs (torch.Tensor): The input sequences.
        embeddings (torch.Tensor): The embeddings for all cells.
        predict (callable): The prediction function.
        clip_soft (float): The soft clipping value.
        model_type (str): The type of the model. Should be one of "multiome", "multiome_rna", "multiome_atac", or "rna".
        num_neighbors (int, optional): The number of neighbors. Defaults to 1.
        chunk_size (int, optional): The chunk size for splitting the embeddings. Defaults to 70000.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The processed RNA profiles for the plus and minus strands, and the processed ATAC profile.
    """
    seqs_rev_comp = torch.flip(seqs.permute(0, 2, 1), (-1, -2)).permute(0, 2, 1)
    stacked_outputs_rna_plus, stacked_outputs_rna_minus, stacked_outputs_atac = [], [], []

    # go over embeddings for all cells, sum the unsquashed predictions over the gene slices
    for embedding in torch.split(embeddings, split_size_or_sections=chunk_size, dim=0):
        conv_weight, conv_bias = csb.forward_cell_embs_only(embedding.unsqueeze(0))
        # get predictions for all cells of one cell type
        outputs = predict(csb, seqs, seqs_rev_comp, conv_weight, conv_bias, bins_to_predict=None)
        outputs.float().detach()
        # get RNA:
        if "multiome" in model_type:
            outputs_rna = outputs[:, :, torch.tensor([1, 1, 0]).repeat(outputs.shape[2] // 3).bool()]
            outputs_atac = outputs[:, :, torch.tensor([0, 0, 1]).repeat(outputs.shape[2] // 3).bool()]
            outputs_atac = outputs_atac
            stacked_outputs_atac.append(outputs_atac[0])
        else:
            outputs_rna = outputs
        outputs_rna = undo_squashed_scale(outputs_rna, clip_soft=clip_soft) * (1 / num_neighbors)

        num_pos = outputs_rna.shape[-1]
        stacked_outputs_rna_plus.append(outputs_rna[0, :, :num_pos:2])
        stacked_outputs_rna_minus.append(outputs_rna[0, :, 1:num_pos:2])
    if "multiome" in model_type:
    
        return (
            torch.hstack(stacked_outputs_rna_plus).permute(1, 0),
            torch.hstack(stacked_outputs_rna_minus).permute(1, 0),
            torch.hstack(stacked_outputs_atac).permute(1, 0),
        )
    else:
        return (
            torch.hstack(stacked_outputs_rna_plus).permute(1, 0),
            torch.hstack(stacked_outputs_rna_minus).permute(1, 0),
        )


def get_pseudobulk_profile_pred(
    csb, seqs, cell_emb_conv_weights_and_biases, predict, clip_soft, model_type, num_neighbors=1
):
    """
    Calculates the predicted pseudobulk profiles for a given cell type.

    This function predicts the genomic profiles for each cell in a cell type using the provided cell embeddings,
    sums the unsquashed predictions over all cells, and returns the total profile for each genomic position.

    Args:
        csb (torch.nn.Module): The model.
        seqs (torch.Tensor): The input sequences.
        cell_emb_conv_weights_and_biases (List[Tuple[torch.Tensor, torch.Tensor]]): A list of tuples containing the convolutional weights and biases for each cell in the cell type.
        predict (callable): The prediction function.
        clip_soft (float): The soft clipping value.
        model_type (str): The type of the model. Should be one of "multiome", "multiome_rna", "multiome_atac", or "rna".
        num_neighbors (int, optional): The number of neighbors. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The pseudobulk RNA profiles for the plus and minus strands, and the pseudobulk ATAC profile.
    """
    seqs_rev_comp = torch.flip(seqs.permute(0, 2, 1), (-1, -2)).permute(0, 2, 1)
    stacked_outputs_rna_plus, stacked_outputs_rna_minus, stacked_outputs_atac = [], [], []

    # go over embeddings for all cells, sum the unsquashed predictions over the gene slices
    for conv_weight, conv_bias in cell_emb_conv_weights_and_biases:
        # get predictions for all cells of one cell type
        outputs = predict(csb, seqs, seqs_rev_comp, conv_weight, conv_bias, bins_to_predict=None)
        outputs.float().detach()
        # get RNA:
        if "multiome" in model_type:
            outputs_rna = outputs[:, :, torch.tensor([1, 1, 0]).repeat(outputs.shape[2] // 3).bool()]
            outputs_atac = outputs[:, :, torch.tensor([0, 0, 1]).repeat(outputs.shape[2] // 3).bool()]
            outputs_atac = outputs_atac * 20
            stacked_outputs_atac.append(outputs_atac[0].sum(axis=-1, keepdim=True))
        else:
            outputs_rna = outputs
        outputs_rna = undo_squashed_scale(outputs_rna, clip_soft=clip_soft) * (1 / num_neighbors)

        num_pos = outputs_rna.shape[-1]
        stacked_outputs_rna_plus.append(outputs_rna[0, :, :num_pos:2].sum(axis=-1, keepdim=True))
        stacked_outputs_rna_minus.append(outputs_rna[0, :, 1:num_pos:2].sum(axis=-1, keepdim=True))
    return (
        torch.hstack(stacked_outputs_rna_plus).permute(1, 0),
        torch.hstack(stacked_outputs_rna_minus).permute(1, 0),
        torch.hstack(stacked_outputs_atac).permute(1, 0),
    )


def get_targets(targets, clip_soft, model_type, num_neighbors=1):
    """
    Processes the target profiles for RNA and ATAC.

    This function applies unsquashing and normalization to the target profiles and splits them into separate 
    tensors for the plus and minus strands of RNA and for ATAC.

    Args:
        targets (torch.Tensor): The target profiles.
        clip_soft (float): The soft clipping value.
        model_type (str): The type of the model. Should be one of "multiome", "multiome_rna", "multiome_atac", or "rna".
        num_neighbors (int, optional): The number of neighbors. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The processed RNA profiles for the plus and minus strands, and the processed ATAC profile.
    """
    if "multiome" in model_type:
        targets_rna = targets[:, :, torch.tensor([1, 1, 0]).repeat(targets.shape[2] // 3).bool()]
        targets_atac = targets[:, :, torch.tensor([0, 0, 1]).repeat(targets.shape[2] // 3).bool()]

        targets_atac = targets_atac[0] * 20
    else:
        targets_rna = targets

    num_pos = targets_rna.shape[-1]
    targets_rna = undo_squashed_scale(targets_rna, clip_soft=clip_soft) * (1 / num_neighbors)
    targets_rna_plus = targets_rna[0, :, :num_pos:2]
    targets_rna_minus = targets_rna[0, :, 1:num_pos:2]
    return targets_rna_plus.permute(1, 0), targets_rna_minus.permute(1, 0), targets_atac.permute(1, 0)


def read_backed(group, key):
    """
    Args:
        group (h5py.Group): HDF5 group object containing the AnnData data.
        key (str): Key within the 'obsm' group specifying the sparse matrix to load.

    Returns:
        anndata.AnnData: The loaded AnnData object.
    """
    return ad.AnnData(
        sparse_dataset(group["X"]),
        obsm={'fragment_single': sparse_dataset(group["obsm"][key])},
        **{
            k: read_elem(group[k]) if k in group else {}
            for k in ["layers", "obs", "var", "varm", "uns", "obsp", "varp"]
        }
    )


def add_weight_decay(model, lr, weight_decay=1e-5, skip_list=()):
    """
    Adding weight decay for model training.
    """
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



import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter


def dna_letter_at(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):
    fp = FontProperties(family="DejaVu Sans", weight="bold")

    globscale = 1.35

    LETTERS = {
        "T": TextPath((-0.305, 0), "T", size=1, prop=fp),
        "G": TextPath((-0.384, 0), "G", size=1, prop=fp),
        "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
        "C": TextPath((-0.366, 0), "C", size=1, prop=fp),
        "UP": TextPath((-0.488, 0), "$\\Uparrow$", size=1, prop=fp),
        "DN": TextPath((-0.488, 0), "$\\Downarrow$", size=1, prop=fp),
        "(": TextPath((-0.25, 0), "(", size=1, prop=fp),
        ".": TextPath((-0.125, 0), "-", size=1, prop=fp),
        ")": TextPath((-0.1, 0), ")", size=1, prop=fp),
    }

    COLOR_SCHEME = {
        "G": "orange",
        "A": "green",
        "C": "blue",
        "T": "red",
        "UP": "green",
        "DN": "red",
        "(": "black",
        ".": "black",
        ")": "black",
    }

    text = LETTERS[letter]

    chosen_color = COLOR_SCHEME[letter]
    if color is not None:
        chosen_color = color

    t = (
        mpl.transforms.Affine2D().scale(1 * globscale, yscale * globscale)
        + mpl.transforms.Affine2D().translate(x, y)
        + ax.transData
    )
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)

    if ax != None:
        ax.add_artist(p)
    return p


def plot_seq_scores(
    importance_scores,
    figsize=(16, 2),
    plot_y_ticks=True,
    y_min=None,
    y_max=None,
    save_figs=False,
    fig_name="default",
):
    importance_scores = importance_scores.T

    fig = plt.figure(figsize=figsize)

    ref_seq = ""
    for j in range(importance_scores.shape[1]):
        argmax_nt = np.argmax(np.abs(importance_scores[:, j]))

        if argmax_nt == 0:
            ref_seq += "A"
        elif argmax_nt == 1:
            ref_seq += "C"
        elif argmax_nt == 2:
            ref_seq += "G"
        elif argmax_nt == 3:
            ref_seq += "T"

    ax = plt.gca()

    for i in range(0, len(ref_seq)):
        mutability_score = np.sum(importance_scores[:, i])
        color = None
        dna_letter_at(ref_seq[i], i + 0.5, 0, mutability_score, ax, color=color)

    plt.sca(ax)
    plt.xticks([], [])
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    plt.xlim((0, len(ref_seq)))

    # plt.axis('off')

    if plot_y_ticks:
        plt.yticks(fontsize=12)
    else:
        plt.yticks([], [])

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    elif y_min is not None:
        plt.ylim(y_min)
    else:
        plt.ylim(
            np.min(importance_scores) - 0.1 * np.max(np.abs(importance_scores)),
            np.max(importance_scores) + 0.1 * np.max(np.abs(importance_scores)),
        )

    plt.axhline(y=0.0, color="black", linestyle="-", linewidth=1)

    # for axis in fig.axes :
    #    axis.get_xaxis().set_visible(False)
    #    axis.get_yaxis().set_visible(False)

    plt.tight_layout()

    if save_figs:
        plt.savefig(fig_name + ".png", transparent=True, dpi=300)
        plt.savefig(fig_name + ".eps")

    plt.show()


def visualize_input_gradient_pair(att_grad_wt, plot_start=0, plot_end=100, save_figs=False, fig_name=""):
    scores_wt = att_grad_wt[plot_start:plot_end, :]

    y_min = np.min(scores_wt)
    y_max = np.max(scores_wt)

    y_max_abs = max(np.abs(y_min), np.abs(y_max))

    y_min = y_min - 0.05 * y_max_abs
    y_max = y_max + 0.05 * y_max_abs

    print("--- Sequence attributions in cells that express the gene ---")
    plot_seq_scores(
        scores_wt,
        y_min=y_min,
        y_max=y_max,
        figsize=(40, 4),
        plot_y_ticks=False,
        save_figs=save_figs,
        fig_name=fig_name + "_wt",
    )
