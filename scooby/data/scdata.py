from typing import Optional
import numpy as np
import pandas as pd
import pybigtools
import scipy.sparse
import torch
import tqdm
from torch.utils.data import Dataset

from enformer_pytorch.data import FastaInterval, GenomeIntervalDataset

min_value = torch.finfo(torch.float16).min
max_value = torch.finfo(torch.float16).max


def _sparse_to_coverage_rna(m, seq_coord, strand):
    _, _, chrom_end, start, end, seq_coord_2, seq_coord_3 = seq_coord
    m = m[:, start:end]
    # Initialize dense matrix with zeros
    dense_matrix = np.zeros(m.shape[1], dtype=np.single)
    # Iterate over non-zero elements of the sparse matrix
    if strand == "plus":
        for row in range(m.shape[0]):
            col_indices = m.indices[m.indptr[row] : m.indptr[row + 1]]
            values = m.data[m.indptr[row] : m.indptr[row + 1]]
            for col_index, value in zip(col_indices, values):
                dense_matrix[col_index : (col_index + value)] += 1 / 90
    elif strand == "minus":
        for row in range(m.shape[0]):
            col_indices = m.indices[m.indptr[row] : m.indptr[row + 1]]
            values = m.data[m.indptr[row] : m.indptr[row + 1]]
            for col_index, value in zip(col_indices, values):
                dense_matrix[(col_index + value + 1) : (col_index + 1)] += 1 / 90
    # restrict to relevant part
    dense_matrix = dense_matrix[min([100, seq_coord_2]) : max([-100, seq_coord_3 - chrom_end])]
    dense_matrix = torch.from_numpy(dense_matrix).unsqueeze(0)
    return dense_matrix


def _sparse_to_coverage_atac(m, seq_coord):
    _, _, chrom_end, start, end, seq_coord_2, seq_coord_3 = seq_coord
    m = m[:, start:end]
    dense_matrix = m.sum(0).astype(np.single).A[0]
    # restrict to relevant part
    dense_matrix = dense_matrix[min([100, seq_coord_2]) : max([-100, seq_coord_3 - chrom_end])]
    # For ATAC it is easy because we can just use the matrix as is
    dense_matrix = torch.from_numpy(dense_matrix).unsqueeze(0)

    return dense_matrix


class onTheFlyDataset(Dataset):
    def __init__(
        self,
        adata_plus,
        adata_minus,
        neighbors,
        embedding,
        ds,
        clip_soft,
        cell_sample_size=32,
        get_targets=True,
        random_cells=True,
        cells_to_run=None,
        cell_weights=None,
    ):
        self.clip_soft = clip_soft
        self.neighbors = neighbors
        self.cell_weights = cell_weights
        self.cells_to_run = cells_to_run
        self.embedding = embedding
        self.get_targets = get_targets
        self.random_cells = random_cells
        if not self.random_cells and not cells_to_run:
            # we are probably just providing seqs?
            self.cells_to_run = np.zeros(1, dtype=np.int64)
        self.genome_ds = ds
        self.cell_sample_size = cell_sample_size
        self.adata_plus = adata_plus
        self.adata_minus = adata_minus
        try:
            self.chrom_sizes = adata_plus.uns["reference_sequences"].copy()
            self.chrom_sizes["offset"] = np.insert(self.chrom_sizes["reference_seq_length"].cumsum()[:-1].values, 0, 0)
            self.chrom_sizes = self.chrom_sizes.set_index("reference_seq_name").to_dict("index")
        except:
            pass

    def __len__(self):
        return len(self.genome_ds)

    def _get_neighbors_for_cell(self, bar_code_id):
        cell_neighbor_ids = self.neighbors[bar_code_id].nonzero()[1]
        neighbors_to_load = cell_neighbor_ids.tolist() + [bar_code_id]
        return neighbors_to_load

    def _process_cells(self, adata, cells, seq_coord, strand):
        m = adata.obsm["fragment_single"][cells]
        tensor = _sparse_to_coverage_rna(m=m, seq_coord=seq_coord, strand=strand)
        seq_cov = torch.nn.functional.avg_pool1d(tensor, kernel_size=32, stride=32) * 32
        seq_cov = -1 + (1 + seq_cov) ** 0.75

        clip_soft = self.clip_soft
        clip = 768

        clip_mask = seq_cov > clip_soft
        seq_cov[clip_mask] = clip_soft - 1 + torch.sqrt(seq_cov[clip_mask] - clip_soft + 1)
        seq_cov = torch.clip(seq_cov, -clip, clip)

        return seq_cov

    def _load_pseudobulk(self, neighbors, seq_coord):
        seq_cov_plus = self._process_cells(self.adata_plus, neighbors, seq_coord, strand="plus")
        seq_cov_minus = self._process_cells(self.adata_minus, neighbors, seq_coord, strand="minus")
        return seq_cov_plus, seq_cov_minus  #

    def _reinit_fasta_reader(self):
        # we seem to need this as pyfaidx and torch multiprocessing are not friends
        self.genome_ds.fasta = FastaInterval(
            fasta_file=self.genome_ds.fasta.seqs.filename,
            context_length=self.genome_ds.fasta.context_length,
            return_seq_indices=self.genome_ds.fasta.return_seq_indices,
            shift_augs=self.genome_ds.fasta.shift_augs,
            rc_aug=self.genome_ds.fasta.rc_aug,
        )

    def __getitem__(self, idx):
        self._reinit_fasta_reader()
        if self.random_cells:
            idx_cells = np.random.choice(self.neighbors.shape[0], size=self.cell_sample_size, p=self.cell_weights)
        else:
            idx_cells = self.cells_to_run
        idx_gene = idx
        seq_coord = self.genome_ds.df[idx_gene]
        inputs, _, rc_augs = self.genome_ds[idx_gene]
        embeddings = torch.from_numpy(np.vstack(self.embedding.iloc[idx_cells]["embedding"].values))
        if self.get_targets:
            chrom_size = self.chrom_sizes[seq_coord["column_1"].item()]
            chrom_start = chrom_size["offset"]
            chrom_end = chrom_size["reference_seq_length"]
            seq_coord_2, seq_coord_3 = seq_coord["column_2"].item(), seq_coord["column_3"].item()
            start = np.max([0, seq_coord_2 - 100]) + chrom_start
            end = np.min([seq_coord_3 + 100, chrom_end]) + chrom_start
            genome_data = [chrom_size, chrom_start, chrom_end, start, end, seq_coord_2, seq_coord_3]
            targets = []

            for cell_idx in tqdm.tqdm(idx_cells, disable=True):
                neighbors_to_load = self._get_neighbors_for_cell(cell_idx)
                targets.extend(self._load_pseudobulk(neighbors_to_load, genome_data))
            targets = torch.vstack(targets)
            return inputs, rc_augs, targets.permute(1, 0), embeddings
        return inputs, rc_augs, embeddings


class onTheFlyPseudobulkDataset(Dataset):
    def __init__(self, cell_types, ds, base_path, seqlevelstyle="UCSC", clip_soft = 384):
        self.cell_types = cell_types
        self.genome_ds = ds
        self.base_path = base_path
        self.seqlevelstyle = seqlevelstyle
        self.clip_soft = clip_soft

    def __len__(self):
        return len(self.genome_ds)

    def _process_paths(self, paths, seq_coord):
        bigwigs = [pybigtools.open(file) for file in paths]
        cons_vals = [
            bw.values(
                seq_coord["column_1"].item().replace("chr", "")
                if self.seqlevelstyle == "ENSEMBL"
                else seq_coord["column_1"].item(),
                seq_coord["column_2"].item(),
                seq_coord["column_3"].item(),
            )
            for bw in bigwigs
        ]
        tensor = torch.nan_to_num(torch.as_tensor(np.array(cons_vals, dtype= np.single)))
        tensor = tensor.sum(axis=0).unsqueeze(0)
        # divide by mean read length
        mean_read_length = 90
        tensor = tensor / mean_read_length
        seq_cov = torch.nn.functional.avg_pool1d(tensor, kernel_size=32, stride=32) * 32
        seq_cov = -1 + (1 + seq_cov) ** 0.75

        clip_soft = self.clip_soft
        clip = 768

        clip_mask = seq_cov > clip_soft
        seq_cov[clip_mask] = clip_soft - 1 + torch.sqrt(seq_cov[clip_mask] - clip_soft + 1)
        seq_cov = torch.clip(seq_cov, -clip, clip)
        seq_cov = torch.clip(seq_cov, min_value, max_value)
        return seq_cov

    def _load_pseudobulk(self, neighbors, seq_coord):
        seq_cov = []
        for neighbor in neighbors:
            file_paths_plus = [f"{self.base_path}/plus.{neighbor}.bw"]
            file_paths_minus = [f"{self.base_path}/minus.{neighbor}.bw"]
            seq_cov.append(self._process_paths(file_paths_plus, seq_coord))
            seq_cov.append(self._process_paths(file_paths_minus, seq_coord))
        return torch.cat(seq_cov)

    def _reinit_fasta_reader(self):
        """We do this because pyfaidX and torch multiprocessing sucks."""
        self.genome_ds.fasta = FastaInterval(
            fasta_file=self.genome_ds.fasta.seqs.filename,
            context_length=self.genome_ds.fasta.context_length,
            return_seq_indices=self.genome_ds.fasta.return_seq_indices,
            shift_augs=self.genome_ds.fasta.shift_augs,
            rc_aug=self.genome_ds.fasta.rc_aug,
        )

    def __getitem__(self, idx):
        self._reinit_fasta_reader()
        idx_gene = idx
        seq_coord = self.genome_ds.df[idx_gene]
        targets = self._load_pseudobulk(self.cell_types, seq_coord)
        inputs, _, rc_augs = self.genome_ds[idx_gene]
        return inputs, rc_augs, targets.permute(1, 0)


# Multiome Dataloaders


class onTheFlyMultiomeDataset(Dataset):  # noqa: D101
    def __init__(
        self,
        adatas: dict,
        neighbors: scipy.sparse.csr_matrix,
        embedding: pd.DataFrame,
        ds: GenomeIntervalDataset,
        clip_soft,
        cell_sample_size: int = 32,
        get_targets: bool = True,
        random_cells: bool = True,
        cells_to_run: Optional[np.ndarray] = None,
        cell_weights: Optional[np.ndarray] = None,
        normalize_atac: bool = False,
    ) -> None:
        self.clip_soft = clip_soft
        self.neighbors = neighbors
        self.cell_weights = cell_weights
        self.cells_to_run = cells_to_run
        self.embedding = embedding
        self.get_targets = get_targets
        self.random_cells = random_cells
        if not self.random_cells and not cells_to_run:
            self.cells_to_run = np.zeros(1, dtype=np.int64)
        self.genome_ds = ds
        self.cell_sample_size = cell_sample_size
        self.adatas = adatas
        self.normalize_atac = normalize_atac

        try:
            self.chrom_sizes = self.adatas["rna_plus"].uns["reference_sequences"].copy()
            if "chr" not in self.chrom_sizes["reference_seq_name"][0]:
                # convert to chr1, chr2, etc
                self.chrom_sizes["reference_seq_name"] = "chr" + self.chrom_sizes["reference_seq_name"].astype(str)
            self.chrom_sizes["offset"] = np.insert(self.chrom_sizes["reference_seq_length"].cumsum()[:-1].values, 0, 0)
            self.chrom_sizes = self.chrom_sizes.set_index("reference_seq_name").to_dict("index")
        except:
            pass

    def __len__(self):
        return len(self.genome_ds)

    def _get_neighbors_for_cell(self, bar_code_id):  # noqa: D102
        cell_neighbor_ids = self.neighbors[bar_code_id].nonzero()[1]
        neighbors_to_load = cell_neighbor_ids.tolist() + [bar_code_id]
        return neighbors_to_load

    def _process_rna(self, adata, cell_indices, seq_coord, strand):
        tensor = _sparse_to_coverage_rna(
            m=adata.obsm["fragment_single"][cell_indices], seq_coord=seq_coord, strand=strand
        )
        seq_cov = torch.nn.functional.avg_pool1d(tensor, kernel_size=32, stride=32) * 32
        seq_cov = -1 + (1 + seq_cov) ** 0.75

        clip_soft = self.clip_soft
        clip = 768

        clip_mask = seq_cov > clip_soft
        if clip_mask.any():
            seq_cov[clip_mask] = clip_soft - 1 + torch.sqrt(seq_cov[clip_mask] - clip_soft + 1)
        seq_cov = torch.clip(seq_cov, -clip, clip)
        return seq_cov

    def _process_atac(self, adata, cell_indices, seq_coord):
        tensor = _sparse_to_coverage_atac(m=adata.obsm["insertion"][cell_indices], seq_coord=seq_coord)
        seq_cov = torch.nn.functional.avg_pool1d(tensor, kernel_size=32, stride=32) * 32

        if self.normalize_atac:
            seq_cov = seq_cov * 0.05
        return seq_cov

    def _load_pseudobulk(self, neighbors, seq_coord):
        # process all modalities
        seq_covs = []
        for modality, adata in self.adatas.items():
            if "rna" in modality:
                strand = modality.split("_")[-1]
                seq_cov = self._process_rna(adata, neighbors, seq_coord, strand=strand)
            elif "atac" in modality:
                seq_cov = self._process_atac(adata, neighbors, seq_coord)
            seq_covs.append(seq_cov)
        return torch.cat(seq_covs)

    def _reinit_fasta_reader(self):
        self.genome_ds.fasta = FastaInterval(
            fasta_file=self.genome_ds.fasta.seqs.filename,
            context_length=self.genome_ds.fasta.context_length,
            return_seq_indices=self.genome_ds.fasta.return_seq_indices,
            shift_augs=self.genome_ds.fasta.shift_augs,
            rc_aug=self.genome_ds.fasta.rc_aug,
        )

    def __getitem__(self, idx):
        self._reinit_fasta_reader()
        if self.random_cells:
            idx_cells = np.random.choice(self.neighbors.shape[0], size=self.cell_sample_size, p=self.cell_weights)
        else:
            idx_cells = self.cells_to_run
        idx_gene = idx
        seq_coord = self.genome_ds.df[idx_gene]
        inputs, _, rc_augs = self.genome_ds[idx_gene]
        embeddings = torch.from_numpy(np.vstack(self.embedding.iloc[idx_cells]["embedding"].values))

        if self.get_targets:
            chrom_size = self.chrom_sizes[seq_coord["column_1"].item()]
            chrom_start = chrom_size["offset"]
            chrom_end = chrom_size["reference_seq_length"]

            seq_coord_2, seq_coord_3 = seq_coord["column_2"].item(), seq_coord["column_3"].item()
            start = np.max([0, seq_coord_2 - 100]) + chrom_start
            end = np.min([seq_coord_3 + 100, chrom_end]) + chrom_start
            genome_data = [chrom_size, chrom_start, chrom_end, start, end, seq_coord_2, seq_coord_3]

            targets = []
            for cell_idx in tqdm.tqdm(idx_cells, disable=True):
                neighbors_to_load = self._get_neighbors_for_cell(cell_idx)
                targets.append(self._load_pseudobulk(neighbors_to_load, genome_data))
            targets = torch.vstack(targets)
            return inputs, rc_augs, targets.permute(1, 0), embeddings
        return inputs, rc_augs, embeddings


class onTheFlyExonMultiomePseudobulkDataset(Dataset):
    def __init__(self, cell_types, ds, base_path, clip_soft, seqlevelstyle="UCSC"):
        self.cell_types = cell_types
        self.genome_ds = ds
        self.base_path = base_path
        self.seqlevelstyle = seqlevelstyle
        self.clip_soft = clip_soft

    def __len__(self):
        return len(self.genome_ds)

    def _process_rna_paths(self, paths, seq_coord):
        bigwigs = [pybigtools.open(file) for file in paths]

        cons_vals = [
            bw.values(seq_coord["column_1"].item(), seq_coord["column_2"].item(), seq_coord["column_3"].item())
            for bw in bigwigs
        ]
        tensor = torch.nan_to_num(torch.as_tensor(np.array(cons_vals)))
        tensor = tensor.sum(axis=0).unsqueeze(0)
        # divide by mean read length
        mean_read_length = 90
        tensor = tensor / mean_read_length
        seq_cov = torch.nn.functional.avg_pool1d(tensor, kernel_size=32, stride=32) * 32
        seq_cov = -1 + (1 + seq_cov) ** 0.75

        clip = 768

        clip_mask = seq_cov > self.clip_soft
        seq_cov[clip_mask] = self.clip_soft - 1 + torch.sqrt(seq_cov[clip_mask] - self.clip_soft + 1)
        seq_cov = torch.clip(seq_cov, -clip, clip)
        return seq_cov

    def _process_atac_paths(self, paths, seq_coord):
        bigwigs = [pybigtools.open(file) for file in paths]

        cons_vals = [
            bw.values(seq_coord["column_1"].item(), seq_coord["column_2"].item(), seq_coord["column_3"].item())
            for bw in bigwigs
        ]
        tensor = torch.nan_to_num(torch.as_tensor(np.array(cons_vals)))
        tensor = tensor.sum(axis=0).unsqueeze(0)
        seq_cov = torch.nn.functional.avg_pool1d(tensor, kernel_size=32, stride=32) * 32
        return torch.clip(seq_cov, min_value, max_value)

    def _load_pseudobulk(self, cell_types, seq_coord):
        seq_cov = []
        for cell_type in cell_types:
            file_paths_plus = [f"{self.base_path}/plus.{cell_type}.bw"]
            file_paths_minus = [f"{self.base_path}/minus.{cell_type}.bw"]
            file_paths_atac = [f"{self.base_path}/atac.{cell_type}.bw"]
            seq_cov.append(self._process_rna_paths(file_paths_plus, seq_coord))
            seq_cov.append(self._process_rna_paths(file_paths_minus, seq_coord))
            seq_cov.append(self._process_atac_paths(file_paths_atac, seq_coord))
        return torch.cat(seq_cov)

    def _reinit_fasta_reader(self):
        self.genome_ds.fasta = FastaInterval(
            fasta_file=self.genome_ds.fasta.seqs.filename,
            context_length=self.genome_ds.fasta.context_length,
            return_seq_indices=self.genome_ds.fasta.return_seq_indices,
            shift_augs=self.genome_ds.fasta.shift_augs,
            rc_aug=self.genome_ds.fasta.rc_aug,
        )

    def __getitem__(self, idx):
        self._reinit_fasta_reader()
        idx_gene = idx
        seq_coord = self.genome_ds.df[idx_gene]
        if self.seqlevelstyle == "ENSEMBL":
            seq_coord = seq_coord.with_columns(seq_coord["column_1"].str.replace("chr", ""))

        targets = self._load_pseudobulk(self.cell_types, seq_coord)
        inputs, _, rc_augs = self.genome_ds[idx_gene]
        return inputs, rc_augs, targets.permute(1, 0)
