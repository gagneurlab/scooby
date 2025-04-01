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
    """
    Converts a sparse RNA expression matrix to a dense coverage vector.

    This function processes a sparse matrix representing RNA expression counts and converts it into a dense 
    coverage vector, accounting for the strand information.

    Args:
        m (scipy.sparse.csr_matrix): Sparse matrix of RNA expression counts.
        seq_coord (tuple): Tuple containing genomic coordinates and sequence information.
        strand (str): Strand of the gene ('plus' or 'minus').

    Returns:
        torch.Tensor: Dense coverage vector for RNA expression.
    """
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
    """
    Converts a sparse ATAC-seq insertion matrix to a dense coverage vector.

    Args:
        m (scipy.sparse.csr_matrix): Sparse matrix of ATAC-seq insertion counts.
        seq_coord (tuple): Tuple containing genomic coordinates and sequence information.

    Returns:
        torch.Tensor: Dense coverage vector for ATAC-seq insertions.
    """
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
        embedding,
        ds,
        clip_soft,
        neighbors: Optional[scipy.sparse.csr_matrix] = None,
        cell_sample_size: int = 32,
        get_targets: bool = True,
        random_cells: bool = True,
        cells_to_run: Optional[np.ndarray] = None,
        cell_weights: Optional[np.ndarray] = None,
    ):
        """
    Dataset for on-the-fly generation of single-cell genomic profiles from sparse data.

    This dataset processes sparse RNA and (optionally) ATAC-seq data to generate dense coverage profiles 
    for individual cells or pseudobulk aggregates of cells. It utilizes cell embeddings to guide the selection 
    of cells and their neighbors.

    Attributes:
        adata_plus (anndata.AnnData): AnnData object containing RNA expression data for the plus strand.
        adata_minus (anndata.AnnData): AnnData object containing RNA expression data for the minus strand.
        embedding (pd.DataFrame): DataFrame containing cell embeddings.
        ds (GenomeIntervalDataset): Dataset providing genomic intervals and sequences.
        clip_soft (float): Soft clipping value for RNA coverage normalization.
        neighbors (scipy.sparse.csr_matrix): Sparse matrix representing cell neighborhood relationships.
        cell_sample_size (int, optional): Number of cells to sample per sequence. Defaults to 32.
        get_targets (bool, optional): Whether to generate target profiles. Defaults to True.
        random_cells (bool, optional): Whether to randomly sample cells. Defaults to True.
        cells_to_run (np.ndarray, optional): Array of cell indices to use (if not random). Defaults to None.
        cell_weights (np.ndarray, optional): Weights for cell sampling. Defaults to None.
        chrom_sizes (dict): Dictionary mapping chromosome names to their sizes and offsets.
    """
        self.clip_soft = clip_soft
        self.cell_weights = cell_weights
        self.cells_to_run = cells_to_run
        self.embedding = embedding
        self.neighbors = (neighbors if neighbors is not None else scipy.sparse.csr_matrix((self.embedding.shape[0], self.embedding.shape[0])))
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
        """
        Processes RNA expression data for the given cells and sequence coordinates.

        This function extracts RNA expression counts from the AnnData object, converts them to dense coverage
        vectors, applies normalization, and returns the processed profiles.

        Args:
            adata (anndata.AnnData): AnnData object containing RNA expression data.
            cells (list): List of cell indices.
            seq_coord (tuple): Tuple containing genomic coordinates and sequence information.
            strand (str): Strand of the gene ('plus' or 'minus').

        Returns:
            torch.Tensor: Processed RNA expression profiles for the given cells.
        """
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
        """
        Loads and processes pseudobulk RNA expression profiles for the given cells.

        Args:
            neighbors (list): List of cell indices.
            seq_coord (tuple): Tuple containing genomic coordinates and sequence information.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed pseudobulk RNA expression profiles for the plus and minus strands.
        """
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
            idx_cells = np.random.choice(self.embedding.shape[0], size=self.cell_sample_size, p=self.cell_weights)
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
        """
    Dataset for loading pre-computed pseudobulk profiles from BigWig files.

    This dataset loads pseudobulk RNA and ATAC-seq profiles from BigWig files for specified cell types, using 
    genomic intervals provided by a `GenomeIntervalDataset`.

    Attributes:
        cell_types (list): List of cell type names for which pseudobulk profiles are available.
        ds (GenomeIntervalDataset): Dataset providing genomic intervals and sequences.
        base_path (str): Path to the directory containing the BigWig files.
        seqlevelstyle (str, optional): Chromosome naming style ('UCSC' or 'ENSEMBL'). Defaults to 'UCSC'.
        clip_soft (float): Soft clipping value for RNA coverage normalization.
    """
        self.cell_types = cell_types
        self.genome_ds = ds
        self.base_path = base_path
        self.seqlevelstyle = seqlevelstyle
        self.clip_soft = clip_soft

    def __len__(self):
        return len(self.genome_ds)

    def _process_paths(self, paths, seq_coord):
        """
        Processes BigWig files to extract and normalize coverage values.

        This function opens BigWig files, extracts coverage values for the specified genomic interval, and 
        applies normalization and clipping.

        Args:
            paths (list): List of paths to BigWig files.
            seq_coord (pd.Series): Pandas Series containing genomic interval information.

        Returns:
            torch.Tensor: Processed and normalized coverage values for the given interval.
        """
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
        """
        Loads and processes pseudobulk profiles for the specified cell types.

        Args:
            neighbors (list): List of cell type names.
            seq_coord (pd.Series): Pandas Series containing genomic interval information.

        Returns:
            torch.Tensor: Concatenated pseudobulk profiles for the given cell types.
        """
        seq_cov = []
        for neighbor in neighbors:
            file_paths_plus = [f"{self.base_path}/plus.{neighbor}.bw"]
            file_paths_minus = [f"{self.base_path}/minus.{neighbor}.bw"]
            seq_cov.append(self._process_paths(file_paths_plus, seq_coord))
            seq_cov.append(self._process_paths(file_paths_minus, seq_coord))
        return torch.cat(seq_cov)

    def _reinit_fasta_reader(self):
        """
        Re-initializes the FastaInterval reader.

        This is necessary because pyfaidx and torch multiprocessing are not compatible.
        """
        self.genome_ds.fasta = FastaInterval(
            fasta_file=self.genome_ds.fasta.seqs.filename,
            context_length=self.genome_ds.fasta.context_length,
            return_seq_indices=self.genome_ds.fasta.return_seq_indices,
            shift_augs=self.genome_ds.fasta.shift_augs,
            rc_aug=self.genome_ds.fasta.rc_aug,
        )

    def __getitem__(self, idx):
        """
        Gets the item at the given index.

        Args:
            idx (int): The index of the item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The input sequences, reverse complemented sequences, and the target pseudobulk profiles.
        """
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
        embedding: pd.DataFrame,
        ds: GenomeIntervalDataset,
        clip_soft,
        neighbors: Optional[scipy.sparse.csr_matrix] = None,
        cell_sample_size: int = 32,
        get_targets: bool = True,
        random_cells: bool = True,
        cells_to_run: Optional[np.ndarray] = None,
        cell_weights: Optional[np.ndarray] = None,
        normalize_atac: bool = False,
    ) -> None:
        """
    Dataset for on-the-fly generation of multi-modal genomic profiles from sparse single-cell data.

    This dataset processes sparse RNA and ATAC-seq data to generate dense coverage profiles for individual
    cells or pseudobulk aggregates. It utilizes cell embeddings to guide cell selection and neighborhood 
    aggregation.

    Attributes:
        adatas (dict): Dictionary mapping modality names (e.g., 'rna_plus', 'atac') to their corresponding AnnData objects.
        embedding (pd.DataFrame): DataFrame containing cell embeddings.
        ds (GenomeIntervalDataset): Dataset providing genomic intervals and sequences.
        clip_soft (float): Soft clipping value for RNA coverage normalization.
        neighbors (scipy.sparse.csr_matrix): Sparse matrix representing cell neighborhood relationships.
        cell_sample_size (int, optional): Number of cells to sample per sequence. Defaults to 32.
        get_targets (bool, optional): Whether to generate target profiles. Defaults to True.
        random_cells (bool, optional): Whether to randomly sample cells. Defaults to True.
        cells_to_run (np.ndarray, optional): Array of cell indices to use (if not random). Defaults to None.
        cell_weights (np.ndarray, optional): Weights for cell sampling. Defaults to None.
        normalize_atac (bool, optional): Whether to normalize ATAC-seq coverage. Defaults to False.
        chrom_sizes (dict): Dictionary mapping chromosome names to their sizes and offsets.
    """
        self.clip_soft = clip_soft
        self.cell_weights = cell_weights
        self.cells_to_run = cells_to_run
        self.embedding = embedding
        self.neighbors = (neighbors if neighbors is not None else scipy.sparse.csr_matrix((self.embedding.shape[0], self.embedding.shape[0])))
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
        """
        Processes RNA expression data for the given cells and sequence coordinates.

        This function extracts RNA expression counts from the AnnData object, converts them to dense coverage
        vectors, applies normalization, and returns the processed profiles.

        Args:
            adata (anndata.AnnData): AnnData object containing RNA expression data.
            cell_indices (list): List of cell indices.
            seq_coord (tuple): Tuple containing genomic coordinates and sequence information.
            strand (str): Strand of the gene ('plus' or 'minus').

        Returns:
            torch.Tensor: Processed RNA expression profiles for the given cells.
        """
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
        """
        Processes ATAC-seq data for the given cells and sequence coordinates.

        This function extracts ATAC-seq insertion counts from the AnnData object, converts them to dense 
        coverage vectors, applies normalization (if specified), and returns the processed profiles.

        Args:
            adata (anndata.AnnData): AnnData object containing ATAC-seq insertion data.
            cell_indices (list): List of cell indices.
            seq_coord (tuple): Tuple containing genomic coordinates and sequence information.

        Returns:
            torch.Tensor: Processed ATAC-seq profiles for the given cells.
        """
        tensor = _sparse_to_coverage_atac(m=adata.obsm["insertion"][cell_indices], seq_coord=seq_coord)
        seq_cov = torch.nn.functional.avg_pool1d(tensor, kernel_size=32, stride=32) * 32

        if self.normalize_atac:
            seq_cov = seq_cov * 0.05
        return seq_cov

    def _load_pseudobulk(self, neighbors, seq_coord):
        """
        Loads and processes pseudobulk profiles for RNA and ATAC-seq data.

        Args:
            neighbors (list): List of cell indices to aggregate into a pseudobulk profile.
            seq_coord (tuple): Tuple containing genomic coordinates and sequence information.

        Returns:
            torch.Tensor: Concatenated pseudobulk profiles for RNA and ATAC-seq data.
        """
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
        """
        Re-initializes the FastaInterval reader.

        This is necessary because pyfaidx and torch multiprocessing can have compatibility issues.
        """
        self.genome_ds.fasta = FastaInterval(
            fasta_file=self.genome_ds.fasta.seqs.filename,
            context_length=self.genome_ds.fasta.context_length,
            return_seq_indices=self.genome_ds.fasta.return_seq_indices,
            shift_augs=self.genome_ds.fasta.shift_augs,
            rc_aug=self.genome_ds.fasta.rc_aug,
        )

    def __getitem__(self, idx):
        """
        Retrieves data for a given genomic interval.

        This function retrieves the DNA sequence, processes RNA and ATAC-seq data for the selected cells and their
        neighbors, and returns the input sequence, reverse complement, target profiles (if `get_targets` is True), 
        and cell embeddings.

        Args:
            idx (int): Index of the genomic interval in the `GenomeIntervalDataset`.

        Returns:
            Tuple: A tuple containing the input sequence, reverse complement sequence, target profiles (optional), and cell embeddings.
        """
        self._reinit_fasta_reader()
        if self.random_cells:
            idx_cells = np.random.choice(self.embedding.shape[0], size=self.cell_sample_size, p=self.cell_weights)
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