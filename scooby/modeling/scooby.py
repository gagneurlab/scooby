from borzoi_pytorch import Borzoi as Borzoi
import torch
import torch.nn as nn
from einops import rearrange
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F


batch_conv = torch.vmap(F.conv1d, chunk_size = 1024)

class Scooby(Borzoi):
    def __init__(self, config, cell_emb_dim, embedding_dim = 1920, n_tracks = 2, disable_cache = False, use_transform_borzoi_emb = False, cachesize = 2, count_only = False, **params):
        """
    Scooby model for predicting single-cell genomic profiles from DNA sequence.

    This model extends the Borzoi architecture to handle single-cell data by 
    incorporating a cell-state-specific decoder. It leverages pre-trained weights 
    from Borzoi and employs low-rank adaptation (LoRA) for parameter-efficient 
    fine-tuning.

    Attributes:
        config: Borzoi model configuration.
        cell_emb_dim: Dimension of cell embeddings.
        embedding_dim: Dimension of sequence embeddings (default: 1920).
        n_tracks: Number of output tracks (e.g., 2 for stranded RNA) (default: 2).
        disable_cache: Whether to disable sequence embedding caching (default: False).
        use_transform_borzoi_emb: Whether to use an additional transformation layer on Borzoi embeddings (default: False).
        cachesize: Size of the sequence embedding cache (default: 2).
    """
        super().__init__(config)
        self.cell_emb_dim = cell_emb_dim
        self.cachesize = cachesize
        self.use_transform_borzoi_emb = use_transform_borzoi_emb
        self.n_tracks = n_tracks
        self.embedding_dim = embedding_dim
        self.disable_cache = disable_cache
        self.count_only = count_only
        dropout_modules = [module for module in self.modules() if isinstance(module, torch.nn.Dropout)]
        batchnorm_modules = [module for module in self.modules() if isinstance(module, torch.nn.BatchNorm1d)]
        [module.eval() for module in dropout_modules] # disable dropout
        [module.eval() for module in batchnorm_modules] # disable batchnorm
        self.cell_state_to_conv = nn.Sequential(
            nn.Linear(cell_emb_dim, 128),
            nn.GELU(), 
            nn.Linear(128, 256),
            nn.GELU(), 
            nn.Dropout(0.2),
            nn.Linear(256, 1024),
            nn.GELU(), 
            nn.Dropout(0.2),
            nn.Linear(1024, (self.embedding_dim + 1)*self.n_tracks) #bias gets one more, and we predict pos. and neg. strand
        ) 
        if use_transform_borzoi_emb:
            self.transform_borzoi_emb = nn.Sequential(
                nn.Conv1d(embedding_dim, 4096, 1), #512 or 4096
                nn.GELU(),
                nn.Conv1d(4096, embedding_dim, 1),
                nn.GELU()
            ) 
            nn.init.zeros_(self.transform_borzoi_emb[-2].weight)
            nn.init.zeros_(self.transform_borzoi_emb[-2].bias)
        nn.init.zeros_(self.cell_state_to_conv[-1].bias)
        self.sequences, self.last_embs = [], []
        del self.human_head


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=1.0)
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()
        
        
    def forward_cell_embs_only(self, cell_emb):
        """
        Processes cell embeddings to generate convolutional filter weights and biases.

        Args:
            cell_emb: Tensor of cell embeddings (batch_size, num_cells, cell_emb_dim).

        Returns:
            Tuple: Convolutional filter weights and biases.
        """
        bs, no_cell_embs ,cell_emb_dim,  = cell_emb.shape
        cell_emb = rearrange(cell_emb, 'b n d -> (b n) d')
        cell_emb_conv_weights = self.cell_state_to_conv(cell_emb) # out shape (b n) ((self.embedding_dim + 1)*self.n_tracks)
        cell_emb_conv_weights = rearrange(cell_emb_conv_weights, '(b n) (l d) -> b (n l) d', b = bs, l = self.n_tracks) #positive and negative strand
        cell_emb_conv_biases = cell_emb_conv_weights[:,:,-1:].view(bs,no_cell_embs*self.n_tracks)
        cell_emb_conv_weights = cell_emb_conv_weights[:,:,:-1].view(bs,no_cell_embs*self.n_tracks, self.embedding_dim ,1)
        return cell_emb_conv_weights,cell_emb_conv_biases 


    def forward_seq_to_emb(self, sequence):
        """
        Processes DNA sequences through Borzoi backbone to obtain sequence embeddings.

        Args:
            sequence: Tensor of DNA sequences (batch_size, seq_len, 4).

        Returns:
            Tensor: Sequence embeddings.
        """
        bs, seq_len, _ = sequence.shape  
        x = sequence
        x = self.conv_dna(x)
        x_unet0 = self.res_tower(x)
        x_unet1 = self.unet1(x_unet0)
        x = self._max_pool(x_unet1)
        x_unet1 = self.horizontal_conv1(x_unet1)
        x_unet0 = self.horizontal_conv0(x_unet0)
        x = self.transformer(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.upsampling_unet1(x)
        x += x_unet1
        x = self.separable1(x)
        x = self.upsampling_unet0(x)
        x += x_unet0
        x = self.separable0(x)
        x = self.crop(x.permute(0, 2, 1))
        x = self.final_joined_convs(x.permute(0, 2, 1))
        if self.use_transform_borzoi_emb:
            x = self.transform_borzoi_emb(x) 
        if not self.training and not self.disable_cache:
            if len(self.sequences) == self.cachesize:
                self.sequences, self.last_embs = [], []
            self.sequences.append(sequence)
            self.last_embs.append(x)
        return x

    def forward_convs_on_emb(self, seq_emb, cell_emb_conv_weights, cell_emb_conv_biases, bins_to_predict = None):
        """
        Applies cell-state-specific convolutions to sequence embeddings.

        Args:
            seq_emb: Tensor of sequence embeddings.
            cell_emb_conv_weights: Convolutional filter weights.
            cell_emb_conv_biases: Convolutional filter biases.
            bins_to_predict (optional): Indices of bins to predict (if None, predicts all bins).

        Returns:
            Tensor: Predicted profiles.
        """
        x = seq_emb
        if bins_to_predict is not None:
            out = batch_conv(x[:,:,bins_to_predict], cell_emb_conv_weights, cell_emb_conv_biases)
        else:
            out = batch_conv(x, cell_emb_conv_weights, cell_emb_conv_biases)
        out = F.softplus(out)
        return out.permute(0,2,1)

    
    def forward_sequence_w_convs(self, sequence, cell_emb_conv_weights, cell_emb_conv_biases, bins_to_predict = None):
        """
        Processes DNA sequence, applies cell-state-specific convolutions, and caches results.

        Args:
            sequence: Tensor of DNA sequences.
            cell_emb_conv_weights: Convolutional filter weights.
            cell_emb_conv_biases: Convolutional filter biases.
            bins_to_predict (optional): Indices of bins to predict.

        Returns:
            Tensor: Predicted profiles.
        """
        if self.sequences and not self.training and not self.disable_cache:                
            for i,s in enumerate(self.sequences):
                if torch.equal(sequence,s):
                    cell_emb_conv_weights, cell_emb_conv_biases = cell_emb_conv_weights.to(self.last_embs[i].dtype), cell_emb_conv_biases.to(self.last_embs[i].dtype)
                    if bins_to_predict is not None: # unclear if this if is even needed or if self.last_embs[i][:,:,bins_to_predict] just also works when bins_to_predict is None 
                        out = batch_conv(self.last_embs[i][:,:,bins_to_predict], cell_emb_conv_weights, cell_emb_conv_biases)
                    else:
                        out = batch_conv(self.last_embs[i], cell_emb_conv_weights, cell_emb_conv_biases)
                    out = F.softplus(out)
                    return out.permute(0,2,1)
        x = self.forward_seq_to_emb(sequence)
        cell_emb_conv_weights, cell_emb_conv_biases = cell_emb_conv_weights.to(x.dtype), cell_emb_conv_biases.to(x.dtype)
        if bins_to_predict is not None:
            out = batch_conv(x[:,:,bins_to_predict], cell_emb_conv_weights, cell_emb_conv_biases)
        else:
            out = batch_conv(x, cell_emb_conv_weights, cell_emb_conv_biases)
        out = F.softplus(out)
        return out.permute(0,2,1)
        
    def forward(self, sequence, cell_emb, gene_slices = None):
        """
        Forward pass of the scooby model.

        Args:
            sequence: Tensor of DNA sequences (batch_size, seq_len, 4).
            cell_emb: Tensor of cell embeddings (batch_size, num_cells, cell_emb_dim).

        Returns:
            Tensor: Predicted profiles for each cell (batch_size, num_cells, seq_len, n_tracks).
        """
        cell_emb_conv_weights,cell_emb_conv_biases = self.forward_cell_embs_only(cell_emb)
        out = self.forward_sequence_w_convs(sequence, cell_emb_conv_weights, cell_emb_conv_biases, bins_to_predict = gene_slices)
        if self.count_only:
            assert gene_slices is not None
            out = torch.log1p(torch.sum(out, dim = -1))
        return out
