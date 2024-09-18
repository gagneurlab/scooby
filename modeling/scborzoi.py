from borzoi_pytorch import Borzoi as Borzoi
import torch
import torch.nn as nn
from einops import rearrange


batch_conv = torch.vmap(nn.functional.conv1d, chunk_size = 1024)

class ScBorzoi(Borzoi):
    def __init__(self, cell_emb_dim, embedding_dim = 1920, n_tracks = 2, disable_cache = False, use_transform_borzoi_emb = False, cachesize = 2, **params):
        self.cell_emb_dim = cell_emb_dim
        self.cachesize = cachesize
        self.use_transform_borzoi_emb = use_transform_borzoi_emb
        self.n_tracks = n_tracks
        self.embedding_dim = embedding_dim
        self.disable_cache = disable_cache
        super(ScBorzoi, self).__init__(**params)
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
        
    def forward_cell_embs_only(self, cell_emb):
        bs, no_cell_embs ,cell_emb_dim,  = cell_emb.shape
        cell_emb = rearrange(cell_emb, 'b n d -> (b n) d')
        cell_emb_conv_weights = self.cell_state_to_conv(cell_emb) # out shape (b n) ((self.embedding_dim + 1)*self.n_tracks)
        cell_emb_conv_weights = rearrange(cell_emb_conv_weights, '(b n) (l d) -> b (n l) d', b = bs, l = self.n_tracks) #positive and negative strand
        cell_emb_conv_biases = cell_emb_conv_weights[:,:,-1:].view(bs,no_cell_embs*self.n_tracks)
        cell_emb_conv_weights = cell_emb_conv_weights[:,:,:-1].view(bs,no_cell_embs*self.n_tracks, self.embedding_dim ,1)
        return cell_emb_conv_weights,cell_emb_conv_biases 


    def forward_seq_to_emb(self, sequence):
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
        x = seq_emb
        if bins_to_predict is not None:
            out = batch_conv(x[:,:,bins_to_predict], cell_emb_conv_weights, cell_emb_conv_biases)
            out = torch.nn.functional.softplus(out).permute(0,2,1)
        else:
            out = batch_conv(x, cell_emb_conv_weights, cell_emb_conv_biases)
            out = torch.nn.functional.softplus(out).permute(0,2,1)
        return out

    
    def forward_sequence_w_convs(self, sequence, cell_emb_conv_weights, cell_emb_conv_biases, bins_to_predict = None):
        if self.sequences and not self.training and not self.disable_cache:                
            for i,s in enumerate(self.sequences):
                if torch.equal(sequence,s):
                    if bins_to_predict is not None: # unclear if this if is even needed or if self.last_embs[i][:,:,bins_to_predict] just also works when bins_to_predict is None 
                        out = batch_conv(self.last_embs[i][:,:,bins_to_predict], cell_emb_conv_weights, cell_emb_conv_biases)
                        out = torch.nn.functional.softplus(out)
                        return out.permute(0,2,1)
                    else:
                        out = batch_conv(self.last_embs[i], cell_emb_conv_weights, cell_emb_conv_biases)
                        out = torch.nn.functional.softplus(out)
                        return out.permute(0,2,1)
        x = self.forward_seq_to_emb(sequence)
        if bins_to_predict is not None:
            out = batch_conv(x[:,:,bins_to_predict], cell_emb_conv_weights, cell_emb_conv_biases)
            out = torch.nn.functional.softplus(out).permute(0,2,1)
        else:
            out = batch_conv(x, cell_emb_conv_weights, cell_emb_conv_biases)
            out = torch.nn.functional.softplus(out).permute(0,2,1)
        return out
        
    def forward(self, sequence, cell_emb):   
        cell_emb_conv_weights,cell_emb_conv_biases = self.forward_cell_embs_only(cell_emb)
        out = self.forward_sequence_w_convs(sequence, cell_emb_conv_weights, cell_emb_conv_biases)
        return out