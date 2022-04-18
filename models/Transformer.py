import math
import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_forecasting.metrics import MAPE
import pytorch_lightning as pl
from torch.optim import Adam

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = (src == 0.0).transpose(0, 1)
    tgt_padding_mask = (tgt == 0.0).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class Seq2SeqTransformer(pl.LightningModule):
    def __init__(self,
                 n_layers: int,
                 input_dim: int,
                 nhead: int,
                 output_dim: int,
                 hidden_dim: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        
        # The high-level Transformer module provided by PyTorch
        self.transformer = nn.Transformer(
                                       # Same as d_model in the Vaswani et al. paper
                                       d_model=input_dim,
                                       # The number of attention heads
                                       nhead=nhead,
                                       # The number of encoder layers
                                       num_encoder_layers=n_layers,
                                       # The number of decoder layers
                                       num_decoder_layers=n_layers,
                                       # The dimension of the feed-forward network for
                                       # both the encoder and the decoder.
                                       dim_feedforward=hidden_dim,
                                       # The dropout probability applied at the
                                       # fully-connected layers of the Transformer.
                                       dropout=dropout)
        
        # This fully connected layer reverses the embedding, giving us a
        # vector with the same number of dimensions as words in the vocab
        self.generator = nn.Linear(hidden_dim, output_dim)
        
   
        # This is a positional encoding layer, which will
        # take our normal embeddings and add those sinusoidal signals
        self.positional_encoding = PositionalEncoding(
            hidden_dim, dropout=dropout)
        
        self.loss_fn = torch.nn.L1Loss()

    def forward(self,
                # The src (or source) is the Article Content in our case
                src: torch.Tensor,
                # The tgt (or target) is the Article Headline in our case
                trg: torch.Tensor,
                # This is the mask we'll apply to the source. This isn't
                # particularly useful in our case, but is good for 
                # language modelling.
                src_mask: torch.Tensor,
                # This is the mask we'll apply to the target. We'll use
                # this to prevent the decoder from peaking ahead.
                tgt_mask: torch.Tensor,
                # We'll also mask any padding tokens so they don't 
                # contribute to the attention.
                src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor):
        
        # We'll apply the positional embedding to the token embeddings.
        src_emb = self.positional_encoding(src)
        tgt_emb = self.positional_encoding(trg)
        
        # We then feed those token embeddings to the Transformer layer
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        
        # The transformer outputs have the same dimension as the embeddings
        # We'll want to use the generator to return outputs with the same number
        # of dimensions as our vocab.
        return self.generator(outs)

    # This exposes the TransformerEncoder that's in our Transformer
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.tok_emb(src)), src_mask)

    # This exposes the TransformerDecoder that's in our Transformer
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tok_emb(tgt)), memory,
                          tgt_mask)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        (feats, labels, lengths) = x

        seq_len = feats.shape[1]
        idx = [i for i, v in enumerate(lengths) if v == seq_len]
        x = torch.cat([feats[idx], labels[idx]], dim=-1)

        
        # The Transformer module expects the batch dimension
        # to be the first dimesion, and the seq len to be the
        # zeroth.
        src = x.permute(1,0, -1)
        # tgt = y.permute(1,0)
        tgt = y
        print(src.shape, tgt.shape)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt)
        
        logits = self(src, tgt, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        return logits.reshape(-1, logits.shape[-1]), tgt[1:, :].reshape(-1)

    def training_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        loss = self.loss_fn(pred, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        loss = self.loss_fn(pred, y)
        return loss

    def configure_optimizers(self):
        
        optimizer = Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        
        return optimizer


class Transformer(LightningModule):
    def __init__(self,
                input_dim,
                hidden_dim,
                output_dim,
                pred_len,
                n_layers,
                criterion, 
                lr, 
                amsgrad):

        super().__init__()

        self.in_dim = input_dim
        self.hid_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = output_dim
        self.pred_len = pred_len

        self.criterion = criterion
        self.lr = lr
        self.amsgrad = amsgrad

        self.pc_err = MAPE()
        self.abs_err = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

        self.rnncell = nn.GRUCell(input_dim, hidden_dim)



    def get_metrics(self, pred, y):
        metrics = dict(
            abs_err=self.abs_err(pred, y),
            pc_err=self.pc_err(pred, y),
            mse=self.mse(pred, y),
        )
        return metrics


    def validation_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        metrics = self.get_metrics(pred, y)
        self.log_dict(
            {f'{k}/validation': v for k, v in metrics.items()},
            on_epoch=True, on_step=False
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            amsgrad=self.amsgrad
            )
        return optimizer