import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_forecasting.metrics import MAPE

class Encoder(torch.nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim,
        ):
        super(Encoder, self).__init__()
        self.in_dim = input_dim
        self.hid_dim = hidden_dim
        
        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, X):
        return self.fc_in(X)

class Decoder(torch.nn.Module):
    def __init__(
        self, 
        hidden_dim,
        output_dim
        ):
        super(Decoder, self).__init__()
        self.hid_dim = hidden_dim
        self.out_dim = output_dim
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, X):
        return self.fc_out(X)


class Seq_to_seq(LightningModule):  
    def __init__(self,
                core_model,
                decoder,
                criterion, 
                lr, 
                amsgrad,
                norm_constants):

        super().__init__()
        self.core_model = core_model
        self.decoder = decoder

        self.criterion = criterion
        self.lr = lr
        self.amsgrad = amsgrad
        self.norm_constants = norm_constants

        self.pc_err = MAPE()
        self.abs_err = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss()


    def forward(self, X):
        return self.core_model(X, self.decoder)

    def scale_feats(self, feats):
        feats = feats - self.norm_constants['feats_mean']
        feats = feats/self.norm_constants['feats_std']
        return feats

    def scale_labels(self, labels):
        labels = labels - self.norm_constants['labels_mean']
        labels = labels/self.norm_constants['labels_std']
        return labels

    def unscale_labels(self, labels):
        labels = labels*self.norm_constants['labels_std']
        labels = labels + self.norm_constants['labels_mean']
        return labels

    def predict_step(self, batch, batch_idx):
        x, y = batch
        (feats, labels, lengths) = x
        # Removing incomplete batch elements
        seq_len = feats.shape[1]
        idx = [i for i, v in enumerate(lengths) if v == seq_len]
        feats, labels, y = feats[idx], labels[idx], y[:, idx]
        # scaling inputs, descale output or scale target?
        x = (self.scale_feats(feats), self.scale_labels(labels))
        pred = self(x)
        y = self.scale_labels(y)
        return pred.flatten(-2, -1), y.flatten(-2, -1)

    def get_metrics(self, pred, y):
        metrics = dict(
            abs_err=self.abs_err(pred, y),
            pc_err=self.pc_err(pred, y),
            mse=self.mse(pred, y),
            kl_div=self.kl(pred, y)
        )
        return metrics

    def training_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        metrics = self.get_metrics(pred, y)
        self.log_dict(
            {f'{k}/train': v for k, v in metrics.items()},
            on_epoch=True, on_step=False
            )
        return metrics[self.criterion]

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
