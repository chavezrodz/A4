import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_forecasting.metrics import MAPE


class FC_out(torch.nn.Module):
    def __init__(
        self, 
        hidden_dim,
        output_dim,
        n_layers=1
        ):
        super(FC_out, self).__init__()
        self.hid_dim = hidden_dim
        self.out_dim = output_dim
        self.n_layers = n_layers
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, X):
        return self.fc_out(X)



class Seq_to_seq(LightningModule):  
    def __init__(self,
                core_model,
                fc_out,
                criterion, 
                lr, 
                amsgrad,
                norm_constants):

        super().__init__()
        self.core_model = core_model
        self.fc_out = fc_out

        self.criterion = criterion
        self.lr = lr
        self.amsgrad = amsgrad
        self.norm_constants = norm_constants

        self.pc_err = MAPE()
        self.abs_err = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss()


    def forward(self, X):
        return self.core_model(X, self.fc_out)

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
        # scaling inputs
        x = (self.scale_feats(feats), self.scale_labels(labels))
        return self(x), y

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
        y = self.scale_labels(y)
        metrics = self.get_metrics(pred.flatten(-2, -1), y.flatten(-2, -1))
        self.log_dict(
            {f'{k}/train': v for k, v in metrics.items()},
            on_epoch=True, on_step=False
            )
        return metrics[self.criterion]

    def validation_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        y = self.scale_labels(y)
        metrics = self.get_metrics(pred.flatten(-2, -1), y.flatten(-2, -1))
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
