import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_forecasting.metrics import MAPE


class MLP(LightningModule):
    def __init__(
        self,
        hidden_dim,
        n_layers,
        input_dim,
        output_dim,
        lr,
        amsgrad,
        criterion
    ):
        super().__init__()

        self.pc_err = MAPE()
        self.abs_err = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss()

        self.criterion = criterion
        self.lr = lr
        self.amsgrad = amsgrad

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
            )
        for i in range(n_layers):
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        (feats, labels) = x

        x = torch.cat([feats, labels], dim=-1)
        x = torch.flatten(x, -2, -1)
 
        return self.mlp(x).square()

    def predict_step(self, batch, batch_idx):
        x, y = batch
        (feats, labels, lengths) = x
        seq_len = feats.shape[1]
        idx = [i for i, v in enumerate(lengths) if v == seq_len]
        x = (feats[idx], labels[idx])
        return self(x), y[idx]

    def get_metrics(self, pred, y):
        metrics = dict(
            abs_err=self.abs_err(pred, y),
            pressure_abs_err=self.abs_err(pred[:, 0], y[:, 0]),
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