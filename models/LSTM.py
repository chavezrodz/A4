import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_forecasting.metrics import MAPE


class LSTM(LightningModule):
    def __init__(self,
                input_dim,
                hidden_dim,
                output_dim,
                n_layers,
                criterion, 
                lr, 
                amsgrad):

        super().__init__()

        self.in_dim = input_dim
        self.hid_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = output_dim

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

        self.rnncell = nn.LSTMCell(input_dim, hidden_dim)

        self.criterion = criterion
        self.lr = lr
        self.amsgrad = amsgrad

        self.pc_err = MAPE()
        self.abs_err = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, X):
        """
        X : (node_features, edge_features, labels_x)
        """
        (feats, labels, lengths) = X
        seq_len = feats.shape[1]
        batch_size = feats.shape[0]

        x = torch.cat([feats, labels], dim=-1)
        h = torch.zeros(batch_size, self.hid_dim)
        c = torch.zeros(batch_size, self.hid_dim)
        out_total= []
        for i in range(seq_len):
            h, c = self.rnncell(x[:, i], (h, c))
            out = self.fc_out(h)
            out_total.append(out)


        out_total = torch.stack(out_total)
        out_total = torch.stack([out_total[lengths[j]-1, j]
                        for j in range(batch_size)])

        return out_total.squeeze()

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x), y

    def get_metrics(self, pred, y):
        metrics = dict(
            abs_err=self.abs_err(pred, y),
            pc_err=self.pc_err(pred, y),
            mse=self.mse(pred, y),
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