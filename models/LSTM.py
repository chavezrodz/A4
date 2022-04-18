import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_forecasting.metrics import MAPE


class LSTM(LightningModule):
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
        self.kl = nn.KLDivLoss()

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

        self.rnncell = nn.LSTMCell(input_dim, hidden_dim)


    def forward(self, X):
        """
        X : (node_features, edge_features, labels_x)
        """
        (feats, labels) = X
        batch_size = labels.shape[0]
        seq_len = labels.shape[1]

        prior_features = feats[:, :seq_len]
        post_features = feats[:, seq_len:]

        x = torch.cat([prior_features, labels], dim=-1)
        h = torch.zeros(batch_size, self.hid_dim)
        c = torch.zeros(batch_size, self.hid_dim)

        for i in range(seq_len):
            h, c = self.rnncell(x[:, i], (h, c))
        ps_labels = self.fc_out(h)
        out_total = [ps_labels]

        for i in range(self.pred_len - 1):
            x = torch.cat([post_features[:, i], ps_labels], dim=-1)
            h, c = self.rnncell(x, (h, c))
            ps_labels = self.fc_out(h)
            out_total.append(ps_labels)

        return torch.stack(out_total)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        (feats, labels, lengths) = x
        seq_len = feats.shape[1]
        idx = [i for i, v in enumerate(lengths) if v == seq_len]
        x = (feats[idx], labels[idx])
        return self(x).flatten(-2, -1), y[:, idx].flatten(-2, -1)

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