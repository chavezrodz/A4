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
                norm_constants,
                scale):

        super().__init__()
        self.core_model = core_model
        self.fc_out = fc_out

        self.criterion = criterion
        self.lr = lr
        self.amsgrad = amsgrad
        self.norm_constants = norm_constants
        self.scale = scale

        self.pc_err = MAPE()
        self.abs_err = nn.L1Loss()
        self.mse = nn.MSELoss()
        # self.kl = nn.KLDivLoss()


    def forward(self, X, y):
        return self.core_model(X, self.fc_out, y)

    def scale_array(self, arr, which='features'):
        mean = self.norm_constants[which]['mean'].type_as(arr)
        scale = self.norm_constants[which][self.scale].type_as(arr)
        arr = arr - mean
        arr = arr/scale
        return arr

    def unscale_arr(self, arr, which='labels'):
        mean = self.norm_constants[which]['mean'].type_as(arr)
        scale = self.norm_constants[which][self.scale].type_as(arr)
        arr = arr*scale
        arr = arr + mean
        return arr

    def predict_step(self, batch, batch_idx):
        """
        Returns batch x pred_len x out_dim
        """
        x, y = batch
        (feats, labels, lengths) = x
        # Removing incomplete batch elements
        seq_len = feats.shape[1]
        idx = [i for i, v in enumerate(lengths) if v == seq_len]
        feats, labels, y = feats[idx], labels[idx], y[:, idx]
        batch_size = feats.shape[0]

        # making it batch x pred x outdim
        y = y.permute(1, 0, 2)
        # scaling inputs
        x = (
            self.scale_array(feats, which='features'),
            self.scale_array(labels, which='labels')
            )
        out = self(x, y)
        assert out.shape[1] == batch_size
        return out.permute(1, 0, 2), y

    def get_metrics(self, pred, y):
        metrics = dict(
            abs_err=self.abs_err(pred, y),
            pc_err=self.pc_err(pred, y),
            mse=self.mse(pred, y),
            # kl_div=self.kl(pred, y)
        )
        return metrics

    def training_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        batch_size = pred.shape[0]
        y = self.scale_array(y, which='labels')
        metrics = self.get_metrics(pred.flatten(-2, -1), y.flatten(-2, -1))
        self.log_dict(
            {f'{k}/train': v for k, v in metrics.items()},
            on_epoch=True, on_step=False, batch_size=batch_size
            )
        return metrics[self.criterion]

    def validation_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        batch_size = pred.shape[0]
        y = self.scale_array(y, which='labels')
        metrics = self.get_metrics(pred.flatten(-2, -1), y.flatten(-2, -1))
        self.log_dict(
            {f'{k}/validation': v for k, v in metrics.items()},
            on_epoch=True, on_step=False, batch_size=batch_size
            )

    def test_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        batch_size = pred.shape[0]

        pred_unscaled = self.unscale_arr(pred, which='labels')
        for i, feature in enumerate(['p (mbar)', 'T (degC)', 'rh (%)', 'wv (m/s)']):
            metrics = self.get_metrics(pred_unscaled[:, :, i], y[:, :, i])
            self.log_dict(
                {f'{feature}/{k}': v for k, v in metrics.items()},
                on_epoch=True, on_step=False, batch_size=batch_size
                )

        metrics = self.get_metrics(pred_unscaled.flatten(-2, -1), y.flatten(-2, -1))
        self.log_dict(
            {f'Total_Unscaled/{k}': v for k, v in metrics.items()},
            on_epoch=True, on_step=False, batch_size=batch_size
            )

        y = self.scale_array(y, which='labels')
        metrics = self.get_metrics(pred.flatten(-2, -1), y.flatten(-2, -1))
        self.log_dict(
            {f'Total_scaled/{k}': v for k, v in metrics.items()},
            on_epoch=True, on_step=False, batch_size=batch_size
            )



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            amsgrad=self.amsgrad
            )
        return optimizer
