import torch.optim
import torch.nn as nn

class LSTM(torch.nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim,
        output_dim,
        pred_len,
        n_layers,
        ):
        super(LSTM, self).__init__()
        self.in_dim = input_dim
        self.hid_dim = hidden_dim
        self.out_dim = output_dim
        self.pred_len = pred_len
        self.n_layers = n_layers
        
        self.encoder = nn.LSTM(input_dim, hidden_dim,
                              num_layers=n_layers, batch_first=True)
        self.decoder = nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, X, fc_out, y):
        (feats, labels) = X
        batch_size = labels.shape[0]
        seq_len = labels.shape[1]

        prior_features = feats[:, :seq_len]
        post_features = feats[:, seq_len:]

        x = torch.cat([prior_features, labels], dim=-1)
        c = torch.zeros(batch_size, self.hid_dim)
        h = self.encoder(x)[0][:, -1]
        ps_labels = fc_out(h)
        out_total = [ps_labels]

        for i in range(self.pred_len - 1):
            x = torch.cat([post_features[:, i], ps_labels], dim=-1)
            h, c = self.decoder(x, (h,c))
            ps_labels = fc_out(h)
            out_total.append(ps_labels)

        return torch.stack(out_total)
