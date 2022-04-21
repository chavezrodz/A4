import torch.optim
import torch.nn as nn

class MLP(torch.nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim,
        output_dim,
        pred_len,
        n_layers,
        ):
        super(MLP, self).__init__()
        self.in_dim = input_dim
        self.hid_dim = hidden_dim
        self.pred_len = pred_len
        self.n_layers = n_layers

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
            )

        for i in range(n_layers):
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(nn.ReLU())


    def pred_step(self, x):
        """
        input: batch x hist_len x in_features
        out: batch x hidden
        """
        x = torch.flatten(x, -2, -1)
        x = self.mlp(x)
        return x

    def forward(self, x, fc_out, y):
        (feats, labels) = x
        seq_len = labels.shape[1]

        prior_features = feats[:, :seq_len]
        post_features = feats[:, seq_len:]

        out_total = list()
        all_ts = torch.cat([prior_features, labels], dim=-1)
        for i in range(self.pred_len):
            prev_ts = all_ts[:, i:]
            ps_labels = fc_out(self.pred_step(prev_ts))
            out_total.append(ps_labels)
            new_ts = torch.cat([post_features[:, i], ps_labels], dim=-1)
            all_ts = torch.cat([all_ts, new_ts.unsqueeze(1)], dim=1)

        return torch.stack(out_total)
