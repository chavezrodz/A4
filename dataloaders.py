import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd


class WeatherData(Dataset):
    def __init__(self,
                 features, 
                 labels,
                 include_last,
                 historical_len,
                 prediction_len=1,
                 ):
        
        self.historical_len = historical_len
        self.pred_len = prediction_len
        self.seq_len = historical_len + prediction_len
        self.include_last = include_last
        features = torch.tensor(features)
        self.labels = torch.tensor(labels)

        feat_mean = features.mean(axis=0)
        feat_sdev = features.std(axis=0)

        self.features = torch.zeros(features.shape)
        for i in range(features.shape[1]):
            self.features[:, i] = (features[:, i] - feat_mean[i])/feat_sdev[i]

    def __len__(self):
        return len(self.labels - self.seq_len)

    def __getitem__(self, idx):
        labels_x = self.labels[idx: idx + self.historical_len]
        if self.include_last:
            features = self.features[idx: idx + self.historical_len + 1]
        else:
            features = self.features[idx: idx + self.historical_len]
        labels_y = self.labels[idx + self.historical_len: idx + self.seq_len]
        return features, labels_x, labels_y


def get_iterators(
    batch_size,
    historical_len,
    include_last,
    data_file='data/weather_train.csv',
    split=0.9,
    ):

    '''
    Returns training and validation dataloaders
    '''

    df = pd.read_csv(data_file)
    df['Date Time'] = df['Date Time'].apply(lambda x: int(x[-8:-6]))
    feat_cols = [
        'Date Time', 'Tpot (K)', 'Tdew (degC)',
        'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)',
        'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)',
        'max. wv (m/s)', 'wd (deg)'
        ]
    labels_cols = ['p (mbar)', 'T (degC)', 'rh (%)', 'wv (m/s)']

    features = df[feat_cols].values
    labels = df[labels_cols].values

    dataset = WeatherData(
        include_last=include_last,
        features=features, 
        labels=labels,
        historical_len=historical_len
        )

    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    def collate_batch(batch):
        '''Takes care of padding at end of sequence
        '''
        feature_batch = [item[0] for item in batch]
        lengths = [x.shape[0] for x in feature_batch]
        feature_batch = pad_sequence(feature_batch, batch_first=True)
        labels_x_b = pad_sequence([item[1] for item in batch], batch_first=True).float()
        x = (feature_batch, labels_x_b, lengths)
        y = pad_sequence([item[2] for item in batch]).squeeze()
        return x, y

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=collate_batch,
        shuffle=True,
        num_workers=8
        )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=collate_batch,
        num_workers=8
        )

    return train_dataloader, val_dataloader
