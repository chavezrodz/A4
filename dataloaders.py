import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd


def normalize_array(arr):
    arr = torch.tensor(arr)
    arr_mean = arr.mean(axis=0)
    arr_std = arr.std(axis=0)
    arr_normed = torch.zeros(arr.shape)
    for i in range(arr.shape[1]):
        arr_normed[:, i] = (arr[:, i] - arr_mean[i])/arr_std[i]
    return arr_normed

class WeatherData(Dataset):
    def __init__(self,
                 features, 
                 labels,
                 historical_len,
                 prediction_len,
                 ):
        
        self.historical_len = historical_len
        self.pred_len = prediction_len
        self.seq_len = historical_len + prediction_len

        self.features = normalize_array(features)
        self.labels = normalize_array(labels)

    def __len__(self):
        return len(self.labels - self.seq_len)

    def __getitem__(self, idx):
        labels_x = self.labels[idx: idx + self.historical_len]
        features = self.features[idx: idx + self.historical_len + self.pred_len]
        labels_y = self.labels[idx + self.historical_len: idx + self.seq_len]
        return features, labels_x, labels_y


def get_iterators(
    batch_size,
    historical_len,
    pred_len,
    data_file='data/weather_train.csv',
    split=0.9,
    ):

    '''
    Returns training and validation dataloaders
    '''

    df = pd.read_csv(data_file)

    feat_cols = [
         'Tpot (K)', 'Tdew (degC)',
        'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)',
        'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)',
        'max. wv (m/s)'
        ]

    df['day'] = df['Date Time'].apply(lambda x: float(x[0:2]))
    df['month'] = df['Date Time'].apply(lambda x: int(x[3:5]))
    df['day_normed'] = (df['day'] + df['month']*30)*2*np.pi/365.25
    df['day_x'] = np.cos(df['day_normed'])
    df['day_y'] = np.sin(df['day_normed'])

    df['hr'] = df['Date Time'].apply(lambda x: int(x[-8:-6]))*2*np.pi/24
    df['hr_x'] = np.cos(df['hr'])
    df['hr_y'] = np.sin(df['hr'])

    df['wd_x'] = np.cos(df['wd (deg)']* 2*np.pi/360)
    df['wd_y'] = np.sin(df['wd (deg)']* 2*np.pi/360)

    feat_cols += ['day_x', 'day_y', 'hr_x', 'hr_y', 'wd_x', 'wd_y']

    labels_cols = ['p (mbar)', 'T (degC)', 'rh (%)', 'wv (m/s)']

    features = df[feat_cols].values
    labels = df[labels_cols].values

    dataset = WeatherData(
        features=features, 
        labels=labels,
        historical_len=historical_len,
        prediction_len=pred_len
        )

    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    def collate_batch(batch):
        '''
        Takes care of padding at end of sequence
        '''
        feature_batch = [item[0] for item in batch]
        lengths = [x.shape[0] for x in feature_batch]
        feature_batch = pad_sequence(feature_batch, batch_first=True)
        labels_x_b = pad_sequence([item[1] for item in batch], batch_first=True).float()
        x = (feature_batch, labels_x_b, lengths)
        y = pad_sequence([item[2] for item in batch])
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
