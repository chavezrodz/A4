from dataloaders import get_iterators

train_dl, val_dl = get_iterators(
    batch_size=64,
    historical_len=7,
    include_last=True
)

for (x, y) in train_dl:
    print(x[0].shape, x[1].shape, len(x[2]), y.shape)