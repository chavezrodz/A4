from dataloaders import get_iterators
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from models.mlp import MLP
from argparse import ArgumentParser
import os

def main(args, avail_gpus):

    train_dl, val_dl = get_iterators(
        batch_size=args.batch_size,
        historical_len=args.historical_len,
        include_last=False
    )

    model = MLP(
        input_dim=int(args.historical_len*14),
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        criterion=args.criterion,
        lr=args.lr,
        amsgrad=args.amsgrad
        )

    logger = TensorBoardLogger(
        save_dir=os.path.join(args.results_dir, "TB_logs"),
        default_hp_metric=True
    )

    trainer = Trainer(
        logger=logger,
        gpus=avail_gpus,
        max_epochs=args.epochs,
        )

    trainer.fit(
        model,
        train_dl,
        val_dl
        )

if __name__ == '__main__':
    AVAIL_GPUS = 0

    parser = ArgumentParser()
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--n_layers", default=6, type=int)
    parser.add_argument("--output_dim", default=4, type=int)

    parser.add_argument("--historical_len", default=4, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)
    parser.add_argument("--criterion", default='abs_err', type=str,
                        choices=['pc_err', 'abs_err', 'mse'])

    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--shuffle_dataset", default=True, type=bool)
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    main(args, AVAIL_GPUS)