from dataloaders import get_iterators
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import utilities
from models.mlp import MLP
from models.GRU import GRU
from argparse import ArgumentParser
import os

def main(args, avail_gpus):
    utilities.seed.seed_everything(seed=args.seed)

    train_dl, val_dl = get_iterators(
        batch_size=args.batch_size,
        historical_len=args.historical_len,
        include_last=False,
        include_time=args.include_time
    )

    input_dim = 18 if args.include_time else 14

    if args.model == 'mlp':
        model = MLP(
            input_dim=args.historical_len*input_dim,
            output_dim=args.output_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            criterion=args.criterion,
            lr=args.lr,
            amsgrad=args.amsgrad
            )
    elif args.model == 'GRU':
        model = GRU(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            n_layers=args.n_layers,
            criterion=args.criterion,
            lr=args.lr,
            amsgrad=args.amsgrad
        )
    else:
        raise Exception('Model Not Found')

    logger = TensorBoardLogger(
        save_dir=os.path.join(args.results_dir, "TB_logs"),
        default_hp_metric=True
    )

    trainer = Trainer(
        logger=logger,
        gpus=avail_gpus,
        max_epochs=args.epochs,
        fast_dev_run=args.fast_dev_run
        )

    trainer.fit(
        model,
        train_dl,
        val_dl
        )

if __name__ == '__main__':
    AVAIL_GPUS = 0

    parser = ArgumentParser()
    parser.add_argument("--model", default='GRU', type=str)
    parser.add_argument("--n_layers", default=8, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--output_dim", default=4, type=int)

    parser.add_argument("--include_time", default=True, type=bool)
    parser.add_argument("--historical_len", default=4, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)
    parser.add_argument("--criterion", default='abs_err', type=str,
                        choices=['pc_err', 'abs_err', 'mse'])

    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    args = parser.parse_args()

    main(args, AVAIL_GPUS)