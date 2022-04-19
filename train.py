from dataloaders import get_iterators
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import utilities
from models.MLP import MLP
from models.GRU import GRU
from models.LSTM import LSTM
from models.Seq2seq import FC_out, Seq_to_seq
from argparse import ArgumentParser
import os

def main(args, avail_gpus):
    utilities.seed.seed_everything(seed=args.seed)

    (train_dl, val_dl, test_dl), norm_constants = get_iterators(
        datapath=args.datapath
        batch_size=args.batch_size,
        historical_len=args.historical_len,
        pred_len=args.pred_len,
        n_workers=args.n_workers
    )

    if args.model == 'mlp':
        core_model = MLP(
            input_dim=args.historical_len*args.input_dim,
            output_dim=args.output_dim,
            hidden_dim=args.hidden_dim,
            pred_len=args.pred_len,
            n_layers=args.n_layers,
            )
    elif args.model == 'gru':
        core_model = GRU(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            pred_len=args.pred_len,
            n_layers=args.n_layers,
        )
    elif args.model == 'lstm':
        core_model = LSTM(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            pred_len=args.pred_len,
            n_layers=args.n_layers,
        )
    else:
        raise Exception('Model Not Found')

    fc_out = FC_out(
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    )

    Model = Seq_to_seq(
        core_model=core_model,
        fc_out=fc_out,
        criterion=args.criterion,
        lr=args.lr,
        amsgrad=args.amsgrad,
        norm_constants=norm_constants
    )

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
        Model,
        train_dl,
        val_dl
        )

if __name__ == '__main__':
    AVAIL_GPUS = 0

    parser = ArgumentParser()
    parser.add_argument("--model", default='gru', type=str, choices=['gru', 'mlp', 'lstm'])
    parser.add_argument("--n_layers", default=1, type=int)
    parser.add_argument("--input_dim", default=19, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--output_dim", default=4, type=int)

    parser.add_argument("--historical_len", default=4, type=int)
    parser.add_argument("--pred_len", default=2, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)
    parser.add_argument("--criterion", default='pc_err', type=str,
                        choices=['pc_err', 'abs_err', 'mse', 'kl_div'])

    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--datapath", default='data', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_workers", default=8, type=int)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    args = parser.parse_args()

    main(args, AVAIL_GPUS)