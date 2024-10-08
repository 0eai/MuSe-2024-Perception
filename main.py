import argparse
import os
import random
import sys
from datetime import datetime

import numpy
import torch
from dateutil import tz
from torch import nn

import config
from config import TASKS, PERCEPTION, HUMOR, ENCODERS
from data_parser import load_data
from dataset import MuSeDataset, custom_collate_fn
from eval import evaluate, calc_auc, calc_pearsons
from model import Model
from train import train_model
from utils import Logger, seed_worker, log_results

import wandb



def parse_args():

    parser = argparse.ArgumentParser(description='MuSe 2024.')

    parser.add_argument('--task', type=str, required=True, choices=TASKS,
                        help=f'Specify the task from {TASKS}.')
    parser.add_argument('--feature', required=True,
                        help='Specify the features used (only one).')
    parser.add_argument('--label_dim', default="assertiv", choices=config.PERCEPTION_LABELS)
    parser.add_argument('--normalize', action='store_true',
                        help='Specify whether to normalize features (default: False).')
    parser.add_argument('--encoder', type=str, required=True, # choices=ENCODERS,
                        help=f'Specify the model encoder from {ENCODERS}.')
    parser.add_argument('--model_dim', type=int, default=64,
                        help='Specify the number of hidden states in the RNN (default: 64).')
    parser.add_argument('--encoder_n_layers', type=int, default=1,
                        help='Specify the number of layers for the RNN/Transformer (default: 1).')
    parser.add_argument('--rnn_bi', action='store_true',
                        help='Specify whether the RNN is bidirectional or not (default: False).')
    parser.add_argument('--d_fc_out', type=int, default=64,
                        help='Specify the number of hidden neurons in the output layer (default: 64).')
    
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads (default: 8).') # TODO
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                        help='Dimension of the feedforward network (default: 2048).') # TODO

    parser.add_argument('--encoder_dropout', type=float, default=0.2)
    parser.add_argument('--linear_dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Specify the number of epochs (default: 100).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Specify the batch size (default: 256).')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Specify initial learning rate (default: 0.0001).')
    parser.add_argument('--seed', type=int, default=101,
                        help='Specify the initial random seed (default: 101).')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Specify number of random seeds to try (default: 5).')
    parser.add_argument('--result_csv', default=None, help='Append the results to this csv (or create it, if it '
                                                           'does not exist yet). Incompatible with --predict')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Specify whether to use gpu for training (default: False).')
    parser.add_argument('--cache', action='store_true',
                        help='Specify whether to cache data as pickle file (default: False).')
    parser.add_argument('--predict', action='store_true',
                        help='Specify when no test labels are available; test predictions will be saved '
                             '(default: False). Incompatible with result_csv')
    parser.add_argument('--regularization', type=float, required=False, default=0.0,
                        help='L2-Penalty')
    # evaluation only arguments
    parser.add_argument('--eval_model', type=str, default=None,
                        help='Specify model which is to be evaluated; no training with this option (default: False).')
    parser.add_argument('--eval_seed', type=str, default=None,
                        help='Specify seed to be evaluated; only considered when --eval_model is given.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Specify device')
    
    args = parser.parse_args()
    if not (args.result_csv is None) and args.predict:
        print("--result_csv is not compatible with --predict")
        sys.exit(-1)
    if args.eval_model:
        assert args.eval_seed
    return args



    #####################
    # Auto Load on Free GPU
    #####################

def get_free_gpu():
    # Get the number of GPUs
    num_gpus = torch.cuda.device_count()

    # If no GPUs are available, return None
    if num_gpus == 0:
        return None

    # Get the memory usage of each GPU
    free_memory = []
    for i in range(num_gpus):
        properties = torch.cuda.get_device_properties(i)
        free_memory.append(properties.total_memory - torch.cuda.memory_allocated(i))

    # Select the GPU with the most free memory
    free_gpu = free_memory.index(max(free_memory))

    return free_gpu

    # Get the best GPU
    
def get_loss_fn(task):
    if task == HUMOR:
        return nn.BCELoss(), 'Binary Crossentropy'
    elif task == PERCEPTION:
        return nn.MSELoss(reduction='mean'), 'MSE'


def get_eval_fn(task):
    if task == PERCEPTION:
        return calc_pearsons, 'Pearson'
    elif task == HUMOR:
        return calc_auc, 'AUC'


def main(args):
    # ensure reproducibility
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # emo_dim only relevant for stress/personalisation
    args.label_dim = args.label_dim if args.task==PERCEPTION else ''
    print('Loading data ...')
    args.paths['partition'] = os.path.join(config.PATH_TO_METADATA[args.task], f'partition.csv')

    data = load_data(args.task, args.paths, args.feature, args.label_dim, args.normalize, save=args.cache)
    datasets = {partition:MuSeDataset(data, partition) for partition in data.keys()}

    args.d_in = datasets['train'].get_feature_dim()

    args.n_targets = config.NUM_TARGETS[args.task]
    args.n_to_1 = args.task in config.N_TO_1_TASKS

    loss_fn, loss_str = get_loss_fn(args.task)
    eval_fn, eval_str = get_eval_fn(args.task)

    if args.eval_model is None:  # Train and validate for each seed
        seed = args.seed
        # seeds = range(args.seed, args.seed + args.n_seeds)    ## remove seed iteration since sweep agent will select it.
        val_losses, val_scores, best_model_files, test_scores = [], [], [], []

        # for seed in seeds:
        torch.manual_seed(seed)
        data_loader = {}
        for partition,dataset in datasets.items():  # one DataLoader for each partition
            batch_size = args.batch_size if partition == 'train' else 2 * args.batch_size
            shuffle = True if partition == 'train' else False  # shuffle only for train partition
            data_loader[partition] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                                    num_workers=4,
                                                                    worker_init_fn=seed_worker,
                                                                    collate_fn=custom_collate_fn)
        
        model = Model(args)
        print(model) # TODO delete

        print('=' * 50)
        print(f'Training model... [seed {seed}] for at most {args.epochs} epochs')

        val_loss, val_score, best_model_file = train_model(args.task, model, data_loader, args.epochs,
                                                            args.lr, args.paths['model'], seed, use_gpu=args.use_gpu,
                                                            loss_fn=loss_fn, eval_fn=eval_fn,
                                                            eval_metric_str=eval_str,
                                                            regularization=args.regularization,
                                                            early_stopping_patience=args.early_stopping_patience, device=args.device)
        # restore best model encountered during training
        model = torch.load(best_model_file)

        if not args.predict:  # run evaluation only if test labels are available
            test_loss, test_score = evaluate(args.task, model, data_loader['test'], loss_fn=loss_fn,
                                                eval_fn=eval_fn, use_gpu=args.use_gpu, device=args.device)
            test_scores.append(test_score)
            print(f'[Test {eval_str}]:  {test_score:7.4f}')

    else:  # Evaluate existing model (No training)
        model_file = os.path.join(args.paths['model'], f'model_{args.eval_seed}.pth')
        model = torch.load(model_file, map_location=torch.device(args.device) if torch.cuda.is_available()
        else torch.device('cpu'))
        data_loader = {}
        for partition, dataset in datasets.items():  # one DataLoader for each partition
            batch_size = args.batch_size if partition == 'train' else 2 * args.batch_size
            shuffle = True if partition == 'train' else False  # shuffle only for train partition
            data_loader[partition] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                                 num_workers=4,
                                                                 worker_init_fn=seed_worker,
                                                                 collate_fn=custom_collate_fn)
        _, valid_score = evaluate(args.task, model, data_loader['devel'], loss_fn=loss_fn, eval_fn=eval_fn,
                                  use_gpu=args.use_gpu, device=args.device)
        print(f'Evaluating {model_file}:')
        print(f'[Val {eval_str}]: {valid_score:7.4f}')
        if not args.predict:
            _, test_score = evaluate(args.task, model, data_loader['test'], loss_fn=loss_fn, eval_fn=eval_fn,
                                     use_gpu=args.use_gpu, device=args.device)
            print(f'[Test {eval_str}]: {test_score:7.4f}')

    if args.predict:  # Make predictions for the test partition; this option is set if there are no test labels
        print('Predicting devel and test samples...')
        best_model = torch.load(model_file, map_location=config.device)
        evaluate(args.task, best_model, data_loader['devel'], loss_fn=loss_fn, eval_fn=eval_fn,
                 use_gpu=args.use_gpu, predict=True, prediction_path=args.paths['predict'],
                 filename='predictions_devel.csv', device=args.device)
        evaluate(args.task, best_model, data_loader['test'], loss_fn=loss_fn, eval_fn=eval_fn,
                 use_gpu=args.use_gpu, predict=True, prediction_path=args.paths['predict'], filename='predictions_test.csv', device=args.device)
        print(f'Find predictions in {os.path.join(args.paths["predict"])}')

    print('Done.')


if __name__ == '__main__':
    print("Start",flush=True)
    args = parse_args()

    if args.use_gpu:
        device_id = get_free_gpu()
        args.device = 'cuda'
        # args.device = f'cuda:{device_id}'
        
    ## initial wandb. A project name will follow the given name in .yaml
    wandb.init()

    ## update selected hyperparameters by sweep agent into args for compatibility between original code and wandb
    for key, value in wandb.config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    args.log_file_name =  '{}_{}_[{}]_[{}_{}_{}_{}]_[{}_{}]_{}'.format(args.encoder, datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M"), args.feature.replace(os.path.sep, "-"),
                                                 args.model_dim, args.encoder_n_layers, args.rnn_bi, args.d_fc_out, args.lr,args.batch_size, args.seed)

    ## change the name of this run
    wandb.run.name = args.log_file_name
    wandb.run.save()

    # adjust your paths in config.py
    task_id = args.task if args.task != PERCEPTION else os.path.join(args.task, args.label_dim)
    args.paths = {'log': os.path.join(config.LOG_FOLDER, task_id) if not args.predict else os.path.join(config.LOG_FOLDER, task_id, 'prediction'),
                  'data': os.path.join(config.DATA_FOLDER, task_id),
                  'model': os.path.join(config.MODEL_FOLDER, task_id, args.log_file_name if not args.eval_model else args.eval_model)}
    if args.predict:
        if args.eval_model:
            args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, task_id, args.eval_model, args.eval_seed)
        else:
            args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, task_id, args.log_file_name)

    for folder in args.paths.values():
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    args.paths.update({'features': config.PATH_TO_FEATURES[args.task],
                       'labels': config.PATH_TO_LABELS[args.task],
                       'partition': config.PARTITION_FILES[args.task]})

    sys.stdout = Logger(os.path.join(args.paths['log'], args.log_file_name + '.txt'))
    print(' '.join(sys.argv))

    main(args)
