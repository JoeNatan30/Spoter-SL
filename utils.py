
import numpy as np

from collections import Counter
import random
import os
import argparse
import json

import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def __balance_val_split(dataset, val_split=0.):
    targets = np.array(dataset.targets)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets
    )

    train_dataset = Subset(dataset, indices=train_indices)
    val_dataset = Subset(dataset, indices=val_indices)

    return train_dataset, val_dataset


def __split_of_train_sequence(subset: Subset, train_split=1.0):
    if train_split == 1:
        return subset

    targets = np.array([subset.dataset.targets[i] for i in subset.indices])  # type: ignore
    train_indices, _ = train_test_split(
        np.arange(targets.shape[0]),
        test_size=1 - train_split,
        stratify=targets
    )

    train_dataset = Subset(subset.dataset, indices=[subset.indices[i] for i in train_indices])

    return train_dataset


def __log_class_statistics(subset: Subset):
    train_classes = [subset.dataset.targets[i] for i in subset.indices]  # type: ignore
    print(dict(Counter(train_classes)))


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Is cuda available?")
    if torch.cuda.is_available():
        print("Cuda available")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_arguments():
    ap = argparse.ArgumentParser()

    #WANDB ARGUMENTS
    ap.add_argument('-w', '--wandb', default=False, action='store_true',
                    help="use weights and biases")
    ap.add_argument('-n  ', '--no-wandb', dest='wandb', action='store_false',
                    help="not use weights and biases")
    ap.add_argument('-r', '--run_name', required=False, type=str, default=None,
                    help="name of the execution to save in wandb")
    ap.add_argument('-t', '--run_notes', required=False, type=str, default=None,
                    help="notes of the execution to save in wandb")

    ap.add_argument("--training_set_path", type=str, default="", help="Path to the training dataset CSV file")
    ap.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")
    ap.add_argument("--validation_set_path", type=str, default="", help="Path to the validation dataset CSV file")


    """
    #TRAINING ARGUMENTS
    ap.add_argument("--experimental_train_split", type=float, default=None,
                        help="Determines how big a portion of the training set should be employed (intended for the "
                             "gradually enlarging training set experiment from the paper)")
    ap.add_argument("--validation_set", type=str, choices=["from-file", "split-from-train", "none"],
                        default="from-file", help="Type of validation set construction. See README for further rederence")
    ap.add_argument("--validation_set_size", type=float,
                        help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
                             " to 'split-from-train'")
    ap.add_argument("--log_freq", type=int, default=1,
                        help="Log frequency (frequency of printing all the training info)")
    ap.add_argument("--save_checkpoints", type=bool, default=True,
                        help="Determines whether to save weights checkpoints")
    ap.add_argument("--scheduler_factor", type=int, default=0.1, help="Factor for the ReduceLROnPlateau scheduler")
    ap.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for the ReduceLROnPlateau scheduler")
    ap.add_argument("--gaussian_mean", type=int, default=0, help="Mean parameter for Gaussian noise layer")
    ap.add_argument("--gaussian_std", type=int, default=0.001,
                        help="Standard deviation parameter for Gaussian noise layer")
    ap.add_argument("--plot_stats", type=bool, default=True,
                        help="Determines whether continuous statistics should be plotted at the end")
    ap.add_argument("--plot_lr", type=bool, default=True,
                        help="Determines whether the LR should be plotted at the end")

    """
    

    args = ap.parse_args()

    return args


def parse_configuration(config_file):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file, 'r') as json_file:
            return json.load(json_file)
    else:
        return config_file


def configure_model(config_file, use_wandb):

    config_file = parse_configuration(config_file)

    config = dict(
        hidden_dim = config_file["hparams"]["hidden_dim"],
        num_classes = config_file["hparams"]["num_classes"],
        epochs = config_file["hparams"]["epochs"],
        num_backups = config_file["hparams"]["num_backups"],
        keypoints_model = config_file["hparams"]["keypoints_model"],
        lr = config_file["hparams"]["lr"],
        keypoints_number = config_file["hparams"]["keypoints_number"],

        experimental_train_split = config_file["hparams"]["experimental_train_split"],
        validation_set = config_file["hparams"]["validation_set"],
        validation_set_size = config_file["hparams"]["validation_set_size"],
        log_freq = config_file["hparams"]["log_freq"],
        save_checkpoints = config_file["hparams"]["save_checkpoints"],
        scheduler_factor = config_file["hparams"]["scheduler_factor"],
        scheduler_patience = config_file["hparams"]["scheduler_patience"],
        gaussian_mean = config_file["hparams"]["gaussian_mean"],
        gaussian_std = config_file["hparams"]["gaussian_std"],
        plot_stats = config_file["hparams"]["plot_stats"],
        plot_lr = config_file["hparams"]["plot_lr"],

        #training_set_path = config_file["data"]["training_set_path"],
        #validation_set_path = config_file["data"]["validation_set_path"],
        #testing_set_path = config_file["data"]["testing_set_path"],

        weights_trained = config_file["weights_trained"],
        save_weights_path = config_file["save_weights_path"]
    )

    if not use_wandb:
        config = type("configuration", (object,), config)

    return config