import argparse
from utils import parse_arguments_automated, set_seed, configure_model
from spoter.training_spoter import TrainingSpoter
import sys
import os
import wandb

CONFIG_FILENAME = "config.json"
PROJECT_WANDB = "spoter-sl"
ENTITY = "stevramos"

def is_there_arg(args, master_arg):
    if(master_arg in args):
        return True
    else:
        return False

def parse_argument(args, master_arg):
    try:
        if(master_arg in args and args.index(master_arg)+1<len(args)):
            arg  = args[args.index(master_arg)+1]
            return arg
    except:
        print("Something went wrong when loading the parameters, Kindly check input carefully!!!")


def train(config_file, use_wandb, run_name, run_notes):
    set_seed(32)
    config = configure_model(config_file, use_wandb)
    
    if use_wandb:
        wandb.init(project=PROJECT_WANDB, entity=ENTITY, config=config, name=run_name, notes=run_notes)
        config = wandb.config
        wandb.watch_called = False
    
    name_train_file = f"{config.dataset}--{config.keypoints_model}-Train.hdf5"
    name_test_file = f"{config.dataset}--{config.keypoints_model}-Val.hdf5"
    name_val_file = f"{config.dataset}--{config.keypoints_model}-Val.hdf5"
    training_set_path = os.path.join(config.save_weights_path, name_train_file)
    testing_set_path = os.path.join(config.save_weights_path, name_test_file)
    validation_set_path = os.path.join(config.save_weights_path, name_val_file)

    spoter_trainer = TrainingSpoter(config=config, use_wandb=use_wandb,
                                    training_set_path=training_set_path,
                                    testing_set_path=testing_set_path,
                                    validation_set_path=validation_set_path)
    print("Starting training ...")
    spoter_trainer.train()

if __name__ == '__main__':
    use_sweep = is_there_arg(sys.argv, '--use_sweep')
    print("Starting training ...")
    if not use_sweep:
        args = parse_arguments_automated()
        use_wandb = args.wandb
        run_name = args.run_name
        run_notes = args.run_notes
    else:
        use_wandb = True
        run_name = None
        run_notes = None

    train(CONFIG_FILENAME, use_wandb=use_wandb, run_name=run_name, run_notes=run_notes)