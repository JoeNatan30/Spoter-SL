import argparse
from utils import parse_arguments, set_seed, configure_model
from spoter.training_spoter import TrainingSpoter
import sys
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


def train(config_file, use_wandb, run_name, run_notes,
        training_set_path, testing_set_path, validation_set_path):
    print("Starting training ...")
    set_seed(32)
    print("Starting training ...")
    config = configure_model(config_file, use_wandb)
    print("Starting training ...")
    if use_wandb:
        wandb.init(project=PROJECT_WANDB, entity=ENTITY, config=config, name=run_name, notes=run_notes)
        config = wandb.config
        wandb.watch_called = False
    print("Starting training ...")
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
        args = parse_arguments()
        use_wandb = args.wandb
        run_name = args.run_name
        run_notes = args.run_notes
        training_set_path = args.training_set_path
        testing_set_path = args.testing_set_path
        validation_set_path = args.validation_set_path
    else:
        use_wandb = True
        run_name = None
        run_notes = None
        training_set_path = parse_argument(sys.argv, '--training_set_path')
        testing_set_path = parse_argument(sys.argv, '--testing_set_path')
        validation_set_path = parse_argument(sys.argv, '--validation_set_path')

    train(CONFIG_FILENAME, use_wandb=use_wandb, run_name=run_name, run_notes=run_notes,
        training_set_path=training_set_path,
        testing_set_path=testing_set_path,
        validation_set_path=validation_set_path)