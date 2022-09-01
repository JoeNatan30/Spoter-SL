
import os
import argparse
import random
import logging
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

from utils import __balance_val_split, __split_of_train_sequence  # type: ignore
#from datasets.czech_slr_dataset import CzechSLRDataset
from Src.Lsp_dataset import LSP_Dataset 
from spoter.spoter_model import SPOTER
from spoter.utils import train_epoch, evaluate, my_evaluate

from spoter.gaussian_noise import GaussianNoise
import json

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="lsa_64_spoter",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--num_classes", type=int, default=15, help="Number of classes to be recognized by the model")
    parser.add_argument("--hidden_dim", type=int, default=137,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")

    # Data
    parser.add_argument("--training_set_path", type=str, default="", help="Path to the training dataset CSV file")
    parser.add_argument("--keypoints_model", type=str, default="openpose", help="Path to the training dataset CSV file")
    parser.add_argument("--keypoints_number", type=int, default=29, help="Path to the training dataset CSV file")

    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")
    parser.add_argument("--experimental_train_split", type=float, default=None,
                        help="Determines how big a portion of the training set should be employed (intended for the "
                             "gradually enlarging training set experiment from the paper)")

    parser.add_argument("--validation_set", type=str, choices=["from-file", "split-from-train", "none"],
                        default="from-file", help="Type of validation set construction. See README for further rederence")
    parser.add_argument("--validation_set_size", type=float,
                        help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
                             " to 'split-from-train'")
    parser.add_argument("--validation_set_path", type=str, default="", help="Path to the validation dataset CSV file")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the model training")
    parser.add_argument("--log_freq", type=int, default=1,
                        help="Log frequency (frequency of printing all the training info)")

    # Checkpointing
    parser.add_argument("--save_checkpoints", type=bool, default=True,
                        help="Determines whether to save weights checkpoints")

    # Scheduler
    parser.add_argument("--scheduler_factor", type=int, default=0.1, help="Factor for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for the ReduceLROnPlateau scheduler")

    # Gaussian noise normalization
    parser.add_argument("--gaussian_mean", type=int, default=0, help="Mean parameter for Gaussian noise layer")
    parser.add_argument("--gaussian_std", type=int, default=0.001,
                        help="Standard deviation parameter for Gaussian noise layer")

    # Visualization
    parser.add_argument("--plot_stats", type=bool, default=True,
                        help="Determines whether continuous statistics should be plotted at the end")
    parser.add_argument("--plot_lr", type=bool, default=True,
                        help="Determines whether the LR should be plotted at the end")

    return parser

def create_one_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def create_folder(directory):
    path = directory.split('/')
    total_path =''
    for i in path:
        total_path = os.path.join(total_path,i)
        create_one_folder(total_path)

def train(args):
    

    create_folder('out-img')
    create_folder('out-checkpoints')
    # MARK: TRAINING PREPARATION AND MODULES

    # Initialize all the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(args.seed)
    print('args.num_classes         :',args.num_classes)
    print('args.hidden_dim          :',args.hidden_dim)
    print('args.keypoints_model     :',args.keypoints_model)
    print('args.experiment_name     :',args.experiment_name)
    print('args.training_set_path   :',args.training_set_path)
    print('args.validation_set_path :',args.validation_set_path)    
    print('args.testing_set_path    :',args.testing_set_path)
    print('args.epochs              :',args.epochs)
    print('args.lr                  :',args.lr)
    print('args.keypoints_number    :',args.keypoints_number)
    create_folder('out-img')
    create_folder('out-img/'+args.experiment_name)
    create_folder('out-checkpoints')
    create_folder('out-checkpoints/'+args.experiment_name)
    create_folder('out-logs')
    create_folder('out-logs/'+args.experiment_name)

    # Set the output format to print into the console and save into LOG file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler('out-logs/'+args.experiment_name+'/'+args.experiment_name.split('/')[-1] + "_" + str(args.experimental_train_split).replace(".", "") + ".log")
        ]
    )

    logging.info('args.num_classes         :'+str(args.num_classes))
    logging.info('args.hidden_dim          :'+str(args.hidden_dim))
    logging.info('args.keypoints_model     :'+str(args.keypoints_model))
    logging.info('args.experiment_name     :'+str(args.experiment_name))
    logging.info('args.training_set_path   :'+str(args.training_set_path))
    logging.info('args.validation_set_path :'+str(args.validation_set_path))
    logging.info('args.testing_set_path    :'+str(args.testing_set_path))
    logging.info('args.epochs              :'+str(args.epochs))
    logging.info('args.lr                  :'+str(args.lr))
    logging.info('args.keypoints_number    :'+str(args.keypoints_number))

    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Construct the model


    
    slrt_model = SPOTER(num_classes=args.num_classes, hidden_dim=args.hidden_dim)
    slrt_model.train(True)
    slrt_model.to(device)

    # Construct the other modules
    cel_criterion = nn.CrossEntropyLoss()
    sgd_optimizer = optim.SGD(slrt_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, factor=args.scheduler_factor, patience=args.scheduler_patience)

    # Ensure that the path for checkpointing and for images both exist
    Path("out-checkpoints/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    Path("out-img/").mkdir(parents=True, exist_ok=True)


    # MARK: DATA

    # Training set
    transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    #train_set = CzechSLRDataset(args.training_set_path, transform=transform, augmentations=False)
    train_set    = LSP_Dataset(args.training_set_path,args.keypoints_model, transform=transform, augmentations=False,keypoints_number = args.keypoints_number)

    print('train_set',len(train_set.data))
    #print('train_set',train_set.data[0])
    print('train_set',train_set.data[0].shape)

    #Save encoders
    name = "out-img/" + args.experiment_name +'/'+args.experiment_name.split('/')[-1]+ "_dict_labels_dataset.json"
    

    with open(name, 'w') as f:
        json.dump(train_set.dict_labels_dataset, f)
    name = "out-img/" + args.experiment_name +'/'+args.experiment_name.split('/')[-1]+ "_inv_dict_labels_dataset.json"
    with open(name, 'w') as f:
        json.dump(train_set.inv_dict_labels_dataset, f)


    print("Training dict encoder"+ "\n" +str(train_set.dict_labels_dataset)+ "\n")
    logging.info("Training dict encoder"+ "\n" + str(train_set.dict_labels_dataset) + "\n")

    print("Training inv dict decoder"+ "\n" +str(train_set.inv_dict_labels_dataset)+ "\n")
    logging.info("Training inv dict decoder"+ "\n" + str(train_set.inv_dict_labels_dataset) + "\n")

    # Validation set
    if args.validation_set == "from-file":
        #val_set = CzechSLRDataset(args.validation_set_path)
        val_set = LSP_Dataset(args.validation_set_path,args.keypoints_model,
                             dict_labels_dataset=train_set.dict_labels_dataset,
                             inv_dict_labels_dataset = train_set.inv_dict_labels_dataset,keypoints_number = args.keypoints_number)
        val_loader = DataLoader(val_set, shuffle=True, generator=g)

    elif args.validation_set == "split-from-train":
        train_set, val_set = __balance_val_split(train_set, 0.2)

        val_set.transform = None
        val_set.augmentations = False
        val_loader = DataLoader(val_set, shuffle=True, generator=g)

    else:
        val_loader = None



    # Testing set
    if args.testing_set_path:
        #eval_set = CzechSLRDataset(args.testing_set_path)
        eval_set = LSP_Dataset(args.testing_set_path,args.keypoints_model,
                             dict_labels_dataset=train_set.dict_labels_dataset,
                             inv_dict_labels_dataset = train_set.inv_dict_labels_dataset,keypoints_number = args.keypoints_number)
        eval_loader = DataLoader(eval_set, shuffle=True, generator=g)

    else:
        eval_loader = None

    # Final training set refinements
    if args.experimental_train_split:
        train_set = __split_of_train_sequence(train_set, args.experimental_train_split)

    train_loader = DataLoader(train_set, shuffle=True, generator=g)
    
    print('train_loader',len(train_loader))


    # MARK: TRAINING
    train_acc, val_acc = 0, 0
    losses, train_accs, val_accs = [], [], []
    lr_progress = []
    top_train_acc, top_val_acc = 0, 0
    checkpoint_index_train = 0
    checkpoint_index_val   = 0

    if args.experimental_train_split:
        print("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
        logging.info("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")

    else:
        print("Starting " + args.experiment_name + "...\n\n")
        logging.info("Starting " + args.experiment_name + "...\n\n")

    for epoch in range(args.epochs):
        train_loss, _, _, train_acc = train_epoch(slrt_model, train_loader, cel_criterion, sgd_optimizer, device)
        losses.append(train_loss.item() / len(train_loader))
        train_accs.append(train_acc)

        if val_loader:
            slrt_model.train(False)
            _, _, val_acc = evaluate(slrt_model, val_loader, device)
            slrt_model.train(True)
            val_accs.append(val_acc)

        print('checkpoint_index_train :',checkpoint_index_train)
        print('checkpoint_index_val   :',checkpoint_index_val)
        # Save checkpoints if they are best in the current subset
        if args.save_checkpoints:
            if train_acc > top_train_acc:
                print("[" + str(epoch + 1) + "] Train  acc top increase to : " + str(train_acc))
                print("[" + str(epoch + 1) + "] Train  acc top save in : " + "out-checkpoints/" + args.experiment_name + "/checkpoint_t_" + str(checkpoint_index_train) + ".pth")
                logging.info("[" + str(epoch + 1) + "] Train  acc top increase to : " + str(train_acc))
                logging.info("[" + str(epoch + 1) + "] Train  acc top save in : " + "out-checkpoints/" + args.experiment_name + "/checkpoint_t_" + str(checkpoint_index_train) + ".pth")


                top_train_acc = train_acc
                torch.save(slrt_model, "out-checkpoints/" + args.experiment_name + "/checkpoint_t_" + str(checkpoint_index_train) + ".pth")
                

            if val_acc > top_val_acc:
                print("[" + str(epoch + 1) + "] Val  acc top increase to : " + str(val_acc))
                print("[" + str(epoch + 1) + "] Val  acc top save in : " + "out-checkpoints/" + args.experiment_name + "/checkpoint_v_" + str(checkpoint_index_val) + ".pth")
                logging.info("[" + str(epoch + 1) + "] Val  acc top increase to : " + str(val_acc))
                logging.info("[" + str(epoch + 1) + "] Val  acc top save in : " + "out-checkpoints/" + args.experiment_name + "/checkpoint_v_" + str(checkpoint_index_val) + ".pth")


                top_val_acc = val_acc
                torch.save(slrt_model, "out-checkpoints/" + args.experiment_name + "/checkpoint_v_" + str(checkpoint_index_val) + ".pth")


        if epoch % args.log_freq == 0:
            print("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item() / len(train_loader)) + " acc: " + str(train_acc))
            logging.info("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item() / len(train_loader)) + " acc: " + str(train_acc))

            if val_loader:
                print("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))
                logging.info("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))

            print("")
            logging.info("")

        # Reset the top accuracies on static subsets\
        '''
        if epoch % 10 == 0:
            top_train_acc, top_val_acc = 0, 0
            checkpoint_index += 1
        '''
        if epoch % 10 == 0:
            print('clean top train acc  and update checkpoint id')
            top_train_acc, top_val_acc = 0, 0
            checkpoint_index_train += 1
            checkpoint_index_val += 1
            print('checkpoint_index_train :',checkpoint_index_train)
            print('checkpoint_index_val   :',checkpoint_index_val)

        lr_progress.append(sgd_optimizer.param_groups[0]["lr"])

    # MARK: TESTING

    print("\nTesting checkpointed models starting...\n")
    logging.info("\nTesting checkpointed models starting...\n")

    top_result, top_result_name = 0, ""

    if eval_loader:
        for checkpoint_id, checkpoint_index in [["t",checkpoint_index_train],["v",checkpoint_index_val]]:
            #for checkpoint_id in ["t", "v"]:
            for i in range(checkpoint_index):
                # tested_model = VisionTransformer(dim=2, mlp_dim=108, num_classes=100, depth=12, heads=8)
                try:
                    tested_model = torch.load("out-checkpoints/" + args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i) + ".pth")
                    tested_model.train(False)
                    _, _, eval_acc = evaluate(tested_model, eval_loader, device, print_stats=True)                

                    if eval_acc > top_result:
                        top_result = eval_acc
                        top_result_name = "out-checkpoints/" + args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i)+ ".pth"

                    print("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))
                    logging.info("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))
                except:
                    pass
        print("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name)
        logging.info("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name)
        
        tested_model = torch.load(top_result_name)
        tested_model.train(False)
        my_evaluate(tested_model,train_set,train_loader,eval_loader,device,args.experiment_name,print_stats=True)
        _, _, eval_acc = evaluate(tested_model, eval_loader, device, print_stats=True)                

    # PLOT 0: Performance (loss, accuracies) chart plotting
    if args.plot_stats:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(losses) + 1), losses, c="#D64436", label="Training loss")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
        ax.grid()

        fig.savefig("out-img/" + args.experiment_name +'/'+args.experiment_name.split('/')[-1]+ "_loss.png")
    if args.plot_stats:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(train_accs) + 1), train_accs, c="#00B09B", label="Training accuracy")

        if val_loader:
            ax.plot(range(1, len(val_accs) + 1), val_accs, c="#E0A938", label="Validation accuracy")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
        ax.grid()

        fig.savefig("out-img/"+ args.experiment_name+'/' + args.experiment_name.split('/')[-1] + "_accuracy.png")

    # PLOT 1: Learning rate progress
    if args.plot_lr:
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
        ax1.set(xlabel="Epoch", ylabel="LR", title="")
        ax1.grid()

        fig1.savefig("out-img/"+ args.experiment_name+'/'  +args.experiment_name.split('/')[-1] + "_lr.png")

    print("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    logging.info("\nAny desired statistics have been plotted.\nThe experiment is finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    train(args)
