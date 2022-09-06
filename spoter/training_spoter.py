import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from spoter.spoter_model import SPOTER
from spoter.gaussian_noise import GaussianNoise
from Src.Lsp_dataset import LSP_Dataset
from utils import __balance_val_split, __split_of_train_sequence
from spoter.utils import train_epoch, evaluate, my_evaluate, evaluate_top_k

class TrainingSpoter():
    def __init__(
        self,
        config,
        training_set_path,
        testing_set_path,
        validation_set_path,
        use_wandb=True,
    ):
        print("Starting training ...")
        self.use_wandb = use_wandb
        self.training_set_path = training_set_path
        self.testing_set_path = testing_set_path
        self.validation_set_path = validation_set_path
        self.config = config
        n_cuda = os.getenv('N_CUDA')if os.getenv('N_CUDA') else "0"
        print(f"Ncuda = {n_cuda}")
        self.device = torch.device("cuda:" + (n_cuda) if torch.cuda.is_available() else "cpu")
        self.slrt_model = SPOTER(num_classes=self.config.num_classes, 
                                hidden_dim=self.config.hidden_dim
                                )

    def get_dataset(
        self,
        path_save_weights
    ):
        g = torch.Generator()
        transform = transforms.Compose([GaussianNoise(self.config.gaussian_mean, self.config.gaussian_std)])
        train_set = LSP_Dataset(self.training_set_path,
                                self.config.keypoints_model, 
                                transform=transform, 
                                augmentations=False,
                                keypoints_number=self.config.keypoints_number
                                )

        print('train_set',len(train_set.data))
        print('train_set',train_set.data[0].shape)

        #TO-DO: Save Encoders
        name_encoder = "dict_labels_dataset.json"
        name_inv_encoder = "inv_dict_labels_dataset.json"
        path_encoder = os.path.join(path_save_weights, name_encoder)
        path_inv_encoder = os.path.join(path_save_weights, name_inv_encoder)
        with open(path_encoder, 'w') as f:
            json.dump(train_set.dict_labels_dataset, f)
        with open(path_inv_encoder, 'w') as f:
            json.dump(train_set.inv_dict_labels_dataset, f)

        if self.use_wandb:
            wandb.save(path_encoder)
            wandb.save(path_inv_encoder)

        print("Training dict encoder"+ "\n" +str(train_set.dict_labels_dataset)+ "\n")

        print("Training inv dict decoder"+ "\n" +str(train_set.inv_dict_labels_dataset)+ "\n")


        # Validation set
        if self.config.validation_set == "from-file":
            val_set = LSP_Dataset(self.validation_set_path,self.config.keypoints_model,
                                dict_labels_dataset=train_set.dict_labels_dataset,
                                inv_dict_labels_dataset = train_set.inv_dict_labels_dataset,keypoints_number = self.config.keypoints_number)
            val_loader = DataLoader(val_set, shuffle=True, generator=g)

        elif self.config.validation_set == "split-from-train":
            train_set, val_set = __balance_val_split(train_set, 0.2)

            val_set.transform = None
            val_set.augmentations = False
            val_loader = DataLoader(val_set, shuffle=True, generator=g)

        else:
            val_loader = None

        # Testing set
        if self.testing_set_path:
            #eval_set = CzechSLRDataset(self.testing_set_path)
            eval_set = LSP_Dataset(self.testing_set_path,self.config.keypoints_model,
                                dict_labels_dataset=train_set.dict_labels_dataset,
                                inv_dict_labels_dataset = train_set.inv_dict_labels_dataset,keypoints_number = self.config.keypoints_number)
            eval_loader = DataLoader(eval_set, shuffle=True, generator=g)

        else:
            eval_loader = None

        # Final training set refinements
        if self.config.experimental_train_split:
            train_set = __split_of_train_sequence(train_set, self.config.experimental_train_split)

        train_loader = DataLoader(train_set, shuffle=True, generator=g)
        
        print('train_loader',len(train_loader))

        if self.config.experimental_train_split:
            print("Starting " + self.config.weights_trained + "_" + str(self.config.experimental_train_split).replace(".", "") + "...\n\n")
        else:
            print("Starting " + self.config.weights_trained + "...\n\n")

        return train_loader, val_loader, eval_loader


    def save_weights(self, model, path_sub, use_wandb=True):

        torch.save(model.state_dict(), os.path.join(path_sub, 'spoter-sl.pth'))

        if use_wandb:
            wandb.save(os.path.join(path_sub, '*.pth'),
                    base_path='/'.join(path_sub.split('/')[:-2]))


    def train(
        self
    ):
        if torch.cuda.is_available():
            print("Training in " + torch.cuda.get_device_name(0))  
        else:
            print("Training in CPU")

        if self.use_wandb:
            path_save_weights = os.path.join(self.config.save_weights_path, wandb.run.id + "_" + self.config.weights_trained)
        else:
            path_save_weights = os.path.join(self.config.save_weights_path, self.config.weights_trained)
        try:
            os.mkdir(path_save_weights)
        except OSError:
            pass

        self.slrt_model.train(True)
        self.slrt_model.to(self.device)

        cel_criterion = nn.CrossEntropyLoss()
        sgd_optimizer = optim.SGD(self.slrt_model.parameters(), 
                                    lr=self.config.lr
                                    )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, 
                                                        factor=self.config.scheduler_factor, 
                                                        patience=self.config.scheduler_patience
                                                        )

        train_loader, val_loader, eval_loader = self.get_dataset(path_save_weights)

        for epoch in tqdm(range(self.config.epochs)):
            train_loss, _, _, train_acc = train_epoch(self.slrt_model, train_loader, cel_criterion, sgd_optimizer, self.device)

            metrics_log = {"train_epoch": epoch+1,
                        "train_loss": train_loss,
                        "train_acc": train_acc
            }

            print("Metrics")
            print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch +
                                                            1, self.config.epochs, train_loss))
            print('Epoch [{}/{}], train_acc: {:.4f}'.format(epoch +
                                                            1, self.config.epochs, train_acc))

            if val_loader:
                self.slrt_model.train(False)
                _, _, val_acc = evaluate(self.slrt_model, val_loader, self.device)
                self.slrt_model.train(True)
                metrics_log["val_acc"] = val_acc

                print('Epoch [{}/{}], val_acc: {:.4f}'.format(epoch +
                                                1, self.config.epochs, val_acc))

            if eval_loader:
                self.slrt_model.train(False)
                _, _, eval_acc = evaluate(self.slrt_model, eval_loader, self.device, print_stats=True)
                _, _, eval_acctop5 = evaluate_top_k(self.slrt_model, eval_loader, self.device, k=5)
                self.slrt_model.train(True)
                metrics_log["eval_acc"] = eval_acc
                metrics_log["eval_acctop5"] = eval_acctop5

                print('Epoch [{}/{}], eval_acc: {:.4f}'.format(epoch +
                                                1, self.config.epochs, eval_acc))
                print('Epoch [{}/{}], eval_acctop5: {:.4f}'.format(epoch +
                                                1, self.config.epochs, eval_acctop5))
                

            if ((epoch+1) % int(self.config.epochs/self.config.num_backups)) == 0:
                path_save_epoch = os.path.join(path_save_weights, 'epoch_{}'.format(epoch+1))
                try:
                    os.mkdir(path_save_epoch)
                except OSError:
                    pass
                self.save_weights(self.slrt_model, path_save_epoch, self.use_wandb)

            if self.use_wandb:
                wandb.log(metrics_log)
        
        if self.use_wandb:
            wandb.finish()