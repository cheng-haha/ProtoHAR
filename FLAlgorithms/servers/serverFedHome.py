
import torch
import os

from FLAlgorithms.users.userhome import UserFedHome
from FLAlgorithms.servers.serverbase import Server
import numpy as np
from torch.utils.data import DataLoader
from utils.decorator import LoggingUserAcc

class FedHome(Server):
    def __init__(self, args ,model , times):
        super().__init__(args, model , times)

        self.SetClient(args,UserFedHome)

        self.per_epochs      = args.fine_epochs
        self.num_glob_iters -= self.per_epochs

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)
    @LoggingUserAcc
    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            self.logging.info(f"-------------Global Round Number:{glob_iter} -------------")
            # Evaluate model each interation
            self.evaluate()
            self.selected_users ,self.user_idxs = self.select_users(glob_iter,self.num_users,return_idx=True)
            for user in self.selected_users:
                user.train(self.local_epochs) #* user.train_samples
            self.aggregate_parameters()
            self.send_parameters()

        #NOTE Fine-tuning personalized models once you have a stable global model
        for user in self.users:
            user.generate_data()
        for per_glob_iter in range(self.per_epochs):
            self.logging.info(f"----------Personalized Round Number:{per_glob_iter}-----------")
            # Personalized evaluation is performed in the last self.per_epochs epochs
            self.evaluate()
            for user in self.users:
                user.train_pred()
        self.save_results()
        self.save_model()
