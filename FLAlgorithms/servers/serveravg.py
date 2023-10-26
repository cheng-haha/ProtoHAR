import torch
import os

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_all_test_data, read_data, read_user_data
import numpy as np
from torch.utils.data import DataLoader
# Implementation for FedAvg Server
from utils.decorator import LoggingUserAcc

class FedAvg(Server):
    def __init__(self, args ,model , times):
        super().__init__(args, model , times)

        # Initialize data for all  users
        self.SetClient(args,UserAVG)

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
            self.logging.info(f"-------------Round number:{glob_iter} -------------")
            #loss_ = 0
            

            # Evaluate model each interation
            self.evaluate()
            self.selected_users ,self.user_idxs= self.select_users(glob_iter,self.num_users,return_idx=True)
            for user in self.selected_users:
                user.train(self.local_epochs) #* user.train_samples
            self.aggregate_parameters()
            self.send_parameters()

        self.save_results()
        self.save_model()
