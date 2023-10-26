'''
Author: your name
Date: 2022-04-17 14:57:54
LastEditTime: 2022-11-24 17:13:43
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /chengdongzhou/federatedLearning/ProtoHARv2/FLAlgorithms/servers/serveravg copy.py
'''
import torch
import os
import copy
from FLAlgorithms.users.userFedRep import UserFedRep
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
from utils.decorator import LoggingUserAcc

# Implementation for FedAvg Server

class FedRep(Server):
    def __init__(self, args ,model , times):
        super().__init__(args, model , times)
        # Initialize data for all  users
        self.SetClient(args,UserFedRep)

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
            self.send_parameters()

            self.evaluate()
            if glob_iter == 0:
                print('==>make global model trans to base')
                self.model = copy.deepcopy(self.model.base)
                print(self.model)
            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs)
            self.aggregate_parameters()

        self.save_results()
        self.save_model()
        
    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
    def add_parameters(self, user, ratio):
        for server_param, user_param in zip(self.model.parameters(), user.get_base_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio