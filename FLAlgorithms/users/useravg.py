import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
from torch.optim import SGD
# Implementation for FedAvg clients

class UserAVG(User):
    def __init__(self,  args, id, model, train_data, test_data   ):
        super().__init__( args, id, model, train_data, test_data)
        self.fineturning_epochs = 2
    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs , lr_decay = True):
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            # X, y = self.get_next_train_batch()
            for X,y in self.trainloader:
                X,y = X.to(self.device),y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)['output']
                loss = self.loss(output, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 100 )
                self.optimizer.step()
            #self.model --> self.local
            # self.clone_model_paramenter(self.model.parameters(), self.local_model)
        # if lr_decay:
        #     self.lr_scheduler.step()
        return LOSS


