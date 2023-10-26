

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
from torch.optim import SGD


class UserFedRep(User):
    def __init__(self,  args, id, model, train_data, test_data):
        super().__init__( args, id, model, train_data, test_data)

        self.optimizer  = SGD(self.model.base.parameters(), lr=self.learning_rate,momentum=args.momentum)
        self.poptimizer = SGD(self.model.predictor.parameters(), lr=self.personal_learning_rate,momentum=args.momentum)

        self.plocal_steps = args.plocal_steps
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
                
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.predictor.parameters():
            param.requires_grad = True

        for epoch in range(1, self.plocal_steps + 1):
            for X,y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)
                self.poptimizer.zero_grad()
                output = self.model(X)['output']
                loss = self.loss(output, y)
                loss.backward()
                nn.utils.clip_grad_norm_( self.model.parameters(), 50 )
                self.poptimizer.step()
            
        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.predictor.parameters():
            param.requires_grad = False

        for epoch in range(1, self.local_epochs + 1):
            for X,y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)['output']
                loss = self.loss(output, y)
                loss.backward()
                nn.utils.clip_grad_norm_( self.model.parameters(), 50 )
                self.optimizer.step()

        return LOSS

        
    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

