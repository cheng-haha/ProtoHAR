import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
from torch.optim import SGD
from imblearn.over_sampling import SMOTE
from collections import Counter
class UserFedHome(User):
    def __init__(self,  args, id, model, train_data, test_data   ):
        super().__init__( args, id, model, train_data, test_data)
        self.trainloader_rep = None
        self.opt_pred = SGD(self.model.predictor.parameters(), lr = self.personal_learning_rate, momentum=self.momentum )

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
            for X,y in self.trainloader:
                X,y = X.to(self.device),y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)['output']
                loss = self.loss(output, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 100 )
                self.optimizer.step()
        return LOSS

    def train_pred(self):
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            for i, (x, y) in enumerate(self.trainloader_rep):
                x, y   = x.to(self.device), y.to(self.device)
                self.opt_pred.zero_grad()
                output = self.model.predictor(x)
                loss   = self.loss(output, y)
                loss.backward()
                self.opt_pred.step()

    def generate_data(self):
        train_data_rep = []
        train_data_y = []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloaderfull):
                x, y   = x.to(self.device) , y.to(self.device)
                train_data_rep.append(self.model.base(x)['output'].detach().cpu().numpy())
                train_data_y.append(y.detach().cpu().numpy())
        train_data_rep = np.concatenate(train_data_rep, axis=0)
        train_data_y   = np.concatenate(train_data_y, axis=0)

        #NOTE that the choice here to ignore handling of sample numbers below a certain threshold is to allow SMOTE to produce data smoothly in the older version
        min_k_neighbors = 5
        useful_idxs = []
        useful_info = dict( Counter(train_data_y) )
        DelateClasses = [ k for k,v in useful_info.items() if v <= min_k_neighbors ]
        [useful_info.pop(k) for k in DelateClasses]
        
        for useful_class in list( useful_info.keys() ):
            useful_idxs = np.concatenate( ( useful_idxs, np.nonzero( train_data_y==useful_class  )[0] ) , axis=0)
        useful_idxs = useful_idxs.astype(int)
        train_data_rep = train_data_rep[useful_idxs]
        train_data_y   = train_data_y[useful_idxs]

        if len(np.unique(train_data_y)) > 1:
            smote = SMOTE()
            X, Y  = smote.fit_resample(train_data_rep, train_data_y)
        else:
            X, Y  = train_data_rep, train_data_y
        print(f'Client {self.id} data ratio: ', '{:.2f}%'.format(100*(len(Y))/len(train_data_y)))
        X_train = torch.Tensor(X).type(torch.float32)
        y_train = torch.Tensor(Y).type(torch.int64)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        self.trainloader_rep = DataLoader(train_data, self.batch_size)
