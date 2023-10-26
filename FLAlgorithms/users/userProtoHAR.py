'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-09-01 21:01:12
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-11-24 16:37:31
FilePath: /chengdongzhou/federatedLearning/ProtoHARv2/FLAlgorithms/users/userProtoHAR copy.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from itertools import count
import torch
from FLAlgorithms.users.userbase import User
import torch.nn as nn
import copy
from torch.optim import SGD
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from FLAlgorithms.metrics.metrics import MMD
import time
from collections import Counter

class UserProtoHAR(User):
    def __init__(self, args, id, model, train_data, test_data):
        super().__init__(args, id, model, train_data, test_data)
        self.loss_mse = nn.MSELoss()
        self.loss = nn.CrossEntropyLoss()
        self.cos = nn.CosineSimilarity(-1)
        self.distance = args.distance
        self.tau = args.tau
        # beta for weighting
        self.Epsilon =args.beta
        self.optimizer = SGD(self.model.base.parameters(), lr=self.learning_rate,momentum=args.momentum)
        self.poptimizer = SGD(self.model.predictor.parameters(), lr=self.personal_learning_rate,momentum=args.momentum)
        self.plocal_steps = args.plocal_steps
        self.exe_weighting_prototypes = args.weighting
        self.inherit_agg_proto = {}
        for data,targets in self.trainloaderfull:
            self.label_size = Counter(targets.cpu().numpy())


    def train(self, glob_iter,global_protos,proto_size = 100):
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

        agg_protos_label      = {}
        agg_protos_simliratiy = {}

        for epoch in range(self.local_epochs):
            self.model.train()
            for X,y in self.trainloader:
                X, y    = X.to(self.device), y.to(self.device)
                label_g = y
                self.optimizer.zero_grad()
                result  = self.model(X)
                output  , protos = result['output'] , result['proto']

                loss1       = self.loss(output, y)
                proto_new   = copy.deepcopy(protos.data)
                if len(global_protos) == 0:
                    loss2   = 0*loss1
                else:
                    i       = 0
                    for label in y:
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i   += 1

                    # Normlize prototypes
                    # proto_new = ( proto_new - proto_new.mean(0) ) / proto_new .std(0)
                    # protos = ( protos - protos.mean(0) ) / protos .std(0)

                    if self.distance == 'MMD':
                        loss2 = MMD( protos , proto_new , 'rbf' , self.device )
                    elif self.distance == 'L1':
                        loss2 = F.l1_loss(protos,proto_new)
                    elif self.distance == 'kl':
                        loss2 = F.kl_div(F.log_softmax(protos, dim=1),F.softmax(proto_new, dim=1))
                    elif self.distance == 'cos':
                        loss2 = 1 - F.cosine_similarity(protos,proto_new,dim=1)
                    else:#MSE
                        loss2 = self.loss_mse( protos , proto_new )
                

                protoloss =  loss1 + loss2 * self.lamda 

                protoloss.backward()
                nn.utils.clip_grad_norm_( self.model.parameters(), 50 )
                self.optimizer.step()
                if self.exe_weighting_prototypes:
                    self.weighting = self.cos(protos,proto_new)
                with torch.no_grad():
                    if self.exe_weighting_prototypes:
                        for i in range(len(label_g)):
                            temp_weight = ( self.Epsilon * self.weighting[i] ).exp()
                            if label_g[i].item() in agg_protos_label:
                                if len(agg_protos_label[label_g[i].item()]) <= proto_size :
                                    agg_protos_label[label_g[i].item()] += temp_weight  * protos[i,:]
                                    agg_protos_simliratiy[label_g[i].item()] = \
                                        torch.cat(
                                                [ agg_protos_simliratiy[label_g[i].item()], temp_weight ]
                                                )
                            else:
                                agg_protos_label[label_g[i].item()] = temp_weight * protos[i,:]
                                agg_protos_simliratiy[label_g[i].item()] = temp_weight
                    else:
                        for i in range(len(label_g)):
                            if label_g[i].item() in agg_protos_label:
                                if len(agg_protos_label[label_g[i].item()]) <= proto_size :
                                    agg_protos_label[label_g[i].item()] += ( 1/self.label_size[label_g[i].item()]) * protos[i,:]
                            else:
                                agg_protos_label[label_g[i].item()] = ( 1/self.label_size[label_g[i].item()]) * protos[i,:]
                                
        if self.exe_weighting_prototypes:
            self.NormPrototype(agg_protos_label,agg_protos_simliratiy)
        return agg_protos_label
        
    def NormPrototype(self,agg_protos_label,agg_protos_simliratiy):
        for label in agg_protos_label.keys():
            agg_protos_label[label] /=  agg_protos_simliratiy[label].sum()

