import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.model_utils import read_user_data
import warnings
warnings.filterwarnings("ignore")
from torch.optim import SGD

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, args, id, model, train_data, test_data):
        num_classes = {
                        'ucihar': 6,
                        'pamap2': 12,
                        'wisdm': 6 ,
                        'unimib':17 ,
                        'hhar':6,
                        'uschad':12,
                        'medmnist':11,
                        'harbox':5
                        }  # [128, 256, 512, 23040]
        self.device = args.device
        
        self.dataset = args.dataset
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.num_classes = num_classes[args.dataset]
        self.personal_learning_rate = args.personal_learning_rate
        self.beta = args.beta
        self.lamda = args.lamda
        self.global_iters = args.num_global_iters
        self.local_epochs = args.local_epochs
        self.momentum = args.momentum
        # shuffle random mini-batch
        self.trainloader = DataLoader(train_data, self.batch_size,shuffle=True )
        # No shuffle mini-batch
        # self.trainloader = DataLoader(train_data, self.batch_size)
        self.testloader =  DataLoader(test_data, self.batch_size )
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        self.fineturning_epochs = args.fineturning_epochs
        self.init_loss_fn()
        self.accuracy = []
        # those parameters are for persionalized federated learing.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model = copy.deepcopy(self.model)
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.optimizer = SGD(self.model.parameters(), lr=self.learning_rate,momentum=self.momentum)
    
    def modelFlatten(self,  source):
            return torch.cat([value.flatten() for value in source.parameters()]) 

    def init_loss_fn(self):
        self.loss      = nn.CrossEntropyLoss()
        self.dist_loss = nn.MSELoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        # self.ce_loss = nn.CrossEntropyLoss()
    
    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
        #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def get_base_parameters(self):
        for param in self.model.base.parameters():
            param.detach()
        return self.model.base.parameters()
    def get_predictor_parameters(self):
        for param in self.model.predictor.parameters():
            param.detach()
        return self.model.predictor.parameters()

    def clone_model_paramenter(self, param, clone_param):
        '''
        current_model -->  clone_model 
        '''
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param
    def clone_model_zero_like(self, model):
        '''
        current_model --> zero_like clone_model 
        '''
        clone_zero_model = copy.deepcopy( model )
        for param, clone_param in zip(model.parameters(), clone_zero_model.parameters()):
            clone_param.data = torch.zeros_like( param.data.clone() )
        return clone_zero_model
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params):
        '''
        new_params --> self.model
        '''
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()


    def update_persionalized_parameters(self, new_params):
        '''
        new_params --> self.model
        '''
        for param , new_param in zip(self.persionalized_model.parameters(), new_params):
            param.data = new_param.data.clone()


    def update_predictor_parameters(self, new_params):
        for param , new_param in zip(self.model.predictor.parameters(), new_params):
            param.data = new_param.data.clone()

    def update_base_parameters(self, new_params):
        for param , new_param in zip(self.model.base.parameters(), new_params):
            param.data = new_param.data.clone()


    def updata_predictor_parameters_FORr2(self, local_predictor_model):
        for param , local_param in zip(self.model.predictor.parameters(), local_predictor_model.parameters()):
            local_param.data = param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self, mode = 'PM'):
        if mode == 'PM':
            model = self.persionalized_model
        else:
            model = self.model
        model.eval()
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        y_preds = []
        y_real = []
        
        with torch.no_grad():
            for x, y in self.testloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)['output']
                preds = torch.argmax(output, dim=1)
                test_acc += (torch.sum(preds == y)).item()
                test_num += y.shape[0]
                y_preds.append(preds.detach().cpu().numpy())
                y_real.append(y.detach().cpu().numpy())

                y_prob.append(output.detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))
        #f1 , recall , precision
        y_preds = np.concatenate(y_preds, axis=0)
        y_real = np.concatenate(y_real, axis=0) 
        # auc      
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        #f1
        f1 = metrics.f1_score(y_real , y_preds , average='macro')
        #recall
        recall = metrics.recall_score(y_real, y_preds, average='macro')
        #precision
        precision = metrics.precision_score(y_real, y_preds, average='macro')
        #acc
        accuracy = test_acc / test_num 
        self.accuracy.append(accuracy)
        return test_acc, test_num , accuracy,auc , f1 , recall, precision


    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)['output']
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        return train_acc, loss , self.train_samples
    
    def test_persionalized_model(self):
        self.persionalized_model.eval()
        test_acc = 0
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.persionalized_model(x)['output']
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            f1 = metrics.f1_score( y , output , average='macro')
        return test_acc, y.shape[0] , f1

    def train_error_and_loss_persionalized_model(self):
        self.persionalized_model.eval()
        train_acc = 0
        loss = 0
        # self.update_parameters(self.persionalized_model_bar)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.persionalized_model(x)['output']
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        return train_acc, loss , self.train_samples


    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
            # print( len(y) )
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))
    
    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))