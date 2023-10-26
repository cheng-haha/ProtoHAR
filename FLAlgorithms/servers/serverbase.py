import torch
import os
import numpy as np
import h5py
import copy
import torch.nn as nn
import json
from sklearn.preprocessing import StandardScaler
import matplotlib as mlp 
import pandas as pd
mlp.use("Agg")
import matplotlib.pyplot as plt

from utils.model_utils import read_data, read_user_data

class Server:
    def __init__(self, args, model , times):

        # Set up the main attributes
        self.vis = args.vis
        self.fineturning_epochs = args.fineturning_epochs
        self.device = args.device
        self.dataset = args.dataset
        self.num_glob_iters = args.num_global_iters
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.niid = args.niid
        self.personal_learning_rate = args.personal_learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model[0])
        self.users = []
        self.selected_users = []
        self.num_users = args.num_users
        self.beta = args.beta
        self.lamda = args.lamda
        self.algorithm = args.algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc,self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.rs_auc, self.rs_f1,self.rs_recall,self.rs_precision = [],[],[],[]
        self.metric_weight = []
        self.single_client_acc=[]
        self.times = times
        self.logging = args.logging
        self.not_save_res = args.notsaveresult
        self.total_users = args.total_users
        self.only_run_once = True

    def SetClient(self,args,cleintObj):
        data = read_data(args,self.dataset)
        self.logging.info("Users in total: {}".format(args.total_users))
        for i in range(args.total_users):
            id, train_data , test_data = read_user_data(i, data, dataset=self.dataset)
            user = cleintObj(args, id, self.model, train_data, test_data  )
            self.users.append(user)
            self.total_train_samples += user.train_samples           
        self.logging.info(f"Number of users / total users:{args.num_users} / {args.total_users}")
        self.logging.info(f"Finished creating {args.algorithm} server.")    

    def metrics_weight(self):
        if self.only_run_once:
            self.only_run_once = False
            for user in self.users:
                self.metric_weight.append(user.train_samples/self.total_train_samples)
            
    def clone_model_zero_like(self, model):
        '''
        current_model --> zero_like clone_model 
        '''
        clone_zero_model = copy.deepcopy( model )
        for param, clone_param in zip(model.parameters(), clone_zero_model.parameters()):
            clone_param.data = torch.zeros_like( param.data.clone() )
        return clone_zero_model
    def clone_model(self, model1 , model2):
        '''
        model1 --> model2
        return model2 
        '''
        for param, clone_param in zip(model1.parameters(), model2.parameters()):
            clone_param.data = param.data.clone()
        return model2

   
    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        # model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

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

    def save_model(self):
        if not self.not_save_res:
            model_path = os.path.join("models", self.dataset)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))


    def plot_2d_features(self ,model , add_legend = True):
        net_logits = np.zeros((10000, 2), dtype=np.float32)
        net_labels = np.zeros((10000,), dtype=np.int32)
        model.eval()
        count = 0
        for u_dx,user in enumerate(self.users):
            with torch.no_grad():
                for b_idx, (data, target) in enumerate(user.testloaderfull):
                    data, target = data.to(self.device), target.to(self.device)
                    rep = model(x=data)['proto']#拿到特征向量
                    output2d = rep.cpu().data.numpy()
                    target = target.cpu().data.numpy()
                    # print(output2d.shape , target.shape,len(test_loader))
                    net_logits[u_dx * ( len(target)) : (u_dx+1)*len(target), :] = output2d
                    net_labels[u_dx * ( len(target)) : (u_dx+1)*len(target)] = target
                    count += len(target)
        # print(len(net_logits) , count)
        net_logits = np.delete( net_logits  , np.s_[count:],0 )
        # print(len(net_logits))
        net_labels = np.delete( net_labels  , np.s_[count:],0 )
        scaler = StandardScaler()
        net_logits = scaler.fit_transform(net_logits)
        classes = np.unique(net_labels).tolist()
        # print(len(net_labels))
        colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 
                'm', 'tab:brown','black' ,'tab:olive','tab:cyan','grey' , 'deeppink','b' ]
        plt.figure(1,figsize=(5, 5))
        plt.grid()
        # plt.title(f"{self.algorithm}"+ "Model")
        plt.ylabel('Activation of the 2st neuron')
        plt.xlabel('Activation of the 1st neuron')     
        
        for label in classes:
            idx = net_labels == label
            # print(len(idx), len(net_logits))
            plt.scatter(net_logits[idx, 0], net_logits[idx, 1], c=colors[label])

        if add_legend:
            plt.legend( np.arange(len(classes), dtype=np.int32),fontsize = 10 ,title='class id:',
                    bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
        # plt.subplots_adjust(wspace=0.8)
        plt.savefig( "runs/"+str(self.algorithm)+'_'+ str( self.dataset) +'_representation.png',dpi=600,bbox_inches='tight')
        path = 'plotresult/'+str(self.algorithm)+'_'+ str( self.dataset)+'.json'
        result = {'classes':classes , 'net_logits':net_logits.tolist(),'net_labels':net_labels.tolist()}
        with open( path ,'w') as outfile:
            json.dump(result, outfile)


    def select_users(self, round, num_users, return_idx=False):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            if return_idx:
                return self.users,[ x for x  in range( len(self.users) ) ]
            else:
                return self.users

        num_users = min(num_users, len(self.users))
        if return_idx:
            user_idxs = np.random.choice(range(len(self.users)), num_users, replace=False)  # , p=pk)
            return [self.users[i] for i in user_idxs], user_idxs
        else:
            return np.random.choice(self.users, num_users, replace=False)

    # define function for persionalized agegatation.
    def persionalized_update_parameters(self,user, ratio):
        # only argegate the local_weight_update
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio


    def persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            #self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta
        # 综合平均模型与使用参数beta的先前模型 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = self.beta*pre_param.data + (1 - self.beta)*param.data
   
    # Save loss, accurancy to h5 fiel
    def save_results(self):
        if not self.not_save_res:
            alg = self.dataset + "_" + self.algorithm
            alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) \
                + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" +"_" + str(self.total_users) +"tu"\
                + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)+'_'+str( self.niid )
            if(self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
                alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
            alg = alg + "_" + str(self.times)
            if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
                print(f"{alg}")
                with h5py.File("./results/"+'{}.h5'.format(alg), 'w') as hf:
                    hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                    hf.create_dataset('rs_f1', data=self.rs_f1)
                    hf.create_dataset('rs_auc', data=self.rs_auc)
                    hf.create_dataset('rs_precision', data=self.rs_precision)
                    hf.create_dataset('rs_recall', data=self.rs_recall)
                    hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                    hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                    hf.close()

    def test(self,mode):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        tot_accurancy = []
        tot_auc = []
        tot_f1 = []
        tot_recall = []
        tot_precision = []
        for c in self.users:
            ct, ns , user_acc ,user_auc,user_f1,user_recall,user_precision  = c.test(mode)
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            tot_accurancy.append( user_acc )
            tot_auc.append( user_auc )
            tot_f1.append( user_f1 )
            tot_recall.append( user_recall )
            tot_precision.append( user_precision )
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct , tot_accurancy, tot_auc,tot_f1,tot_recall,tot_precision



    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def test_persionalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        tot_accurancy = []
        tot_f1 = []
        for c in self.users:
            c.FineTurning()
            ct, ns, f1 = c.test_persionalized_model()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            tot_accurancy.append( ct/ns )
            tot_f1.append(f1)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct , tot_accurancy , tot_f1

    def train_error_and_loss_persionalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate(self,mode=None):
        self.metrics_weight()
        #ids, num_samples, tot_correct, tot_accurancy, tot_auc, tot_f1, tot_recall, tot_precision
        stats = self.test(mode)
        stats_train = self.train_error_and_loss()

        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        glob_acc_avg = np.sum( [ score*self.metric_weight[i] for i , score in enumerate(stats[3]) ])
        auc_avg = np.sum( [ score*self.metric_weight[i] for i , score in enumerate(stats[4]) ])
        f1_avg = np.sum( [ score*self.metric_weight[i] for i , score in enumerate(stats[5]) ])
        recall_avg = np.sum( [ score*self.metric_weight[i] for i , score in enumerate(stats[6]) ])
        precision_avg = np.sum( [ score*self.metric_weight[i] for i , score in enumerate(stats[7]) ])

        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        
        self.rs_glob_acc.append(glob_acc_avg)#denote : last evalution accuracy is mean( each user accuracy )
        self.rs_auc.append(auc_avg)
        self.rs_f1.append(f1_avg)
        self.rs_recall.append(recall_avg)
        self.rs_precision.append(precision_avg)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        # self.logging.info(f'')
        self.logging.info(f"Average Global Accurancy: {glob_acc:.6f}  |  Average Client Accurancy:    {glob_acc_avg:.6f}" )
        self.logging.info(f"Average AUC Accurancy:    {auc_avg:.6f}  |  Average F1-Score Accurancy:  {f1_avg:.6f}" )
        self.logging.info(f"Average Recall Accurancy: {recall_avg:.6f}  |  Average Precision Accurancy: {precision_avg:.6f}" )
        self.logging.info(f"Average Train Accurancy:  {train_acc:.6f}  |  Average Train Loss:          {train_loss:.6f}")

    def evaluate_personalized_model(self):
        stats = self.test_persionalized_model()  
        stats_train = self.train_error_and_loss_persionalized_model()
        # do not use this metric
        glob_acc_per = np.sum( [ score*self.metric_weight[i] for i , score in enumerate(stats[3]) ])
        f1_avg_per = np.sum( [ score*self.metric_weight[i] for i , score in enumerate(stats[4]) ])
        train_acc_per = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss_per = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc_per)
        self.rs_train_acc_per.append(train_acc_per)
        self.rs_train_loss_per.append(train_loss_per)
        #print("stats_train[1]",stats_train[3][0])
        self.logging.info('------------------------------------Personal Evaluation------------------------------------------')
        self.logging.info(f"Personal ACC Accurancy:    {glob_acc_per:.6f}  |  Personal F1-Score Accurancy:  {f1_avg_per:.6f}" )
        self.logging.info(f"Personal Train Accurancy:  {train_acc_per:.6f}  |  Personal Train Loss:          {train_loss_per:.6f}")

