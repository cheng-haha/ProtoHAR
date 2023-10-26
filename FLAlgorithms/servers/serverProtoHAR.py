
import torch
import copy

from FLAlgorithms.users.userProtoHAR import UserProtoHAR
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
# Implementation for FedProx Server
from utils.decorator import LoggingUserAcc
import time
class ProtoHAR(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        self.SetClient(args,UserProtoHAR)
        self.logging.info(f"It is {str(args.weighting)} weighting version, scaled parameter is {str(args.beta) if args.weighting else 0}")
    @LoggingUserAcc
    def train(self):

        global_protos = {}

        for glob_iter in range(self.num_glob_iters):
            self.logging.info(f"-------------Round number:{glob_iter} -------------")
            self.send_parameters()
            local_protos = {} 
            self.evaluate()
            if glob_iter == 0:
                print('==>make global model trans to base')
                self.model = copy.deepcopy(self.model.base)
                print(self.model)
            self.selected_users ,user_idx = self.select_users( glob_iter,self.num_users , return_idx=True)

            for user ,idx in zip( self.selected_users , user_idx ): # allow selected users to train
                    agg_protos = user.train( glob_iter , global_protos )
                    local_protos[idx] = agg_protos
            global_protos = self.proto_aggregation( local_protos  )
            
            self.aggregate_parameters()

        self.save_results()
        self.save_model()

    def proto_aggregation(self , local_protos_list   ):
        agg_protos_label = dict()
        for idx in local_protos_list:
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = [proto / len(proto_list)]
            else:
                agg_protos_label[label] = [proto_list[0].data]
        return agg_protos_label


    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters( user, user.train_samples / total_train )
    def add_parameters( self, user, ratio ):
        for server_param, user_param in zip(self.model.parameters(), user.get_base_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio
