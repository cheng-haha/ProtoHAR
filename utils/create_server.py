from FLAlgorithms.trainmodel.cnnbase_fc import LocalModel
from utils.model_utils import create_model
import copy
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverlocal import Local
from FLAlgorithms.servers.serverFedRep import FedRep
from FLAlgorithms.servers.serverProtoHAR import ProtoHAR
import torch.nn as nn
from FLAlgorithms.servers.serverFedHome import FedHome

def create_server_n_user(args, i):
    model = create_model(args.model, args.dataset, args.device )
            # select algorithm
    algorithm = args.algorithm
    if(algorithm == "FedAvg"):
        server = FedAvg(args , model , i )
    elif(algorithm == 'FedHome'):
        predictor = copy.deepcopy(model[0].classifier)
        model[0].classifier = nn.Identity()
        model = LocalModel(model[0], predictor).to(args.device),model[1]
        server = FedHome( args , model , i )
    elif(algorithm == 'FedRep'):
        predictor = copy.deepcopy(model[0].classifier)
        model[0].classifier = nn.Identity()
        model = LocalModel(model[0], predictor).to(args.device),model[1]
        server = FedRep( args , model , i )
    elif(algorithm == 'ProtoHAR'):
        predictor = copy.deepcopy(model[0].classifier)
        model[0].classifier = nn.Identity()
        model = LocalModel(model[0], predictor).to(args.device),model[1]
        server = ProtoHAR( args , model , i )
    elif(algorithm == 'Local'):
        server = Local( args , model , i )
    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()
    return server