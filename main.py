import matplotlib.pyplot as plt
import numpy as np
import argparse

import os
import math
from utils.ResultLogging import loadLogger
from utils.create_server import create_server_n_user
import torch


def setup_seed(seed):
    if seed != -1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        #  random.seed(seed)
        torch.backends.cudnn.deterministic = True
    else:
        print('==>No Seed')



def main( times, gpu ):
    args.logging = loadLogger(args)
    args.device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    # allUserAcc = 0.0
    for i in range(times):
        # i=2
        setup_seed(i)
        
        args.logging.info(f'==>seed is {i}')
        args.logging.info("\n\n         [ Start training iteration {} ]           \n\n".format(i))
        server = create_server_n_user(args, i)
        allUserAcc = server.train()

    args.logging.info("Finished training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # save setting
    parser.add_argument('--savelogging','--slog', default=False, action='store_true',
                          help='If yes, only output log to terminal.')
    parser.add_argument('--notsaveresult','--nsr', default=False, action='store_true',
                          help='The default value is saved.')
    parser.add_argument('--work_dir', default='./work_dir',
                        help='the work folder for storing results')

    #ProtoHAR
    parser.add_argument("--beta", type=float, default=0.0, help="Average moving parameter for pFedMe, \
                        or Second learning rate of Per-FedAvg , temperature of MOON")
    parser.add_argument("--lamda", type=float, default=0.1, help="Regularization term")
    parser.add_argument("--distance", type=str, default='MSE',choices=['MMD','L1','MSE','infonce'],help="distance between embedding vectors and global prototypes")
    parser.add_argument("--weighting", default=False,action='store_true' , help="weighting prototypes")
    parser.add_argument("--tau", default=0,type=float, help="Parametrization of normalized classifier")
        #-------------------------------Share setting--------------------------------#
    parser.add_argument("--dataset", type=str, default="uschad",choices=['ucihar','uschad','unimib','pamap2','wisdm','harbox','hhar'])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--algorithm", type=str, default="pFedMe") 
    parser.add_argument("--sampleratio", type=float, default=0.2) 
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--fineturning_epochs", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=100.0, help="clip grad")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--num_classes", type=int, default=20, help="num_classes")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--plocal_steps", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--niid", type=str, default="0.2p")
    parser.add_argument("--total_users", type=int, default=30, help="Number of all Users per round")
    parser.add_argument("--personal_learning_rate", type=float, default=0.001, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=5, help="running time")
    parser.add_argument("--device", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enables visualizing 2d features (default: False).')
    parser.add_argument('--vis_proto', action='store_true', default=False,
                        help='enables visualizing 2d features (default: False).')
    parser.add_argument('--imbalance', action='store_true', default=False)
    args = parser.parse_args()
    # get the subset of total numuers
    args.num_users = 1 if args.sampleratio == 0.0 else math.ceil( args.total_users * args.sampleratio )

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("Device            : {}".format(args.device))
    print("=" * 80)
    
    if not os.path.exists("runs/"):
        print('==>create runs file')
        os.mkdir("runs/")
    main(times = args.times,gpu=args.device)
