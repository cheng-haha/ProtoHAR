

import numpy as np
import os
import argparse
import pandas as pd
import numpy as np
import scipy.io as scio
import sys 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from DataSplitting import random_creat_json,creat_json
WINDOW_SIZE=512 
DATASET_DIR='/data1/experiment/chengdongzhou/data/raw/large_scale_HARBox/'
NUM_USERS = 120
NUM_LABELS = 5
CLASS_SET = ['Call','Hop','typing','Walk','Wave']
DIMENSION_OF_FEATURE = 900


def process_data(n_users):
    total_class = 0
    user_ids = []
    coll_class = []
    coll_label = []

    for user_id in range( 1,n_users+1):    
        print(F'==> {user_id} user_id is processing data..')
        for class_id in range(NUM_LABELS):
            read_path = DATASET_DIR  +  str(user_id) + '/' + str(CLASS_SET[class_id]) + '_train' + '.txt'

            if os.path.exists(read_path):

                temp_original_data = np.loadtxt(read_path)
                temp_reshape = temp_original_data.reshape(-1, 100, 10)
                temp_coll = temp_reshape[:, :, 1:10].reshape(-1, DIMENSION_OF_FEATURE)
                temp_coll = np.expand_dims(temp_coll.reshape(-1,DIMENSION_OF_FEATURE//9,9),axis=1)
                count_img = temp_coll.shape[0]
                temp_label = class_id * np.ones(count_img)

                user_ids.extend([user_id]*len(temp_label))

                coll_class.extend(temp_coll)
                coll_label.extend(temp_label)

                total_class += 1
            # print(temp_coll.shape )
        # print(len(coll_class))
        # print(len(coll_label))
    
    user_ids = np.array(user_ids)
    reshaped_segments = np.array(coll_class)
    labels = np.array(coll_label,dtype=int)
    print( f'==>all users is {np.unique(user_ids)} , users data classes is {np.unique(labels)}')
    print(len(user_ids),len(reshaped_segments),len(labels))
    assert len(user_ids) == len(reshaped_segments)

    return user_ids , reshaped_segments, labels 

def create_user_data( user_id , num_users , reshaped_segments , labels , stdv  ):
    X = []
    Y = []

    for i in range( num_users ):
        index = np.where( user_id == i + 1   )
    #############################全体病态切分#########################    
        num_labels = len(np.unique(labels[index[0]]))
        num_classes =  num_labels - stdv
        activity_sample = np.random.choice(np.unique(labels[index[0]]), max(1,num_classes),  replace=False)        # print(shared)
        print( 'activity_sample is ' ,  activity_sample)
        activity_index = np.empty(0,dtype=int)
        for elem in activity_sample:
            activity_index = np.append( activity_index ,  np.where( labels[index[0]] == elem )[0] )
        X.append(reshaped_segments[index[0]][activity_index]   )
        Y.append(labels[index[0]][activity_index] )
        print('the  re labels is ' , np.unique( labels[index[0]]) )
        print('now the labels is ' , np.unique( labels[index[0]][activity_index]  ))
    print('Y is ',len(Y))
    count= 0
    for i in range(len(X)):
        print(len(X[i]))
        count+=len(X[i])
    print('X的长度是',count)
    return X,Y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_class", type=int, default=NUM_LABELS, help="number of classification labels")
    parser.add_argument("--min_sample", type=int, default=20, help="Min number of samples per user.")
    parser.add_argument("--n_ways", type=int, default=NUM_LABELS, help="n ways")
    parser.add_argument("--k_shots", type=int, default=20, help="k shot for train samples")
    parser.add_argument("--stdv", type=int, default=2, help="noise for choosing k shots and deleting n ways")
    parser.add_argument("--n_user", type=int, default=NUM_USERS,
                        help="number of local clients, should be muitiple of 10.")

    parser.add_argument('--imbalance',action='store_true',default=False)
    parser.add_argument("--sample_size", type=int, default=256, help="Min number of samples per user.")
    args = parser.parse_args()
    print()
    print("Number of classes: {}".format(args.n_class))
    print("stdv for noisy: {}".format(args.stdv))
    print("n_ways: {}".format(args.n_ways))
    print("k_shots: {}".format(args.k_shots))
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    print(dir_path)
    if args.imbalance:
        # assert args.stdv == 0 , print('do not delete some classes')
        train_path      =  os.path.join( dir_path,f'harbox/{args.n_user}u_{args.sample_size}l_data/train/')
        test_path       =  os.path.join( dir_path,f'harbox/{args.n_user}u_{args.sample_size}l_data/test/')
        logging_path    =  os.path.join( dir_path,'harbox/VisualDataDistribution',f'{args.n_user}u_{args.sample_size}l_logging.json')
    else:
        train_path      =  os.path.join( dir_path,f'harbox/{args.k_shots}k_{args.n_user}b_{args.stdv}p_data/train/')
        test_path       =  os.path.join( dir_path,f'harbox/{args.k_shots}k_{args.n_user}b_{args.stdv}p_data/test/')
        logging_path    =  os.path.join( dir_path,'harbox/VisualDataDistribution',f'{args.k_shots}k_{args.n_user}b_{args.stdv}p_logging.json')

    dir_path = os.path.dirname(train_path)
    for path in (train_path,test_path,logging_path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    user_id , segments, labels = process_data(args.n_user)
    print('useid num is ', len(user_id), 'segments num is ' , len(segments))
    user_data_x,user_data_y = create_user_data( user_id ,NUM_USERS, segments , labels  ,stdv=args.stdv )
    if args.imbalance:
        random_creat_json(user_data_x , user_data_y , train_path , test_path, logging_path=logging_path,sample_size=args.sample_size,num_users=NUM_USERS,min_sample=args.min_sample)
    else:
        creat_json( user_data_x , user_data_y , train_path , test_path, k_shots=args.k_shots , stdv=args.stdv, logging_path=logging_path,num_users=NUM_USERS) 
    print("Finish Generating Samples")

if __name__ == '__main__':

    main()