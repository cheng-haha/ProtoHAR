import os
import random
import json
from scipy import stats
from collections import Counter
import argparse
import numpy as np
import sys 
import scipy.io as scio
from sklearn.model_selection import train_test_split
import logging
import math
np.random.seed(1)
random.seed(1)



def create_user_data( user_id ,num_users, reshaped_segments , labels , stdv  ):
    X = []
    Y = []

    for i in range( num_users ):
        index = np.where( user_id == i )

        ############################# Pathological distribution #########################
        user_labels =  np.unique(labels[index[0]])
        print(f'the re labels for {i} user is  ' , user_labels )

        num_labels = len( np.unique(labels[index[0]]) )
        num_classes =  num_labels - stdv
        activity_sample = np.random.choice(np.unique(labels[index[0]]), max(1,num_classes),  replace=False)
        print(f'activity_sample for {i} user  is ' ,  activity_sample)
        
        activity_index = np.empty(0,dtype=int)
        for elem in activity_sample:
            activity_index = np.append( activity_index ,  np.where( labels[index[0]] == elem )[0] )
        X.append(reshaped_segments[index[0]][activity_index]   )
        Y.append(labels[index[0]][activity_index] )
        
        print(f'now the labels for {i} user is ' , np.unique( labels[index[0]][activity_index]  ))
    return X,Y

def create_user_data_for_one_idx( user_id ,num_users, reshaped_segments , labels  , stdv = 0 ):
    X = []
    Y = []

    for i in range( num_users ):
        index = np.where(user_id == i + 1 )

    #############################Pathological distribution#########################    
        num_labels = len(np.unique(labels[index[0]]))
        num_classes =  num_labels - stdv
        activity_sample = np.random.choice(np.unique(labels[index[0]]), max(2,num_classes),  replace=False)        # print(shared)
        print( 'activity_sample is ' ,  activity_sample)
        activity_index = np.empty(0,dtype=int)
        for elem in activity_sample:
            activity_index = np.append( activity_index ,  np.where( labels[index[0]] == elem )[0] )
        X.append(reshaped_segments[index[0]][activity_index]   )
        Y.append(labels[index[0]][activity_index] -1)
        print('the  re labels is ' , np.unique(labels[index[0]] -1 ) )
        print('now the labels is ' , np.unique( labels[index[0]][activity_index] -1 ))
    return X,Y



def CountLabels( uname , y_train ,logging, mode = 'train'):
    labels = {}
    labels_num = {}
    count_y = Counter( y_train )
    for idx in count_y:
        labels_idxs = np.where( y_train == idx)[0]
        labels[idx] = [ count_y[idx] , labels_idxs ]#当前标签的总数，对应标签的索引
        labels_num[str(idx)] = count_y[idx]
        logging.info(f'the {mode} user {uname} contains the label {idx} samples is {count_y[idx]}')
    return labels,labels_num



def Count_EveryClass(total_labels,num_users):
    classes_num = {}
    for i in range(num_users):
        for  label , Samplenum_Sampleidx in total_labels[i].items():
            #累加当前标签的数量
            if label in classes_num:
                classes_num[label] += Samplenum_Sampleidx[0]
            else:
                classes_num[label] = Samplenum_Sampleidx[0]
    return classes_num

def shot_data( X , y , label_idx ,logging, k_shots ,stdv, split_radio=0.7 ):
    trainx , trainy , testx , testy = [],[],[],[]

    for idx in label_idx:
        len_this_label = int(label_idx[idx][0] * split_radio)
        train_size = len_this_label if len_this_label < k_shots else  k_shots
        if train_size != 0:
            if stdv != 0 and train_size == k_shots:
                train_size = np.random.randint( max( 1, train_size - stdv ) ,  train_size + stdv )        

            test_size= max( int(train_size*(1-split_radio) / split_radio ) , 1 )
            shffle_idxs = label_idx[idx][1]
            assert len(shffle_idxs) == label_idx[idx][0] 
            random.shuffle( shffle_idxs )
            if len(shffle_idxs) != 1:
            # logging.info(f'==>label {idx} of  k_shots is {k_shots} ')
                if trainx == []:
                    # print('!!!!!')
                    trainx = X[ shffle_idxs[:train_size] ]
                    trainy = y[ shffle_idxs[:train_size] ]
                    testx  = X[ shffle_idxs[train_size:train_size+test_size] ]
                    testy  = y[ shffle_idxs[train_size:train_size+test_size] ]
                else:
                    trainx = np.concatenate( (trainx ,X[ shffle_idxs[:train_size] ] ) )
                    trainy = np.concatenate( (trainy ,y[ shffle_idxs[:train_size] ] ) )
                    testx  = np.concatenate( (testx  ,X[ shffle_idxs[train_size:train_size+test_size] ] ) )
                    testy  = np.concatenate( (testy  ,y[ shffle_idxs[train_size:train_size+test_size] ] ) )
            else:
                pass
            logging.info(f'==>train label {idx} of length is {train_size} ')
        else:
            logging.info(f'==>label {idx} is only one sample so discarded')
    return trainx , trainy , testx , testy



def ShotDataForAllSamples( X , y , label_idx ,logging, k_shots ,stdv, split_radio=0.7 ):
    trainx , trainy , testx , testy = [],[],[],[]

    for idx in label_idx:
        len_this_label = label_idx[idx][0] if label_idx[idx][0] <= k_shots else  k_shots
        if len_this_label != 1:
            if stdv != 0:
                # print(max( 1, len_this_label - stdv + 1) ,  min( len_this_label + stdv - 1 , label_idx[idx][0] ))
                len_this_label = np.random.randint( max( 1, len_this_label - stdv + 1) ,  min( len_this_label + stdv - 1 , label_idx[idx][0] ) )        
            train_size = max( int( len_this_label * split_radio ) , 1 )
            test_size  = len_this_label - train_size 
            shffle_idxs = label_idx[idx][1]
            assert len(shffle_idxs) == label_idx[idx][0] 
            random.shuffle( shffle_idxs )
            if len(shffle_idxs) != 1:
            # logging.info(f'==>label {idx} of  k_shots is {k_shots} ')
                if trainx == []:
                    # print('!!!!!')
                    trainx = X[ shffle_idxs[:train_size] ]
                    trainy = y[ shffle_idxs[:train_size] ]
                    testx  = X[ shffle_idxs[train_size:train_size+test_size] ]
                    testy  = y[ shffle_idxs[train_size:train_size+test_size] ]
                else:
                    trainx = np.concatenate( (trainx ,X[ shffle_idxs[:train_size] ] ) )
                    trainy = np.concatenate( (trainy ,y[ shffle_idxs[:train_size] ] ) )
                    testx  = np.concatenate( (testx  ,X[ shffle_idxs[train_size:train_size+test_size] ] ) )
                    testy  = np.concatenate( (testy  ,y[ shffle_idxs[train_size:train_size+test_size] ] ) )
            else:
                pass
            logging.info(f'==>label {idx} of length is {len_this_label} ')
        else:
            logging.info(f'==>label {idx} is only one sample so discarded')
    return trainx , trainy , testx , testy


def creat_json( X,y,train_path , test_path ,k_shots ,stdv , logging_path, num_users=10  ):
    data_logging = {}
    num_samples  = {'train':[],'test':[]}
    logging.basicConfig(filename=os.path.join( os.path.dirname(os.path.dirname(train_path)),\
        f'dataInfo_{str(num_users)}u_{str(stdv)}p.log'), level=logging.DEBUG , filemode='w') 
    logger = logging.getLogger()
    formatter = logging.Formatter(fmt='[%(asctime)s]  %(message)s',
        datefmt='%m-%d %H:%M')

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    
    logger.addHandler(sHandler)
    for i in range( num_users ):
        # print(f'==>original k_shots is{k_shots}')

        uname = i #'f_{0:05d}'.format(i)
        # X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.5 , stratify=y[i] ,random_state= 42 )   
        user_labels, user_label_num= CountLabels( uname , y[i],logging=logger,mode = 'all')
        X_train , y_train ,  X_test , y_test= shot_data( X[i] , y[i] , user_labels , logger , k_shots , stdv = stdv )
        user_train_labels, user_train_label_num= CountLabels( uname , y_train,logging=logger)
        user_test_labels,_ = CountLabels(uname , y_test , mode='test',logging=logger )        
        data_logging[str(uname)] = user_train_label_num
        count_y = Counter( y_train )
        # for idx in count_y:
        #     print(f'the user{uname} contains the label {idx} samples is {count_y[idx]}')
        print('************************************************************')
        train_data = {'x': X_train, 'y': y_train}
        num_samples['train'].append(len(y_train))
        with open(train_path + str(i) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_data)
        test_data  = {'x': X_test, 'y': y_test}
        num_samples['test'].append(len(y_test))
        with open(test_path + str(i) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_data)
        
    # print(data_logging)    
    logger.info(f"train {num_samples['train']}")
    logger.info(f"test {num_samples['test']}")
    logger.info(f"Num_samples:{num_samples['train'] + num_samples['test']}")
    logger.info(f"Total_samples:{sum(num_samples['train'] + num_samples['test'])} ,Train_samples:{sum(num_samples['train'])} , Test_samples:{sum(num_samples['test'])}" )
    with open(logging_path,'w') as outfile:
        json.dump(data_logging, outfile )    


def delete_sparse_class(x,y, min_samples = 3 ):
    count_y = Counter( y )
    # print(count_y)
    labels_idxs = np.arange(len(y))
    delete_labels_idxs = None
    for idx in count_y:
        if count_y[idx]  < min_samples:
            if delete_labels_idxs is not None:
                delete_labels_idxs = np.concatenate([delete_labels_idxs,np.where( y == idx)[0]])
            else:
                delete_labels_idxs = np.where( y == idx)[0]
            print(f'the class {idx} of original length is {count_y[idx]}, deleting the class {idx}')
    print(f'deleting idxs of the classes is {delete_labels_idxs}')
    if delete_labels_idxs is not None:
        labels_idxs = np.delete(labels_idxs,delete_labels_idxs)
    return x[labels_idxs],y[labels_idxs]


def random_creat_json( X,y, train_path , test_path , logging_path, sample_size = 280 , num_users=10 , min_sample = 10  ):
    '''
    imbalance distribution . pelease generating balance distribution by use 'creat_json' function with alpha = 0.
    '''
    data_logging = {}
    num_samples  = {'train':[],'test':[]}
    logging.basicConfig(filename=os.path.join( os.path.dirname(os.path.dirname(train_path)),\
        f'dataInfo_{str(num_users)}u_{str(sample_size)}l.log'), level=logging.DEBUG , filemode='w') 
    logger = logging.getLogger()
    formatter = logging.Formatter(fmt='[%(asctime)s]  %(message)s',
        datefmt='%m-%d %H:%M')

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    
    logger.addHandler(sHandler)
    for i in range( num_users ):
        uname = i #'f_{0:05d}'.format(i)
        imbalance_sample_size   = np.random.randint(min_sample,sample_size)
        shuffle_idxs            = np.arange(len(X[i]))
        np.random.shuffle( shuffle_idxs )
        
        imbalance_sample_idxs   = shuffle_idxs[:imbalance_sample_size] if len(X[i]) > imbalance_sample_size else np.arange(len(X[i]))
        # print(len(X[i]),imbalance_sample_idxs)
        random_x                = X[i][ imbalance_sample_idxs ]
        random_y                = y[i][ imbalance_sample_idxs ]
        # print(len(random_x),len(random_y))
        random_x , random_y     = delete_sparse_class(random_x , random_y)
        X_train, X_test, y_train, y_test            = train_test_split( random_x , random_y , test_size=0.3 , stratify=random_y ,random_state= 42 )   
        user_train_labels,  user_train_label_num    = CountLabels( uname , y_train,logging=logger)
        user_test_labels ,  _                       = CountLabels(uname , y_test , mode='test',logging=logger )        
        data_logging[str(uname)]    = user_train_label_num
        count_y = Counter( y_train )
        # for idx in count_y:
        #     print(f'the user{uname} contains the label {idx} samples is {count_y[idx]}')
        print('************************************************************')
        train_data = {'x': X_train, 'y': y_train}
        num_samples['train'].append(len(y_train))
        with open(train_path + str(i) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_data)
        test_data  = {'x': X_test, 'y': y_test}
        num_samples['test'].append(len(y_test))
        with open(test_path + str(i) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_data)
        
    logger.info(f"train {num_samples['train']}")
    logger.info(f"test {num_samples['test']}")
    logger.info(f"Num_samples:{num_samples['train'] + num_samples['test']}")
    logger.info(f"Total_samples:{sum(num_samples['train'] + num_samples['test'])} ,Train_samples:{sum(num_samples['train'])} , Test_samples:{sum(num_samples['test'])}" )
    with open(logging_path,'w') as outfile:
        json.dump(data_logging, outfile )   
