import ujson
import numpy as np
import os
import torch
import torch.nn as nn
import numpy as np
import random
import re
from FLAlgorithms.trainmodel.cnnbase_fc import HARCNN

DATA_USER_INFO = {
    'pamap2':9,
    'uschad':14,
    'harbox':120,
    'unimib':30
                    }


def suffer_data(data):
    data_x = data['x']
    data_y = data['y']
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    return (data_x, data_y)
    
def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)

def create_model(model, dataset,device):
    passed_dataset = get_dataset_name(dataset)
    # print(passed_dataset)
    if passed_dataset in ['pamap2' ,'unimib' ,'uschad','harbox'] :
        # print(passed_dataset)
        print(f'==>create model HARCNN')
        model = HARCNN(dataset).to(device) , model 
    else:
        raise('no model')
    
    return model
def get_dataset_name(dataset):
    dataset=dataset.lower()
    # print(dataset)
    passed_dataset=dataset.lower()
    if 'pamap2' in dataset:
        passed_dataset = 'pamap2'
    elif 'unimib' in dataset:
        passed_dataset = 'unimib'
    elif 'uschad' in dataset:
        passed_dataset = 'uschad'
    elif 'harbox' in dataset:
        passed_dataset = 'harbox'
    else:
        raise ValueError('Unsupported dataset {}'.format(dataset))
    return passed_dataset



def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts +1)))
        sample_index = batch_idx*batch_size
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x,data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)


def read_data(args,dataset):
    '''
    just check the file path , not really read all data due to I rewrite this funciton
    '''
    try:
        niid = re.findall(r"\d+\.?\d*",args.niid)[0]
    except Exception as e:
        raise ValueError('please check your niid command')
    if  isinstance(niid,float) :
        # Dirchlet distribution
        train_data_dir = os.path.join('data',dataset,f'{args.total_users}b_{niid}p_data', 'train/')
        test_data_dir  = os.path.join('data',dataset,f'{args.total_users}b_{niid}p_data', 'test/')
    else:
        if args.imbalance:
            samplelength    = re.findall(r"\d+\.?\d*",args.niid)[0]
            train_data_dir  = os.path.join('data',dataset,f'{args.total_users}u_{samplelength}l_data', 'train/')
            test_data_dir   = os.path.join('data',dataset,f'{args.total_users}u_{samplelength}l_data', 'test/')
        else:
            assert args.imbalance is False , print('please check your niid command')
            # Pathological distribution (include iid)
            k_shots , stdv  = re.findall(r"\d+\.?\d*",args.niid)
            train_data_dir = os.path.join('data',dataset,f'{k_shots}k_{args.total_users}b_{stdv}p_data', 'train/')
            test_data_dir = os.path.join('data',dataset,f'{k_shots}k_{args.total_users}b_{stdv}p_data', 'test/')
    return train_data_dir , test_data_dir

def read_all_test_data(test_data , total_users ):

    all_test_data = []
    for id in range(total_users):
        # print(test_data.shape)
        id = str(id)
        X_test, y_test = test_data[id]['x'], test_data[id]['y']
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
        test_id_data = [(x, y) for x, y in zip(X_test, y_test)]
        all_test_data+=test_id_data
    return all_test_data

def read_user_data(index,data , dataset):
    train_data_path = data[0]
    test_data_path  = data[1]
    user_file = str(index) + '.npz'
    assert user_file in os.listdir(train_data_path) , f'There is not have the {index} user'
    with open( train_data_path + user_file, 'rb') as f:
        train_data = np.load(f, allow_pickle=True)['data'].tolist()
    with open( test_data_path  + user_file, 'rb') as f:
        test_data = np.load(f, allow_pickle=True)['data'].tolist()
    X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']

    X_train = torch.Tensor(X_train).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.int64)
    X_test  = torch.Tensor(X_test).type(torch.float32)
    y_test  = torch.Tensor(y_test).type(torch.int64)
    
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data  = [(x, y) for x, y in zip(X_test, y_test)]
    return index, train_data, test_data
