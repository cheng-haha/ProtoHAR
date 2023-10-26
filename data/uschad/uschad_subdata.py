
import numpy as np
import os
import argparse
import numpy as np
import scipy.io as scio
import sys 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from DataSplitting import create_user_data,creat_json, random_creat_json



WINDOW_SIZE=512 
OVERLAP_RATE=0.0
NUM_USERS = 14
NUM_LABELS = 12



def slide_window(array, w_s, stride):
    '''
    滑窗处理
    array: ---
    w_s: 窗口大小
    stride： 滑动步长
    '''
    x = []
    times = (array.shape[0] - w_s) // stride + 1
    i=0
    for i in range(times):
        x.append(array[stride*i: stride*i+w_s]) 
    #最后一个保留处理 
    if stride*i+w_s < array.shape[0]-1:
        x.append(array[-w_s:])
    return x

def merge_data(path, w_s, stride):
    '''
    所有数据按类别进行合并
    path: 原始 USC_HAD 数据路径
    w_s： 指定滑窗大小
    stride： 指定步长
    '''
    result = [] # 12类，按索引放置每一类数据
    '''对每一个数据进行滑窗处理，将滑窗后的数据按类别叠加合并放入result对应位置'''
    subject_list = os.listdir(path)
    subject_list = [ subject for subject in subject_list if subject.find('Subject')!=-1 ]
    os.chdir(path)
    user_id =[]
    labels = []
    print(subject_list)
    for i , subject in enumerate( subject_list):
        print(i)
        if not os.path.isdir(subject):
            continue
        print('======================================================\n         current Subject sequence: 【%s】\n'%(subject))
        mat_list = os.listdir(subject)
        os.chdir(subject)

        for mat in mat_list:
            category = int(mat[1:-6])-1 #获取类别
            content = scio.loadmat(mat)['sensor_readings']
            

            x = slide_window(content, w_s, stride)
            # print(x.)
            result.extend(x)
            user_id.extend([i]*len(x))
            labels.extend([category]*len(x))

        os.chdir('../')
    os.chdir('../')
    print(len(user_id) ,len(result), len(labels))
    return np.array(user_id,dtype=int) , np.expand_dims(np.array(result),axis=1 ), np.array( labels , dtype=int)


def process_data(dataset_dir):
    user_id , reshaped_segments ,  labels = merge_data(
                                    path = dataset_dir, 
                                    w_s = WINDOW_SIZE, 
                                    stride = int(WINDOW_SIZE*(1-OVERLAP_RATE))  )
    return user_id , reshaped_segments, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_class", type=int, default=NUM_LABELS, help="number of classification labels")
    parser.add_argument("--min_sample", type=int, default=20, help="Min number of samples per user.")
    parser.add_argument("--n_ways", type=int, default=NUM_LABELS, help="n ways")
    parser.add_argument("--k_shots", type=int, default=20, help="k shot for train samples")
    parser.add_argument("--stdv", type=int, default=2, help="noise for choosing k shots and deleting n ways")
    parser.add_argument("--n_user", type=int, default=NUM_USERS,
                        help="number of local clients, should be muitiple of 10.")
    parser.add_argument("--dataset_dir", type=str, default='/data1/experiment/chengdongzhou/data/raw/USC-HAD')
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
        train_path      =  os.path.join( dir_path,f'uschad/{args.n_user}u_{args.sample_size}l_data/train/')
        test_path       =  os.path.join( dir_path,f'uschad/{args.n_user}u_{args.sample_size}l_data/test/')
        logging_path    =  os.path.join( dir_path,'uschad/VisualDataDistribution',f'{args.n_user}u_{args.sample_size}l_logging.json')
    else:
        train_path      =  os.path.join( dir_path,f'uschad/{args.k_shots}k_{args.n_user}b_{args.stdv}p_data/train/')
        test_path       =  os.path.join( dir_path,f'uschad/{args.k_shots}k_{args.n_user}b_{args.stdv}p_data/test/')
        logging_path    =  os.path.join( dir_path,'uschad/VisualDataDistribution',f'{args.k_shots}k_{args.n_user}b_{args.stdv}p_logging.json')

    dir_path = os.path.dirname(train_path)
    for path in (train_path,test_path,logging_path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    user_id , segments, labels = process_data(dataset_dir=args.dataset_dir)
    print('useid num is ', len(user_id), 'segments num is ' , len(segments))
    user_data_x,user_data_y = create_user_data( user_id ,NUM_USERS, segments , labels  ,stdv=args.stdv )
    if args.imbalance:
        random_creat_json(user_data_x , user_data_y , train_path , test_path, logging_path=logging_path,sample_size=args.sample_size,num_users=NUM_USERS,min_sample=args.min_sample)
    else:
        creat_json( user_data_x , user_data_y , train_path , test_path, k_shots=args.k_shots , stdv=args.stdv, logging_path=logging_path,num_users=NUM_USERS) 
    print("Finish Generating Samples")

if __name__ == '__main__':

    main()