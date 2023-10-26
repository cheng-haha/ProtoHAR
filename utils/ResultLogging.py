import logging
import os 
import h5py
import numpy as np
import pathlib
import time

def simple_read_data(alg):
    
    hf = h5py.File("./results/"+'{}.h5'.format(alg), 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_glob_f1 = np.array(hf.get('rs_f1')[:])
    rs_glob_auc = np.array(hf.get('rs_auc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    print(alg)
    return rs_glob_acc,rs_glob_f1,rs_glob_auc,rs_train_acc, rs_train_loss 

def get_training_data_value(num_users=5, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[],beta=[],algorithms_list=[], batch_size=[], dataset="", k= [] , personal_learning_rate = [] , niidstryle = 'balance',total_users=0):
    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_f1 = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_auc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        string_learning_rate = str(learning_rate[i])  
        string_learning_rate = string_learning_rate + "_" +str(beta[i]) + "_" +str(lamb[i])
        try:
            if type(total_users) is list:
                current_alg = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users[i]) + "u"+"_" + str(total_users[i]) +"tu" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i])+'_'+str(niidstryle)
            elif type(niidstryle) is list:
                current_alg = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users[i]) + "u"+"_" + str(total_users) +"tu" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i])+'_'+str(niidstryle[i])
            else:
                current_alg = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users[i]) + "u"+"_" + str(total_users) +"tu" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i])+'_'+str(niidstryle)

            glob_acc[i, :],glob_f1[i, :],glob_auc[i, :], train_acc[i, :] , train_loss[i, :] = np.array(
                simple_read_data(dataset +"_"+ current_alg + "_avg"))[:, :Numb_Glob_Iters]
            algs_lbl[i] = algs_lbl[i]
        except:
            # print("!!!")
            current_alg = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users[i]) + "u" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i])+'_'+str(niidstryle)

            glob_acc[i, :],glob_f1[i, :],glob_auc[i, :], train_acc[i, :] , train_loss[i, :] = np.array(
                simple_read_data(dataset +"_"+ current_alg + "_avg"))[:, :Numb_Glob_Iters]
            algs_lbl[i] = algs_lbl[i]
           
    return glob_acc,glob_f1,glob_auc, train_acc, train_loss


def get_diffglobaliters_training_data_value(num_users=5, loc_ep1=5, Numb_Glob_Iters=[], lamb=[], learning_rate=[],beta=[],algorithms_list=[], batch_size=[], dataset="", k= [] , personal_learning_rate = [] , niidstryle = 'balance',total_users=0):
    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters[0]))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters[0]))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters[0]))
    glob_f1 = np.zeros((Numb_Algs, Numb_Glob_Iters[0]))
    glob_auc = np.zeros((Numb_Algs, Numb_Glob_Iters[0]))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        string_learning_rate = str(learning_rate[i])  
        string_learning_rate = string_learning_rate + "_" +str(beta[i]) + "_" +str(lamb[i])
        current_alg = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users[i]) + "u"+"_" + str(total_users[i]) +"tu" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i])+'_'+str(niidstryle)
        Numb_Glob_Iter = Numb_Glob_Iters[i]
        glob_acc[i, :Numb_Glob_Iter],glob_f1[i, :Numb_Glob_Iter],glob_auc[i, :Numb_Glob_Iter], train_acc[i, :Numb_Glob_Iter] , train_loss[i, :Numb_Glob_Iter] = np.array(
            simple_read_data(dataset +"_"+ current_alg + "_avg"))[:, :Numb_Glob_Iter]
    
           
    return glob_acc,glob_f1,glob_auc, train_acc, train_loss


def get_all_training_data_value(num_users=100, loc_ep1=5, total_users=30,Numb_Glob_Iters=10, lamb=0, learning_rate=0,beta=0,algorithms="", batch_size=0, dataset="", k= 0 , personal_learning_rate =0 ,times = 5,niid='n'):
    train_acc = np.zeros((times, Numb_Glob_Iters))
    avg_f1 = np.zeros((times, Numb_Glob_Iters))
    avg_auc = np.zeros((times, Numb_Glob_Iters))
    train_loss = np.zeros((times, Numb_Glob_Iters))
    avg_acc = np.zeros((times, Numb_Glob_Iters))
    algorithms_list  = [algorithms] * times
    for i in range(times):
        string_learning_rate = str(learning_rate)  
        string_learning_rate = string_learning_rate + "_" +str(beta) + "_" +str(lamb)
        algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate \
            + "_" + str(num_users) + "u" +"_" + str(total_users)+"tu"\
            + "_" + str(batch_size) + "b"  "_" +str(loc_ep1) +"_" +str(niid) +  "_" +str(i)

        avg_acc[i, :],avg_f1[i, :],avg_auc[i, :], train_acc[i, :] , train_loss[i, :] = np.array(
            simple_read_data(dataset +"_"+ algorithms_list[i]))[:, :Numb_Glob_Iters]
    return avg_acc,avg_f1,avg_auc, train_acc, train_loss


def get_data_label_style(input_data = [], linestyles= [], algs_lbl = [], lamb = [], loc_ep1 = 0, batch_size =0):
    data, lstyles, labels = [], [], []
    for i in range(len(algs_lbl)):
        data.append(input_data[i, ::])
        lstyles.append(linestyles[i])
        labels.append(algs_lbl[i]+str(lamb[i])+"_" +
                      str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b")

    return data, lstyles, labels
def write_result_to_csv(**kwargs):

    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Time,"
            "DataSet, "
            "Algorithm, "
            "epochs, "
            "learning_rate, "
            "num_users, "
            "toal_users, "
            "beta, "
            "lamda, "
            "batch_size, "
            "local_epochs, "
            "times, "
            "niid, "
            "BestTestAcc, "
            "BestTestF1, "
            "BestTestAUC, "
            "MeanTestAcc, "
            "MeanTestF1, "
            "MeanTestAUC,\n "
            # "std,\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{dataset}, "
                "{algorithm},"
                "{epochs}, "
                "{learning_rate}, "
                "{num_users}, "
                "{total_users}, "
                "{beta}, "
                "{lamda}, "
                "{batch_size}, "
                "{local_epochs}, "
                "{times}, "
                "{niid}, "
                "{BestTestAcc}, "
                "{BestTestF1}, "
                "{BestTestAUC}, "
                "{MeanTestAcc}, "
                "{MeanTestF1}, "
                "{MeanTestAUC},\n "
                # "{std},\n"

            ).format(now=now, **kwargs)
        )         
def average_data(logging , num_users=100,total_users=30, loc_ep1=5, Numb_Glob_Iters=10, lamb="", learning_rate="", beta="", algorithms="", batch_size=0, dataset = "", k = "", personal_learning_rate = "", times = 5,niid = 'n'):
    glob_acc,glob_f1,glob_auc,train_acc, train_loss = \
        get_all_training_data_value( 
                                    num_users, 
                                    loc_ep1, 
                                    total_users,
                                    Numb_Glob_Iters, 
                                    lamb, 
                                    learning_rate, 
                                    beta, 
                                    algorithms, 
                                    batch_size, 
                                    dataset, 
                                    k, 
                                    personal_learning_rate,
                                    times , 
                                    niid= niid
                                    )
    glob_acc_data = np.average(glob_acc, axis=0)
    glob_f1_data = np.average(glob_f1,axis=0)
    glob_auc_data = np.average(glob_auc,axis=0)
    train_acc_data = np.average(train_acc, axis=0)
    train_loss_data = np.average(train_loss, axis=0)
    max_accurancy = []
    max_f1 = []
    max_auc = []
    for i in range(times):
        max_accurancy.append(glob_acc[i].max())
        max_f1.append(glob_f1[i].max())
        max_auc.append(glob_auc[i].max())
    MaxTestAcc = np.mean(max_accurancy)
    MaxF1 = np.mean(max_f1)
    MaxAUC = np.mean(max_auc)
    MeanTestAcc ,accstd= np.mean(glob_acc_data) , np.std(glob_acc_data)
    MeanF1 , f1std= np.mean(glob_f1_data) , np.std(glob_f1_data)
    MeanAUC , aucstd = np.mean(glob_auc_data) , np.std(glob_auc_data)
    # print("std:", teststd)
    logging.info(f"Best acc:{MaxTestAcc:.5f}  Best F1: {MaxF1:.5f}  Best AUC:{MaxAUC:.5f}")
    logging.info(f"Mean acc:{MeanTestAcc:.5f}±{accstd:.5f}  Mean F1: {MeanF1:.5f}±{f1std:.5f}  Mean AUC:{MeanAUC:.5f}±{aucstd:.5f}")
    write_result_to_csv( dataset = dataset ,algorithm =algorithms,  epochs = Numb_Glob_Iters , \
                        learning_rate = learning_rate , num_users = num_users,total_users=total_users,
                        beta = beta , lamda = lamb , batch_size = batch_size , local_epochs =  loc_ep1 , 
                        times = times , niid = niid ,  \
                        BestTestAcc = str( np.around(MaxTestAcc,5)),
                        BestTestF1 = str( np.around(MaxF1,5)),
                        BestTestAUC = str( np.around(MaxAUC,5)),
                        MeanTestAcc = str( np.around(MeanTestAcc,3))+'±'+ str(np.around(accstd,3)), \
                        MeanTestF1 = str(np.around(MeanF1,3))+'±'+ str(np.around(f1std,3)) , \
                        MeanTestAUC = str(np.around(MeanAUC,3)) +'±'+str(np.around(aucstd,3))  
                        )  
    alg = dataset + "_" + algorithms
    alg = alg + "_" + str(learning_rate) + "_" + str(beta) + "_" + str(lamb) + "_" + str(num_users) + "u" +"_" + str(total_users) +"tu"+ "_" + str(batch_size) + "b" + "_" + str(loc_ep1)
    alg = alg + '_'+ str(niid) +"_" + "avg"
    if (len(glob_acc) != 0 &  len(train_acc) & len(train_loss)) :
        with h5py.File("./results/"+'{}.h5'.format(alg,loc_ep1), 'w') as hf:
            hf.create_dataset('rs_glob_acc', data=glob_acc_data)
            hf.create_dataset('rs_f1', data=glob_f1_data)
            hf.create_dataset('rs_auc', data=glob_auc_data)
            hf.create_dataset('rs_train_acc', data=train_acc_data)
            hf.create_dataset('rs_train_loss', data=train_loss_data)
            hf.close()


def loadLogger(args , savefileLength = 100 ,removefiles = 2 ):
    '''
    Each method only keeps ten logs , savefileLength determines how many files to keep , 
    removefiles determines how many files to delete
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='[%(asctime)s]  %(message)s',
        datefmt='%m-%d %H:%M')

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)

    logger.addHandler(sHandler)
    path = os.path.dirname(os.path.dirname(__file__))
    if args.savelogging:
        work_dir = os.path.join(path,args.work_dir,args.dataset,args.algorithm)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        files = sorted(os.listdir(work_dir))
        [ os.remove(os.path.join(work_dir,file)) for file in files[:removefiles]] if len(files) >= savefileLength else []
        path = os.path.join(work_dir,f'experiment_arguments-{args.learning_rate}lr-{args.beta}-{args.lamda}-{args.num_users}u-{args.total_users}tu-{args.batch_size}b-{args.local_epochs}le-{args.niid}-{args.times}t.log')
        
        fHandler = logging.FileHandler(path, mode='w+')
        fHandler.setLevel(logging.DEBUG)
        fHandler.setFormatter(formatter)

        logger.addHandler(fHandler)
        logger.info(str(args))
    return logger