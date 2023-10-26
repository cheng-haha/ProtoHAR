from FLAlgorithms.users.userlocal import UserLocal
from FLAlgorithms.servers.serverbase import Server
from utils.decorator import LoggingUserAcc

class Local(Server):
    def __init__(self, args, model, times):
        #dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
        #         local_epochs, num_users, K, personal_learning_rate, times):
        super().__init__(args, model, times)#dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                         #local_epochs, num_users, times)
        self.SetClient(args,UserLocal)

    @LoggingUserAcc
    def train(self):
        #仅仅分发一次参数
        
        self.send_parameters()
        print(self.model)
        for glob_iter in range(self.num_glob_iters):
            self.logging.info(f"-------------Round number:{glob_iter} -------------")
            self.selected_users = self.select_users(glob_iter,self.num_users)
            # self.GeneralizationEvaluate()

            for user in self.selected_users: # allow selected users to train
                user.train(glob_iter )
            
            self.evaluate()
        if self.vis:
            self.plot_2d_features(self.users[0].model , self.users[0].testloaderfull )
        if self.vis_proto:
            self.save_protos()
        self.save_results()
        self.save_model()