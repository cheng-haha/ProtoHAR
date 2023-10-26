import numpy as np
def LoggingUserAcc(func):
    allUserAcc = None
    def wrapper(self,*args,**kwargs):
        nonlocal allUserAcc
        func(self,*args,**kwargs)
        max_acc_idx = self.rs_glob_acc.index(max(self.rs_glob_acc))
        # print(f'{user.accuracy}')
        if self.times == 0:
            # shape is one dim
            allUserAcc = np.array([user.accuracy[max_acc_idx] for user in self.users])
        elif self.times == 1:
            # shape is two dim
            if allUserAcc is not None:
                allUserAcc = np.stack((allUserAcc , np.array([user.accuracy[max_acc_idx] for user in self.users])))
            else:
                allUserAcc = np.array([user.accuracy[max_acc_idx] for user in self.users])
        else:
            if allUserAcc is not None:
                allUserAcc = np.append(allUserAcc , np.expand_dims(np.array([user.accuracy[max_acc_idx] for user in self.users]) , axis = 0 ) , axis= 0 )
            else:
                allUserAcc = np.array([user.accuracy[max_acc_idx] for user in self.users])
        self.logging.info('----------------All Users Acc--------------------')
        self.logging.info(f'Logging all users best acc with n times(You should record last result):\n {allUserAcc if self.times == 0 else np.mean(allUserAcc,axis=0)}\n, bset acc is {np.mean(allUserAcc)}')
        self.logging.info('-------------------------------------------------')
        return allUserAcc if self.times == 0 else np.mean(allUserAcc,axis=0)
    return wrapper