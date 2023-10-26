
from FLAlgorithms.users.userbase import User
import torch.nn as nn

class UserLocal(User):
    def __init__(self, args, id, model, train_data, test_data  ):
        super().__init__(args, id, model, train_data, test_data  )


    def train(self, glob_iter, lr_decay=True, count_labels=False):
        self.model.train()
        for epoch in range(self.local_epochs):
            self.model.train()
            for X,y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output=self.model(X)['output']
                loss=self.loss(output, y)
                loss.backward()
                nn.utils.clip_grad_norm_( self.model.parameters(), 100 )
                self.optimizer.step()



