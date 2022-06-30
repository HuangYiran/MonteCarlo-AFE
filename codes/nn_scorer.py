from value_network import Value_network

import numpy as np
import random
import torch


class NN_scorer():
    def __init__(self, network = None, 
                 loss = 'L1', #CrossEntropyLoss(), 
                 optimizer = 'Adam',  
                 batch_size = 10, 
                 num_epochs = 2,
                 logger = None):
        super(NN_scorer, self).__init__()
        self.logger = logger
        if network:
            self.model = network
        else:
            self.model = Value_network(input_size = 6, 
                                       hidden_size = 32, 
                                       batch_first = True, 
                                       bidirectional = True, 
                                       logger = self.logger)
        self.loss = self._load_loss(loss)
        self.optimizer = self._load_optimizer(optimizer)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
    
    def fit(self, Xs, ys):
        """
        Xs: type of list of list of np.array
        ys: type of list of float
        """
        # load the collected data
        Xs = Xs # list of list of np.array
        ys = ys # list of int
        ys = torch.FloatTensor(ys)
        # train the model
        for epoch in range(self.num_epochs):
            #print(epoch)
            #scheduler.step()
            self.model.train()
            train_loss = 0
            counter = 0
            random.seed(epoch)
            random.shuffle(Xs)
            random.shuffle(ys)
            #for batch_idx, dat in enumerate(dl_train):
            self._epoch(Xs, ys, epoch)
    
    def predict(self, Xs):
        self.model.eval()
        out = self.model(Xs)
        return out.squeeze().detach().numpy()
    
    def evaluate(self, Xs, ys):
        self.model.eval()
        out = self.model(Xs)
        ys = torch.FloatTensor(ys)
        lo = self.loss(out.squeeze(), ys.squeeze())
        if self.logger:
            self.logger.info('Evaluate loss: %f'%(lo))
        return lo
        
    def _epoch(self, Xs, ys, epoch = 0):
        self.model.train()
        random.shuffle(Xs)
        random.shuffle(ys)
        los = []
        for i in np.arange(0, len(Xs), self.batch_size):
            tx = Xs[i: i + self.batch_size]
            ty = ys[i: i + self.batch_size]
            self.optimizer.zero_grad()
            out = self.model(tx)
            lo = self.loss(out.squeeze(), ty.squeeze())
            lo.backward()
            los.append(float(lo.detach().numpy()))
            self.optimizer.step()
            #if self.logger:
            #    self.logger.info('Epoch %d : \t %d/%d (i/len(Xs)) \t loss: %f'%(epoch, i, len(Xs), lo))
        if self.logger:
            self.logger.info('Epoch %d:\t mean loss: %f, \t std loss: %f'%(epoch, np.mean(los), np.std(los)))
            self.logger.info('='*90)
    
    def _load_loss(self, loss):
        if loss == 'L1':
            loss = torch.nn.L1Loss()
        elif loss == 'CrossEntropy':
            loss = torch.nn.CrossEntropyLoss()
        else:
            if self.logger:
                self.logger.info('Target loss function %s not found, use default loss function: %s'%(loss, 'L1Loss'))
            loss = torch.nn.L1Loss()
        return loss
    
    def _load_optimizer(self, optimizer):
        if optimizer == 'Adam':
            opt = torch.optim.Adam(self.model.parameters(), lr = 1e-3)
        else:
            if self.logger:
                self.logger.info('Target optimizer %s not found, use default optimizer: %s'%(optimizer, 'Adam'))
            opt = torch.optim.Adam(self.model.parameters(), lr = 1e-3)
        return opt
