import numpy as np
import pandas as pd
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as fn
from sklearn.model_selection import train_test_split


class Value_network(torch.nn.Module):
    def __init__(self, input_size = 6, hidden_size = 32, batch_first = True, bidirectional = False, logger = None):
        super(Value_network, self).__init__()
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.input_size  = input_size # or onehot 18
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=self.hidden_size, 
                          batch_first=self.batch_first, 
                          bidirectional=self.bidirectional)
        if self.bidirectional:
            bi = 2
        else:
            bi = 1
        self.ln = nn.Sequential(
            nn.Linear(bi * hidden_size, 1), # or use softmax to get possibility of each action
            #nn.ReLU(),
            #nn.Linear(32, 1),
            nn.Sigmoid()
            #nn.Softmax()
        )
        
    def forward(self, inp):
        """
        inp, type of list of list of np.array with different length
        """
        #self.logger.info('- start to transform state to network input')
        # get maxlength and set container
        lengths = [len(i) for i in inp]
        maxlength = max(lengths)
        x = np.zeros((len(lengths), max(lengths), self.input_size))
        # padding
        for index, state in enumerate(inp):
            for i in range(len(state)):
                x[index, i, :] = state[i]
        x = torch.from_numpy(x)
        ## pack the padding input
        batch_first = self.rnn.batch_first
        # get sort indices
        lengths = torch.tensor(lengths, dtype=torch.long)
        sorted_seq_lengths, indices = torch.sort(lengths, descending=True)
        # get recover indices
        _, desorted_indices = torch.sort(indices, descending=False)
        # sort data
        if batch_first:
            x = x[indices, :, :]
        else:
            x = x[:, indices, :]
        # pack padded
        x = x.float() 
        packed_inputs = nn.utils.rnn.pack_padded_sequence(x,
                                              sorted_seq_lengths.cpu().numpy(),
                                              batch_first=batch_first)
        #self.logger.info('+ end to transform state')
        # run rnn
        #self.logger.info('- start to run rnn to get reward')
        res, state = self.rnn(packed_inputs)
        # unpacked
        padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=batch_first)
        # recover sequence
        if batch_first:
            desorted_res = padded_res[desorted_indices]
        else:
            desorted_res = padded_res[:, desorted_indices]
        out = self.ln(padded_res[ :, -1, :])
        #print(padded_res.shape)
        #self.logger.info('+ end with rnn model')
        return out
    
