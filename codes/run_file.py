import pandas as pd
import os
import sys
from joblib import Parallel, delayed, parallel_backend
#from dask.distributed import Client, progress
#from dask_jobqueue import HTCondorCluster
import pickle
import torch

# ignore warning
import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_columns = 50

# MCTS
#from mct import MC_node, MC_edge, MCFE_tree
from state import State
from agent import Agent
from game import FE
from value_network import Value_network
from nn_scorer import NN_scorer
#from pipeline import Pipeline
#from transforms import *

# set logger
import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)


#root = '/smartdata/hj7422/Documents/Workplace/MCTS_feature_engineering/'

def main():
    root = '../'
    #root_data = '../data/ibm_rl/classification/'
    filename = sys.argv[1]
    #fn = root + 'data/automlbenchmark/KDDCup09_appetency/'+filename+'.csv'
    fn = root + '/data/' + filename +'.csv'
    
    model = Value_network(input_size = 26, 
                           hidden_size = 32, 
                           batch_first = True, 
                           bidirectional = True, 
                           logger = logger)
    scorer = NN_scorer(model, num_epochs = 20, logger= logger)
#fn = '../data/ibm_rl/regression/openml_618.csv'
    print('@*@'*100)
    print(fn)
    print('@*@'*100)
    #scorer = NN_scorer(model, num_epochs = 20, logger= logger)
    fe = FE(fn, art = 'C', logger = logger, parallel = False, datatype = 'normal', subsampling=True, balanced_sampling=False) # sep = '\t', index_col = 0,
    ag = Agent(fe, scorer = scorer, max_depth = 6, logger = logger)
    root, init_perf, init_dat, top_five, top_features, buffer= ag.run(path = '../results/')
    record_result = {'file': fn, 'game': fe, 'agent': ag, 'top_five': top_five, 'buffer':buffer}
    #dict_out.append(record_result)
    with open(root + '/results/'+fn.split('/')[-1][:-4]+'.pkl', 'wb') as handle:
        pickle.dump(record_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(fe.record_top_five)





if __name__ == '__main__':
    main()
