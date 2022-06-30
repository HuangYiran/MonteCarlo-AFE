from abc import ABCMeta, abstractmethod

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as fn
import traceback

from scipy.stats import  ttest_ind
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR, SVC, LinearSVC, LinearSVR
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend

from pipeline import Pipeline
from state import State
from mct import MC_node

class Game(metaclass = ABCMeta):
    @abstractmethod
    def get_available_actions(self, state):
        """
        input:
          state, can be any kind of type
        output:
          actions: list of action, action should be string
        """
        pass
    @abstractmethod
    def get_current_state(self):
        pass
    @abstractmethod
    def get_reward(self, state):
        """
        evaluate the inputted state
        """
        pass
    @abstractmethod
    def is_done(self, state):
        """
        input:
          state
        output:
          out: if the game is done return the result, else return 0
        """
        pass
    @abstractmethod
    def restore_game(self):
        pass
    @abstractmethod
    def simulate_action(self):
        pass
    @abstractmethod
    def take_action(self):
        pass
    
# because of balanced sampling, the initial dataset are different according to timestamp
class FE(Game):
    def __init__(self, dataname, art = 'C', sep = None, header = True, index_col = None, datatype = 'normal', logger = None, parallel = False, subsampling=True, balanced_sampling= True, seed = 0):
        """
        ===========
        input:
        ===========
        dataname: type of string: 
            - the path to load the target dataset
        art: type of string, in ['C', 'R']
            - set the art of the problem, either classification of regression
        sep, head, index_col: type of string, boolean, int, default = [None, True, None]
            - parameters to load the data with pandas.read_csv
        datatype, type of string, default == 'normal'
            - specify the type of the data, following type are available ['normal', 'picture', 'timeserie']
            - datatype to be added later, include ['video', ...]
        """
        super(FE, self).__init__()
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
        self.logger.info('##### Init FE')
        self.seed = seed
        self.rf_seed = 0
        self.dataname = dataname
        self.type = datatype
        self.lookup = {} # transformation name to index
        self.lookup2 = {} # transformation from name to onehot encoding
        self.lookup2r = {} # transformation from onehot encoding to name 
        self.actions = self._load_available_actions() 
        self.art = art # C (classification) or R (regression)
        self.numeric_cols = [] # initialized in load_data
        self.restore_game() # set current_state to State([])
        self.dat = self._load_data(dataname, sep, header, index_col, subsampling=subsampling, balanced_sampling=balanced_sampling) # load data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dat.iloc[:, :-1], self.dat.iloc[:, -1], test_size = 0.33, random_state = self.seed)
        self.init_dat = [self.dat.iloc[:, :-1], self.dat.iloc[:, -1]]
        self.init_performance = self._get_init_performance() # get performance of the original dataset, return a list
                                                             # if num_cols>2000, do feature selection first
        print(f'Init performance: {np.mean(self.init_performance)}')
        self.pipeline = Pipeline(self.dat, self.art, logger = self.logger, seed = self.seed) # init test pipeline
        self.record_top_five = []
        self.record_best_features = {}
        self.buffer = pd.DataFrame(columns = ['X', 'y'])
        self.parallel = parallel

    def get_logger(self):
        return self.logger
    def get_available_actions(self, node = None):
        self.logger.info('Get availabel action')
        return list(self.actions)
    def get_current_state(self):
        self.logger.info('Get current state')
        return self.current_state
    def get_init_performance(self):
        return self.init_performance
    def get_original_state(self):
        self.logger.info('Get original state')
        # start with null
        return State([])
    
    def get_reward(self, node):
        X_train, X_test, y_train, y_test = self._get_dataset_from_node(node)
        out, fe = self._get_reward_for_dataset(X_train, X_test, y_train, y_test)
        self.logger.info('Performance of the given node is: %s'%(str(out)))
        return (out, fe)
    
    def get_reward_cv(self, node, cv = 4, shuffle = False):
        # split training set to train and val sets, n fold on these sets.
        X_train, X_test, y_train, y_test = self._get_dataset_from_node(node) # X is type of dataframe, y is type of np.array
        X_train = pd.concat([self.X_train, X_train], axis = 1)
        X_test = pd.concat([self.X_test, X_test], axis = 1)
        #xs = pd.concat([X_train, X_test])
        #ys = np.concatenate([y_train, y_test])
        out = []
        seed_init = 0
        if shuffle:
            seed_init = random.randint(0,10)
        for i in range(cv):
            #X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size = 0.33, random_state = seed_init+i)
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = seed_init+i)
            tmp, fe = self._get_reward_for_dataset(X_train, X_test, y_train, y_test)
            out.append(tmp)
        self.logger.info('Performance of the given node is: %s'%(str(out)))
        return (np.mean(out), fe)
    
    def get_reward_cv_on_test_set(self, node):
        X_train, X_test, y_train, y_test = self._get_dataset_from_node(node) # X is type of dataframe, y is type of np.array
        tmp, fe = self._get_reward_for_dataset(X_train, X_test, y_train, y_test)
        return tmp, fe
        
    def get_reward_cv_on_test_set2(self, node, cv = 4, shuffle = False): # !!!abandon!!!
        # train with training set and test with test set
        X_train, X_test, y_train, y_test = self._get_dataset_from_node(node) # X is type of dataframe, y is type of np.array
        X_train = pd.concat([self.X_train, X_train], axis = 1)
        X_test = pd.concat([self.X_test, X_test], axis = 1)
        #xs = pd.concat([X_train, X_test])
        #ys = np.concatenate([y_train, y_test])
        out = []
        seed_init = 0
        if shuffle:
            seed_init = random.randint(0,10)
        for i in range(cv):
            _, X_test, _, y_test = train_test_split(X_test, y_test, test_size = 0.80, random_state = seed_init+i)
            tmp, fe = self._get_reward_for_dataset(X_train, X_test, y_train, y_test)
            out.append(tmp)
        self.logger.info('Performance of the given node is: %s'%(str(out)))
        return (np.mean(out), fe)  
    
    def get_avg_reward_for_multi_nodes(self, nodes):
        if self.parallel:
            return self._get_avg_reward_for_multi_nodes_parallel(nodes)
        else:
            return self._get_avg_reward_for_multi_nodes_single_core(nodes)

    def get_top_five(self):
        return self.record_top_five
    
    def collect_dataset_from_actions(self, actions):
        # actions: list of action(string)
        X_train_collect = []
        X_test_collect = []
        for i in actions:
            self.logger.info('- Start to do feature transformation for action %s'%(i)) 
            #self.pipeline.set_transform(state.state)
            X_train, X_test, y_train, y_test = self.pipeline.run([i])
            X_train_collect.append(X_train)
            X_test_collect.append(X_test)
        X_train = pd.concat(X_train_collect, axis = 1)
        X_test = pd.concat(X_test_collect, axis = 1)
        X_train = X_train.loc[:,~X_train.columns.duplicated()]
        X_test = X_test.loc[:,~X_test.columns.duplicated()]
        self.logger.info('+ end with feature transformation')
        return X_train, X_test, y_train, y_test
    
    def restore_game(self):
        self.current_state = self.get_original_state()
        
    def simulate_action(self, cuu_state, action):
        """
        input:
            cuu_state, type of State
        output:
            state, type of State
        """
        cuu_state = copy.deepcopy(cuu_state)
        #cuu_state = cuu_state.get_state()
        state = cuu_state.get_state()
        state.append(action)
        return State(state)
    
    def states_to_onehot(self, states):
        """
        states, type of list of state.state
        state.state is list of names
        """
        out = []
        for i in states:
            tmp = []
            for j in i.state:
                tmp.append(self._name_to_onehot(j))
            out.append(tmp)
        return out
        
    def take_action(self, action):
        # change state after taking the action
        self.current_state.take_action(action)
    
    def update_init_performance_with_data_in_buffer(self):
        # select the best score in buffer
        node = self.buffer.sort_values('y', ascending=False).iloc[0, 0]
        node = MC_node(State(node.replace(' ', '').split(',')))
        # run cv
        X_train, X_test, y_train, y_test = self._get_dataset_from_node(node) # X is type of dataframe, y is type of np.array
        xs = pd.concat([X_train, X_test])
        ys = np.concatenate([y_train, y_test])
        out = []
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size = 0.33, random_state = i)
            tmp, fe = self._get_reward_for_dataset(X_train, X_test, y_train, y_test)
            out.append(tmp)
        if self.init_performance < out:
            self.logger.warn('the new init_performance is: %s'%(str(out)))
            self.init_performance = out
            self.init_data = [xs, ys]
        return 1
    
    def update_init_performance_with_node(self, node, reward):
        if reward > np.mean(self.init_performance):
            X_train, X_test, y_train, y_test = self._get_dataset_from_node(node) # X is type of dataframe, y is type of np.array
            xs = pd.concat([X_train, X_test])
            ys = np.concatenate([y_train, y_test])
            out = []
            for i in range(10):
                _, X_test, _, y_test = train_test_split(X_test, y_test, test_size = 0.8, random_state = i)
                tmp, fe = self._get_reward_for_dataset(X_train, X_test, y_train, y_test)
                out.append(tmp)
            self.logger.warning(f'Mean init is: {np.mean(self.init_performance)}, mean of new node is: {np.mean(out)}')
            if np.mean(self.init_performance) < np.mean(out):
                self.logger.warn('the new init_performance is: %s'%(str(out)))
                self.init_performance = out
                self.init_data = [xs, ys]
                return node
            else:
                return None
        else:
            return None
        
    def _add_item_to_buffer(self, name, value):
        ind = self.buffer.shape[0]
        self.buffer.loc[ind, 'X'] = name
        self.buffer.loc[ind, 'y'] = value
        return 1
    
    def _add_item_to_record_best_features(self, fe, reward):
        name = fe[0]
        value = fe[1]
        if reward > np.mean(self.init_performance):
            if name not in self.record_best_features.keys():
                self.logger.warn('Add the best feature in set to the best features record.')
                self.record_best_features[name] = value
            if len(self.record_best_features) > 2000:
                self.logger.warn('Since the set is full, remove a feature')
                self.record_best_features.pop()
            
    def _add_item_to_record_top_five(self, name, value):
        flag = True
        for i in np.arange(len(self.record_top_five), 0, -1):
            if self.record_top_five[i - 1][1] < value:
                continue
            self.record_top_five.insert(i, (name, value))
            flag = False
            break
        if flag:
            self.record_top_five.insert(0, (name, value))
        if len(self.record_top_five) > 5:
            del(self.record_top_five[-1])
            self.logger.info('Update top five record, the best five records are:')
        for i in range(len(self.record_top_five)):
            self.logger.info('- %s:\t%s'%(self.record_top_five[i][0], str(self.record_top_five[i][1])))
        
    def _data_preprocessing(self, dat, art = 'R', y = None, logger = None, remove = True):
        """
        Encoding + remove columns with more than 1/2 na if remove == True + remove columns with all na + imputation
        if art == 'C', will do labelencoding first for the target column
        ================
        Parameter:
        ================
        dat - type of DataFrame
        art - type of string
            either C for classifcation of R for regression. indicates the type of problem 
        y - type of string
            the name of the target columns; if None, set the last columns of the data set as target
        logger - type of Logger
        remove - type of bollean
            whether remove the columns with na value more than half length or not
        =================
        Output
        =================
        dat - type of Dataframe 
            the dataframe after preprocessing
        cols - type of list of string
            the name of the numerical columns
        """
        if logger:
            logger.info('- Start preprocessing function')
        dat = dat.reset_index().iloc[:, 1:]
        if art == 'C':
            if logger:
                logger.info('- start to label target feature y for classification task')
            le = LabelEncoder()
            dat.iloc[:, -1] = le.fit_transform(dat.iloc[:, -1])
            # check balanced
            #dat = self._balanced_sampling(dat)
            if logger:
                logger.info('+ end with label encoding the target feature')
        if remove:
            # remove columns with more than 1/2 na
            dat = dat.loc[:, dat.isna().sum()/len(dat) < .5]
            if logger:
                logger.info('Following features are removed from the dataframe because half of their value are NA: %s'%(dat.columns[dat.isna().sum()/len(dat) > .5].to_list()))
        # Encoding
        oe = OneHotEncoder(drop = 'first')
        ## get categorical columns
        if y:
            dat_y = dat[[y]]
            cols = dat.columns.to_list()
            cols.remove(y)
            dat_x = dat[cols]
        else:
            dat_y = dat[[dat.columns[-1]]]
            dat_x = dat[dat.columns[:-1]]
        dat_category = dat_x.select_dtypes(include = ['object'])
        # get kterm of cate features
        for i in dat_category.columns:
            # save output to dat 
            tmp = dat_x[i].value_counts()
            dat_x[i + '_kterm'] = dat_x[i].map(lambda x: tmp[x] if x in tmp.index else 0)
        # float columns including the k term cols
        dat_numeric =  dat_x.select_dtypes(include=['float64', 'int'])
        # onehot encoding and label encoding
        dat_cate_onehot = dat_category.iloc[:, dat_category.apply(lambda x: len(x.unique())).values<8] 
        dat_cate_label = dat_category.iloc[:, dat_category.apply(lambda x: len(x.unique())).values>=8]
        flag_onehot = False
        flag_label = False
        ## oe
        if dat_cate_onehot.shape[1] > 0:
            if logger:
                logger.info('- start to do onehot to the following categoric features: %s'%(str(dat_cate_onehot.columns.to_list())))
            dat_onehot = pd.DataFrame(oe.fit_transform(dat_cate_onehot.astype(str)).toarray(), columns = oe.get_feature_names(dat_cate_onehot.columns))
            if logger:
                logger.info('+ end with onehot')
            flag_onehot = True
        else:
            dat_onehot = None
        ## le
        if dat_cate_label.shape[1] > 0:
            if logger:
                logger.info('- start to do label encoding to the following categoric features: %s'%(str(dat_cate_label.columns.to_list())))
            dat_cate_label = dat_cate_label.fillna('NULL')
            dat_label = pd.DataFrame(columns = dat_cate_label.columns)
            for i in dat_cate_label.columns:
                le = LabelEncoder()
                dat_label[i] = le.fit_transform(dat_cate_label[i].astype(str))
            if logger:
                logger.info('+ end with label encoding')
            flag_label = True
        else:
            dat_label = None
        # scaling
        ## combine
        if flag_onehot and flag_label:
            dat = pd.concat([dat_numeric, dat_onehot, dat_label], axis = 1)
        elif flag_onehot:
            dat = pd.concat([dat_numeric, dat_onehot], axis = 1)
        elif flag_label:
            dat = pd.concat([dat_numeric, dat_label], axis = 1)
        else:
            dat = dat_numeric
        dat = pd.concat([dat, dat_y], axis = 1)
        # imputation
        dat = dat.dropna(axis = 1, how = 'all')
        if dat.isna().sum().sum() > 0:
            if logger:
                logger.info('- na value exist, start to fill na with iterative imputer: '+ str(dat.isna().sum().sum()))
            # include na value, impute with iterative Imputer or simple imputer
            columns = dat.columns
            imp = IterativeImputer(max_iter=10, random_state=0)
            # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            dat = imp.fit_transform(dat)
            dat = pd.DataFrame(dat, columns = columns)
        dat_numeric =  dat.iloc[:, :-1].select_dtypes(include=['float64', 'int'])
        if logger:
            logger.info('+ end with fill na')
        return dat, dat_numeric.columns
    
    def _balanced_sampling(self, dat):
        # upsampling
        self.logger.info('Balanced sampling')
        subsample = []
        num_each_classes = dat.iloc[:, -1].value_counts().values
        if num_each_classes.std()*1.0/num_each_classes.mean() < 0.1:
            self.logger.info('- The given data is balance.')
            # the dataset is balanced
            return dat
        self.logger.info('- Given dataset is unbalance')
        self.logger.info('- Sampling data from each class to generate a new dataset')
        n_smp = num_each_classes.max()
        for label in dat.iloc[:, -1].value_counts().index:
            samples = dat[dat.iloc[:, -1] == label]
            num_samples = len(samples)
            index_range = range(num_samples)
            indexes = list(np.random.choice(index_range, size=num_samples, replace=False)) # take all from the set
            indexes2 = list(np.random.choice(index_range, size =n_smp-num_samples, replace=True)) # add random items
            indexes.extend(indexes2)
            subsample.append(samples.iloc[indexes, :])
        self.logger.info('+ end with sampling')
        out = pd.concat(subsample)
        out = out.sample(frac=1).reset_index(drop=True) # shuffle and re index
        return out
    
    def _feature_selection(self, dat):
        self.logger.info('the number of columns exceed max number(%d), try columns selection first'%(2000))
        # feature selection
        if self.art == 'C':
            clf = RandomForestClassifier(random_state = self.rf_seed).fit(dat.iloc[:, :-1], dat.iloc[:, -1])
        else:
            clf = RandomForestRegressor(random_state = self.rf_seed).fit(dat.iloc[:, :-1], dat.iloc[:, -1])
        fs = SelectFromModel(clf, threshold= 'mean', prefit=True)
        supp = fs.get_support()
        df = pd.DataFrame(fs.transform(dat.iloc[:, :-1]), columns = dat.iloc[:, :-1].columns[supp], index = dat.index)
        df['target'] = dat.iloc[:, -1]
        self.logger.info('End with columns selection, number of columns now is: %s'%(str(dat.iloc[:, :-1].shape[1])))
        return df
    
    def _get_avg_reward_for_multi_nodes_parallel(self, nodes):
        #with parallel_backend('dask'):
        outs = Parallel(n_jobs = -1)(delayed(self.get_reward_cv_on_test_set)(node) for node in tqdm(nodes))
        tmp2 = []
        for node, out in zip(nodes, outs):
            tmp = out[0]
            fe = out[1]
            tmp2.append(tmp)
            self._add_item_to_record_top_five(str(node.id), tmp)
            self._add_item_to_record_best_features(fe, tmp)
            self._add_item_to_buffer(str(node.id), tmp)
        self.logger.info('Avg performance of the given node is: %s'%(str(tmp2)))
        stat, p = ttest_ind(tmp2, self.init_performance)
        if p < 0.05:
            reward = np.ceil((np.mean(tmp2) - np.mean(self.init_performance))/0.01) #np.std(self.init_performance))
            self.logger.info('Mean perf is: %f, Mean init is: %f, Std init is: %f ==> reward: %f'%(np.mean(tmp2), np.mean(self.init_performance), np.std(self.init_performance), reward))
            if reward <=0:
                return 0, np.mean(out)
            else:
                return min(5, reward), np.mean(out)#1 #reward
        else:
            reward = 0
            self.logger.info('Mean perf is: %f, Mean init is: %f, Std init is: %f ==> reward: %f'%(np.mean(tmp2), np.mean(self.init_performance), np.std(self.init_performance), reward))
            return 0, np.mean(out)
        
    def _get_avg_reward_for_multi_nodes_single_core(self, nodes):
        out = []
        ids = []
        for node in nodes:
            #X_train, X_test, y_train, y_test = self._get_dataset_from_node(node)
            #tmp, fe = self._get_reward_for_dataset(X_train, X_test, y_train, y_test)
            tmp, fe = self.get_reward_cv_on_test_set(node)
            self.logger.info('Performance of the given node %s is: %s'%(str(node.id), str(tmp)))
            self._add_item_to_record_top_five(str(node.id), tmp)
            self._add_item_to_record_best_features(fe, tmp)
            self._add_item_to_buffer(str(node.id), tmp)
            ids.append(str(node.id))
            out.append(tmp)
        # find long
        def _long_str(li):  # 支持中文字符    
            result=''    
            for i in zip(*li):        
                if len(set(i))==1:            
                    result +=i[0]        
                else:            
                    break    
            return result
        prefix = _long_str(ids)
        if len(prefix) != len(ids[0]):
            prefix = ', '.join(prefix.split(', ')[:-1])
        self._add_item_to_buffer(prefix, np.mean(out))
        #out = np.mean(out)
        self.logger.warn('Performances of the given node is: %s'%(str(out)))
        stat, p = ttest_ind(out, self.init_performance)
        #if p < 0.05:
        if np.mean(out) > np.mean(self.init_performance):
            reward = np.ceil((np.mean(out) - np.mean(self.init_performance))/0.01)#np.std(self.init_performance))
            self.logger.warn('Mean perf is: %f, Mean init is: %f, Std init is: %f ==> reward: %f'%(np.mean(out), np.mean(self.init_performance), np.std(self.init_performance), reward))
            return min(3, reward), np.mean(out)#1 #reward
        else:
            reward = np.floor((np.mean(out) - np.mean(self.init_performance))/0.02)
            self.logger.warn('Mean perf is: %f, Mean init is: %f, Std init is: %f ==> reward: %f'%(np.mean(out), np.mean(self.init_performance), np.std(self.init_performance), reward))
            return max(reward, -3), np.mean(out)
        
    def _get_dataset_from_node(self, node):
        self.logger.info('- Start to do feature transformation') 
        #self.pipeline.set_transform(state.state)
        X_train, X_test, y_train, y_test = self.pipeline.run(node.get_state().state)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        self.logger.info('+ end with feature transformation')
        return X_train, X_test, y_train, y_test
    
    def _get_reward_for_dataset(self, X_train, X_test, y_train, y_test):
        """
        #########
        output
        #########
        reward, type of float, performance of the model
        fe, type of tuple, contains: name of the most important feature and data
        """
        reward = 0.0
        if X_train.isna().sum().sum() > 0:
            print('NA value exist')
            print(X_train)
            X_train.to_csv('../data/pro.csv')
        try:
            self.logger.info('Get final reward')           
            # feature selection miss
            #print('finish fillna and feature transformation', str(time.time()))
            if self.art == 'C':
                self.logger.info('- start to run classification model') #
                #model = SVC()
                model = RandomForestClassifier(random_state = self.rf_seed)
                model.fit(X_train, y_train)
                predict = model.predict(X_test)
                self.logger.info('+ end with model running and return evaluate result') #
                # evaluate
                #reward = roc_auc_score(self.y_test, predict) # two class
                #reward = accuracy_score(y_test, predict) # or f1_score
                #if len(set(predict)) >2 or len(y_test) > 2:
                reward = f1_score(y_test, predict, average='weighted')
                #else:
                #    reward = f1_score(y_test, predict)
            else:
                self.logger.info('- start to run regression model') #
                model = RandomForestRegressor(random_state = self.rf_seed)
                model.fit(X_train, y_train)
                predict = model.predict(X_test)
                self.logger.info('- end with model running and return evaluate result') #
                # evaluate
                #reward = 1- mean_absolute_error(y_test, predict)
                reward = 1 - self._relative_absolute_error(predict, y_test)
        except Exception as e:
            ex_type, ex_value, ex_traceback = sys.exc_info()
            # Extract unformatter stack traces as tuples
            trace_back = traceback.extract_tb(ex_traceback)
            # Format stacktrace
            stack_trace = list()
            for trace in trace_back:
                stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
            print('Error appear:')
            print(state.state)
            self.logger.error(e, exc_info=True)
            print("Stack trace : %s" %stack_trace)
        # get the most important feature, 
        ind_fe = model.feature_importances_
        name_col = X_train.columns[np.argmax(ind_fe)]
        return reward, (name_col, X_train[name_col])
    
    def _relative_absolute_error(self, pred, y):
        dis = abs((pred-y)).sum()
        dis2 = abs((y.mean() - y)).sum()
        #print(dis, dis2)
        if dis2 == 0 :
            return 1
        return dis/dis2
    
    def _get_init_performance(self):
        self.logger.info('Get init performance of the given dataset with cv = 5')
        rs = []
        #dat = copy.deepcopy(self.dat)
        xs = pd.concat([self.X_train, self.X_test])
        ys = np.concatenate([self.y_train, self.y_test])
        #while dat.shape[1]>2000:
        #    dat = self._feature_selection(dat)
        for i in range(10):
            #X_train, X_test, y_train, y_test = train_test_split(dat.iloc[:, :-1], dat.iloc[:, -1])
            _, X_test, _, y_test = train_test_split(self.X_test, self.y_test, test_size = 0.8, random_state = i)
            rs.append(self._get_reward_for_dataset(self.X_train, X_test, self.y_train, y_test)[0])
        self.logger.warn('The Init mean performance of the given dataset is:\t %s, std is:\t %s'%(str(np.mean(rs)), str(np.std(rs))))
        return rs
    
    def _load_available_actions(self):
        """
        use vector
        """
        # load lookout table for index
        # A: 数学操作: 单元
        typeA = {'abs': 0, 'cos': 1, 'exp':2, 'ln': 3, 'sigmoid': 5, 'square': 6, 'tanh': 7 , 'relu': 8,
                'reciprocal': 60, 'negative': 62, 'adde': 61, 'degree': 62, 'radian': 63}#, 'clustering': 65}# 'leakyInfo':64, 'clustering': 65}
        # B: 数学操作：双元（因为时耗, 且数量指数增长，所以单独拉出来）
        typeB = {'div': 9, 'minus': 10, 'prod': 11, 'add': 12}
        # C: 时间序列处理
        typeC = {'timeagg': 13, 'diff': 23}
        # D: 核方法
        typeD = {'kernelapproxrbf': 30}
        # E: 全局相关
        typeE = {'zscore': 18, }#'clustering_count':19, 'minmaxnorm': 16, }
        # F: Y相关
        #typeC = {'percentile25': 13, 'percentile75': 14, 'percentile50': 15, 'maximum': 16, 'std': 17, 'minimum': 18, 'diff': 23,} # add special
        #typeD = {'autoencoder': 20, 'clustering': 21, 'binning': 22}
        #typeE = { 'timeBinning': 24, 'tempWinAgg': 25, 'spatialAgg': 26, 'spatioAgg': 27, 'ktermFreq': 28, 'quanTransform': 29, 'nominalExpansion': 31, 'isomap': 32}
        # 需要y辅助的属性
        typeF_C = {'decisionTreeClassifierTransform': 31, 'mlpClassifierTransform': 32, 'svcTransform': 34,  'randomForestClassifierTransform':36, 'xgbClassifierTransform': 37} #'nearestNeighborsClassifierTransform': 33, 'gauRBFClassifierTransform': 36, 'gauExpClassifierTransform': 35,(it take too long) 'gauDotClassifierTransform': 34 (error, negative value exist),
        typeF_R = {'decisionTreeRegressorTransform':32, 'mlpRegressorTransform': 33, 'linearRegressorTransform': 33, 'svrTransform': 34, 'xgbRegressorTransform': 39}# 'gauDotWhiteRegressorTransform': 35, 'gauExpRegressorTransform': 36, 'gauRBFRegressorTransform': 37, 'RandomForestRegressorTransform': 38,}#'nearestNeighborsRegressorTransform': 39,
        # for test
        typeT = {'clustering': 34, 'leakyInfo': 35}
        if self.type == 'normal':
            #self.lookup.update(typeT)
            self.lookup.update(typeA)
            self.lookup.update(typeB)
            self.lookup.update(typeC)
            self.lookup.update(typeD)
            self.lookup.update(typeE)
            #self.lookup.update(typeF_C)
            self.lookup.update(typeF_R)
        elif self.type == 'picture':
            self.lookup.update(typeG)
        elif self.type == 'timeserie':
            self.lookup.update(typeC)
        else:
            self.logger.info('Unrecognized data type: %s, set it to normal'%self.datatype)
            self.lookup.update(typeA)
            self.lookup.update(typeB)
            self.lookup.update(typeD)
            self.lookup.update(typeE)
        self.logger.info('Data type %s, following transformation are loaded: %s'%(self.type, str(self.lookup.keys())))
        # create onehot dict
        number_of_actions = len(self.lookup.keys())
        for i, j in enumerate(self.lookup.keys()):
            tmp = np.zeros(number_of_actions)
            tmp[i] = 1
            self.lookup2[j] = tmp
            self.lookup2r[str(tmp)] = j
        return list(self.lookup.keys())
    
    def _load_data(self, dataset_name, sep = None, header = True, index_col = None, subsampling=True, balanced_sampling= True):
        self.logger.info('Load data from file: %s'%(dataset_name)) #
        if sep:
            if header:
                if type(index_col) != type(None):
                    dat = pd.read_csv(dataset_name, sep = sep, index_col = index_col)
                else:
                    dat = pd.read_csv(dataset_name, sep = sep)
            else:
                if type(index_col) != type(None):
                    dat = pd.read_csv(dataset_name, sep = sep, header = None, index_col = index_col)
                else:
                    dat = pd.read_csv(dataset_name, sep = sep, header = None)
        else:
            if header:
                if type(index_col) != type(None):
                    dat = pd.read_csv(dataset_name, index_col = index_col)
                else:
                    dat = pd.read_csv(dataset_name)
            else:
                if type(index_col) != type(None):
                    dat = pd.read_csv(dataset_name, header = None, index_col = index_col)
                else:
                    dat = pd.read_csv(dataset_name, header = None)
        dat = dat.reset_index(drop = True)
        self.logger.warn('Original shape of the dataset is %s'%(str(dat.shape)))
        self.logger.info('- check data menge')
        dat.columns = [str(i) for i in dat.columns]
        self.logger.info('- check data balance')
        if self.art == 'C': # up sampling, if art of task is classification
            if balanced_sampling:
                dat = self._balanced_sampling(dat)
            if subsampling:
                dat = self._subsampling(dat) # sub sampling, if number of data point >= 10000
        else:
            if subsampling:
                dat = self._subsampling(dat) # sub sampling, if number of data point >= 10000
        self.logger.info('- start to do data preprocessing') #
        dat, self.numeric_cols = self._data_preprocessing(dat, art = self.art, logger = self.logger)
        dat.columns = [str(i) if ('[' not in str(i)) and ('[' not in str(i)) and ('>' not in str(i)) and ('=' not in str(i)) and ('<' not in str(i)) else 'new'+str(j) for i, j in zip(dat.columns, range(len(dat.columns)))]
        self.logger.info('+ end with data preprocessing') #
        self.logger.warn('Shape of the dataset after label encoding is %s'%(str(dat.shape)))
        print(dat.head())
        return dat
    
    def _name_to_onehot(self, name):
        return self.lookup2[name]
    
    def _onehot_to_name(self, oh):
        return self.lookup2r[oh]
    
    def _subsampling(self, dat):
        # when number of instance to large only use 10000 data to do the feature engineering
        if dat.shape[0] > 10000:
            return dat.sample(n = 10000, random_state=1).reset_index(drop= True)
        else:
            return dat
        
    def is_done(self, state):
        if len(state.state) == self.lens:
            # done
            return True
        else:
            # not jet
            return False
  