import copy
import numpy as np
import pandas as pd
import pickle
import random
import time

from sklearn.model_selection import train_test_split

from transforms import *

class Pipeline():
    def __init__(self, data, art = 'C', logger = None, seed = 0,):
        """
        self attribute is not safe here because of Parallel
        ==========
        inputs:
        ==========
        data, type of dataframe, target dataset
        numeric_cols, type of list of string
            numeric columns in the dataset
        actions, type of list of string
            transformation sequence
        logger, type of Logger
        seed, type of int
            random seed used in the train test split
        """
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
        self.dat = data # save the original data that used in the convertion later
        self.seed = seed
        self.rf_seed = 0
        self.art = art # ??
        self.max_db = 500
        self.database = {} # database used to save history converted data
                           # keys is ','.join(actions) value is type of dict
    
    def insert_action(self, action):
        self.transforms.append(self._convert_action_to_transform(action))
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
            
    def clean_data(self, train, test = None):
        self.logger.info('Clean Data, number of inf and nun are for train set: (%d, %d)'%((train==np.inf).sum().sum(), train.isna().sum().sum()))
        if type(test) != type(None):
            self.logger.info('Clean Data, number of inf and nun are: (%d, %d) for test set'%((test==np.inf).sum().sum(), test.isna().sum().sum()))
        # set type to float32
        train = train.astype(np.float32)
        if type(test) != type(None):
            test = test.astype(np.float32)
        # deal with inf.
        train = train.replace([np.inf, -np.inf], np.nan)
        if type(test) != type(None):
            test = test.replace([np.inf, -np.inf], np.nan)
        # remove columns half of na 
        train = train.dropna(axis = 1, thresh=len(train)*.5)
        # remove costant columns
        train = train.loc[:, (train != train.iloc[0]).any()] 
        if type(test) != type(None):
            test = test.loc[:, train.columns]
        #train = train.loc[:, (train.isna().sum()/len(train) < .3)&(test.isna().sum()/len(test) < .3)]
        # fillna
        if train.isna().sum().sum() > 0:
            self.logger.info('- start to fill na for the new feature')
            columns = train.columns
            #imp = IterativeImputer(max_iter=10, random_state=0)
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            #train = train.fillna(train.mean())
            index_train = train.index
            tmp = imp.fit_transform(train)
            if tmp.shape[1] != train.shape[1]:
                tmp = train.fillna(0)
            train = pd.DataFrame(tmp, columns = columns, index = index_train)
        if type(test) != type(None):
            if test.isna().sum().sum() > 0:
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                columns = test.columns
                index_test = test.index
                tmp = imp.fit_transform(test)
                if tmp.shape[1] != test.shape[1]:
                    tmp = test.fillna(0)
                test = pd.DataFrame(tmp, columns = columns, index = index_test)
                #test = test.fillna(test.mean())
        self.logger.info('End with Data cleaning, number of inf and nun are for train set: (%d, %d)'%((train==np.inf).sum().sum(), train.isna().sum().sum()))
        if type(test) != type(None):
            self.logger.info('End with Data cleaning, number of inf and nun are: (%d, %d) for test set'%((test==np.inf).sum().sum(), test.isna().sum().sum()))
        if type(test) != type(None):
            return train, test
        else:
            return train
    
    def reset_seed(self, seed):
        # this process will clear the buffer
        self.seed = seed
        self.database = {}
        
    def run(self, actions):
        self.logger.warn('Run feature transformation: %s'%(str(actions)))
        transforms, X_train, X_test, y_train, y_test = self._load_actions(actions)
        # run feature transform
        train = X_train
        test = X_test
        num_transformations_load = len(actions) - len(transforms)
        for ind, i in enumerate(transforms):
            # feature selection when number of columns exceed the thredhold
            if len(train.columns) > 1000:
                train, test = self._feature_selection(train, test, y_train)
            # operate transformation one by one
            self.logger.info('Executing transformation: %s, shape of current data is: [%s, %s]'%(str(i.name), str(train.shape[0]), str(train.shape[1])))
            time_begin = time.time()
            mnc = 50
            if i.type == 2 and len(train.columns) > mnc:
                while True:
                    self.logger.info('Target transformation belong to type 2')
                    train, test = self._feature_selection(train, test, y_train)
                    if train.shape[1] <= mnc:
                        break
                train, test = i.fit(train, test)
            elif i.type == 5:
                if len(train.columns) > 100:
                    train, test = self._feature_selection(train, test, y_train)
                train, test = i.fit(train, test, y_train, y_test)
            else:
                train, test = i.fit(train, test)
            # drop columns with duplicate name
            train = train.loc[:,~train.columns.duplicated()]
            test = test.loc[:,~test.columns.duplicated()]
            # clean data
            train, test = self.clean_data(train, test)
            time_end = time.time()
            self.logger.info('Time expend for transformation %s is %s sec'%(str(i.name), str(round(time_end - time_begin))))
            # save data for every dataset, because the get_reward method will repeat the same sequence multi time
            self._save_data_to_database(actions[:num_transformations_load + ind + 1] , train, test, y_train, y_test)
        return train, test, y_train, y_test
    
    def _feature_selection(self, train, test, y_train):
        self.logger.info('The number of columns exceed max number, try columns selection first')
        # feature selection
        if self.art == 'C':
            clf = RandomForestClassifier(random_state = self.rf_seed).fit(train, y_train)
        else:
            clf = RandomForestRegressor(random_state = self.rf_seed).fit(train, y_train)
        fs = SelectFromModel(clf, threshold= 'mean', prefit=True)
        supp = fs.get_support()
        train = pd.DataFrame(fs.transform(train), columns = train.columns[supp], index = train.index)
        test = pd.DataFrame(fs.transform(test), columns = test.columns[supp], index = test.index)
        self.logger.info('End with columns selection, number of columns now is: %s'%(str(train.shape[1])))
        return train, test
    
    def _load_actions(self, actions):
        """
        check if transformation existes, if so, load the data set
        convert actions name to real transformation functions
        ==========
        outputs:
        ==========
            tfs, list of transform left
            dat, base dataset
        """
        self.logger.info('Load actions: %s'%(actions))
        # check if transformation is already exists
        actions_left = actions
        dat = self.dat
        flag = False
        for i in np.arange(len(actions), -1, -1):
            ## check database existence
            tmp = actions[:i]
            tmp = ', '.join(tmp) # should be same as saving the dataset
            if tmp in self.database.keys():
                self.logger.info('- part of sequence existed, direct use the dataset in the database')
                self.logger.info('- transformation of existed data set: %s'%(tmp))
                self.logger.info('- transformation still lack of: %s'%(actions[i:]))
                actions_left = actions[i:] # reset actions list
                self.database[tmp]['hit_count'] += 1 # add hit count
                X_train = copy.deepcopy(self.database[tmp]['X_train'])
                X_test  = copy.deepcopy(self.database[tmp]['X_test'])
                y_train = copy.deepcopy(self.database[tmp]['y_train'])
                y_test  = copy.deepcopy(self.database[tmp]['y_test'])
                flag = True
                break
        tfs = [self._convert_action_to_transform(i) for i in actions_left]
        if not flag:
            X_train, X_test, y_train, y_test = train_test_split(dat.iloc[:, :-1], dat.iloc[:, -1], test_size = 0.33, random_state = self.seed)
        return tfs, X_train, X_test, y_train, y_test
    
    def _set_transform(self, actions):
        self.logger.info('Set transforms')
        return self._load_actions(actions)
    
    def _save_data_to_database(self, actions, train, test, y_train, y_test):
        # check condition, maximal number or left memory
        tmpkey = ', '.join(actions)
        if tmpkey in self.database.keys():
            self.logger.info('- data set already exist')
            #print('actions already exist %s'%str(actions))
            return 0
        self.logger.info('Save data set to database: %s'%(actions))
        dict_data = {'hit_count': 1, 'X_train': copy.deepcopy(train), 'X_test': copy.deepcopy(test), 'y_train': copy.deepcopy(y_train), 'y_test': copy.deepcopy(y_test)}
        if len(self.database) <= self.max_db:
            self.database[tmpkey] = dict_data
        else:
            self.logger.info('Database is full, remove the one with smallest count')
            tg = sorted(self.database.items(), key = lambda x: x[1]['hit_count'])[0][0]
            del self.database[tg]
            self.database[tmpkey] = dict_data
        return 1
    
    def _convert_action_to_transform(self, action):
        if action == 'abs':
            self.logger.info('- load method abs')
            tf = Abs()
        elif action == 'add':
            self.logger.info('- load method add')
            tf = Add()
        elif action == 'adde':
            self.logger.info('- load method add e')
            tf = Adde()
        elif action == 'autoencoder':
            self.logger.info('- load method autoencoder')
            tf = Autoencoder()
        elif action == 'binning':
            self.logger.info('- load method binning')
            tf = Binning()
        elif action == 'clustering':
            self.logger.info('- load method clustering')
            tf = Clustering()
        elif action == 'cos':
            self.logger.info('- load method cos')
            tf = Cos()
        elif action == 'decisionTreeClassifierTransform':
            self.logger.info('- load method DecisionTreeClassifierTransform')
            tf = DecisionTreeClassifierTransform()
        elif action == 'decisionTreeRegressorTransform':
            self.logger.info('- load method DecisionTreeRegressorTransform')
            tf = DecisionTreeRegressorTransform()
        elif action == 'degree':
            self.logger.info('- load method degree')
            tf = Degree()
        elif action == 'diff':
            self.logger.info('- load method diff')
            tf = Diff()
        elif action == 'div':
            self.logger.info('- load method div')
            tf = Div()
        elif action == 'exp':
            self.logger.info('- load method exp')
            tf = Exp()
        elif action == 'ln':
            self.logger.info('- load method ln')
            tf = Ln()
        elif action == 'minmaxnorm':
            self.logger.info('- load method minmaxnorm')
            tf = Minmaxnorm()
        elif action == 'gauDotClassifierTransform':
            self.logger.info('- load method GauDotClassifierTransform')
            tf = GauDotClassifierTransform()
        elif action == 'gauDotWhiteRegressorTransform':
            self.logger.info('- load method GauDotWhiteTransform')
            tf = GauDotWhiteRegressorTransform()
        elif action == 'gauExpClassifierTransform':
            self.logger.info('- laod method GauExpClassifierTransform')
            tf = GauExpClassifierTransform()
        elif action == 'gauExpRegressorTransform':
            self.logger.info('- load method gauexptransform')
            tf = GauExpRegressorTransform()
        elif action == 'gauRBFClassifierTransform':
            self.logger.info('- load method GauRBFClassifierTransform')
            tf = GauRBFClassifierTransform()
        elif action == 'gauRBFRegressorTransform':
            self.logger.info('- load method gaurbftransform')
            tf = GauRBFRegressorTransform()
        elif action == 'prod':
            self.logger.info('- load method prod')
            tf = Product()
        elif action == 'isomap':
            self.logger.info('- load method isomap')
            tf = IsoMap()
        elif action == 'ktermFreq':
            self.logger.info('- load method k term frequence')
            tf = KTermFreq()
        elif action == 'kernelapproxrbf':
            self.logger.info('- load method binning D')
            tf = KernelApproxRBF()
        elif action == 'leakyInfo':
            self.logger.info('- load method LeakyInfo')
            tf = LeakyInfo()
        elif action == 'linearRegressorTransform':
            self.logger.info('- load method linear regressor transform')
            tf = LinearRegressorTransform()
        elif action == 'maximum':
            self.logger.info('- load method maximum')
            tf = Maximum()
        elif action == 'minimum':
            self.logger.info('- load method minimum')
            tf = Minimum()
        elif action == 'minus':
            self.logger.info('- load method minus')
            tf = Minus()
        elif action == 'mlpClassifierTransform':
            self.logger.info('- load method mlp classification transform')
            tf = MLPClassifierTransform()
        elif action == 'mlpRegressorTransform':
            self.logger.info('- load method mlp regressor transform')
            tf = MLPRegressorTransform()
        elif action == 'negative':
            self.logger.info('- load method negative')
            tf = Negative()
        elif action == 'nearestNeighborsClassifierTransform':
            self.logger.info('- load method NearestNeighborsClassifierTransform')
            tf = NearestNeighborsClassifierTransform()
        elif action == 'nearestNeighborsRegressorTransform':
            self.logger.info('- load method NearestNeighborsRegressorTransform')
            tf = NearestNeighborsRegressorTransform()
        elif action == 'nominalExpansion':
            self.logger.info('- load method norminalExpansion')
            tf = NominalExpansion()
        elif action == 'percentile25':
            self.logger.info('- load method percentile25')
            tf = Percentile25()
        elif action == 'percentile50':
            self.logger.info('- load method percentile50')
            tf = Percentile50()
        elif action == 'percentile75':
            self.logger.info('- load method percentile75')
            tf = Percentile75()
        elif action == 'quanTransform':
            self.logger.info('- load method binning U')
            tf = QuanTransform()
        elif action == 'radian':
            self.logger.info('- load method radian')
            tf = Radian()
        elif action == 'randomForestClassifierTransform':
            self.logger.info('- load method RandomForestClassifierTransform')
            tf = RandomForestClassifierTransform()
        elif action == 'randomForestRegressorTransform':
            self.logger.info('- load method RandomForestRegressorTransform')
            tf = RandomForestRegressorTransform()
        elif action == 'reciprocal':
            self.logger.info('- load method reciprocal')
            tf = Reciprocal()
        elif action == 'svrTransform':
            self.logger.info('- load method svr transform')
            tf = SVRTransform()
        elif action == 'svcTransform':
            self.logger.info('- load method svc transform')
            tf = SVCTransform()
        elif action == 'relu':
            self.logger.info('- load method relu')
            tf = Relu()
        elif action == 'sin':
            self.logger.info('- load method sin')
            tf = Sin()
        elif action == 'sigmoid':
            self.logger.info('- load method sigmoid')
            tf = Sigmoid()
        elif action == 'spatialAgg':
            self.logger.info('- load method spatial aggregation')
            tf = SpatialAgg()
        elif action == 'spatioAgg':
            self.logger.info('- load method spatio aggregation')
            tf = SpatioAgg()
        elif action == 'square':
            self.logger.info('- load method square')
            tf = Square()
        elif action == 'std':
            self.logger.info('- load method std')
            tf = Std()
        elif action == 'tanh':
            self.logger.info('- load method tanh')
            tf = Tanh()
        elif action == 'timeagg':
            self.logger.info('- load method time agg')
            tf = Timeagg()
        elif action == 'timeBinning':
            self.logger.info('- load method time binning')
            tf = TimeBinning()
        elif action == 'tempWinAgg':
            self.logger.info('- load method time window aggregation')
            tf = TempWinAgg()
        elif action == 'xgbClassifierTransform':
            self.logger.info('- load method XGBClassifierTransform')
            tf = XGBClassifierTransform()
        elif action == 'xgbRegressorTransform':
            self.logger.info('- load method XGBRegressorTransform')
            tf = XGBRegressorTransform()
        elif action == 'zscore':
            self.logger.info('- load method zscore')
            tf = Zscore()
        else:
            #self.logger.error('Target transform %s not found in the method set, jump over the method'%action)
            #print('target transform method not found!!')
            tf = None
        return tf