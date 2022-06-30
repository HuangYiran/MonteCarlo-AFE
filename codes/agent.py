from mct import MC_node, MC_edge, MCFE_tree
from state import State

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import copy
import numpy as np
import pandas as pd
import random
from scipy.stats import poisson

from state import State

#应该区分root和current state
class Agent():
    def __init__(self, game, scorer, max_depth = 10, logger = None):
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
        self.logger.info('##### Init Agent')
        self.game = game
        self.init_performance = self.game.get_init_performance()
        self.scorer = scorer # for roll out, update when new data come
        self.scorer2 = copy.deepcopy(self.scorer) # for expand, update only when loss decrease
        self.root_state = game.get_current_state()
        self.max_depth = max_depth
        self.mct = MCFE_tree(self.root_state, max_depth= self.max_depth, logger = self.logger)
        self.root = self.mct.root
        self.num_episode = 0
    
    def run(self, budget = 100, random_exp = False, path = '../results/ibm_rl/classification/'):
        df = pd.DataFrame(columns = ['filename', 'performance', 'node', 'top_five'])
        bs = [int(poisson.pmf(i, 1)*budget) for i in range(5)] # set the number of episode in each step
        init_perf = self.game.init_performance
        counter = 0
        cn = None
        records = []
        for i in bs:
            for j in range(i):
                self.logger.warn('X'*60)
                self.logger.warn('Round:\t%d'%(counter))
                self.logger.warn('X'*60)
                node, acc = self.episode(random_exp = random_exp)
                counter+=1
                node = self._update_init_performance(node, acc) # if reward better, 10 fold node and reset init performance.
                if node:
                    cn = node
                df.loc[counter, 'filename'] = self.game.dataname.split('/')[-1]
                df.loc[counter, 'performance'] = str(self.game.init_performance)
                df.loc[counter, 'node'] = cn
                df.loc[counter, 'top_five'] = str(self.game.record_top_five)
                #df.to_csv(path + self.game.dataname.split('/')[-1][:-4])
                records.append(self.root.get_infor_of_edges())
            #self.take_step()
            if self.max_depth == len(self.root.state):
                break
        #return self.root, df, records
        return self.root, self.game.init_performance, self.game.init_dat, self.game.record_top_five, self.game.record_best_features, self.game.buffer
    
    def get_performance_for_transformations(self, li):
        # e.g. li: ['relu', 'square', 'square', 'cos', 'relu']
        node = MC_node(State(li))
        out = self.game.get_reward_cv(node, cv = 10)
        return out[0]
    
    def episode(self, random_exp = False):
        #print(self.root.edges)
        self.num_episode += 1
        self.logger.info('#'*50)
        self.logger.info('Selection')
        self.logger.info('#'*50)
        #node, path = self.mct.selection(self.root)
        node, path = self.mct.selection_ucb(self.root)
        #print(self.root.id)
        #print(node.id)
        #print('======= expansion ==========')
        self.logger.info('#'*50)
        self.logger.info('Expansion')
        self.logger.info('#'*50)
        #print(self.root.edges)
        if len(node.state.state) < self.max_depth:
            # didn't reach the maximum level, expand
            self.logger.info("Didn't reach the maximum level jet, expand the node in root tree.")
            # evaluate node state
            if random_exp:
                edges, values = self._create_edges_for_leaf_and_evaluate_with_random(node)
            else:
                edges, values = self._create_edges_for_leaf_and_evaluate(node)
            # expansion with ts
            expanded_edge = self.mct.expansion(node, edges, values)
            path.append(expanded_edge)
            node = expanded_edge.get_out_node()
        else:
            # reach the maximum level
            self.logger.info('Reach the maximum level in root tree, return the current node')
        #print('========= roll out ==========')
        #print(self.root.edges)
        self.logger.info('#'*50)
        self.logger.info('Roll out')
        self.logger.info('#'*50)
        #node = self.mct.roll_out(node, self.game.actions)
        nodes = self.mct.roll_out_cv(node, self.game.actions, num_cv = 10) # random roll out 10 times
        #print('======= evaluate the result with hyperband =========')
        #print(self.root.edges)
        self.logger.info('#'*50)
        self.logger.info('Evaluate')
        self.logger.info('#'*50)
        #reward = self.game.get_reward(node)
        reward, acc = self.game.get_avg_reward_for_multi_nodes(nodes)
        #print('======= back fill =========')
        #print(self.root.edges)
        self.logger.info('#'*50)
        self.logger.info('Back fill')
        self.logger.info('#'*50)
        self.mct.back_fill(reward, path)
        #print('======= update network =========')
        #print(self.root.edges)
        self.logger.info('#'*50)
        self.logger.info('Update network')
        self.logger.info('#'*50)
        self._update_scorer()
        return node, acc
    
    def data_preparation_for_network_from_buffer(self, game):
        # select data for training from the buffer
        buffer = game.buffer
        avg_perf = buffer['y'].quantile(.7)
        pos_data = buffer[buffer['y'] > avg_perf]
        if pos_data.shape[0] < 4 or buffer[buffer['y'] <= avg_perf].shape[0] < 4:
            return 0, 0 
        neg_data = buffer[buffer['y'] <= avg_perf].sample(pos_data.shape[0], replace = True)
        data = pd.concat([pos_data, neg_data])
        # shuffle
        data = data.sample(frac = 1)
        # change type of X from pandas.Series to list of list of numpy
        xs = [State(i.replace(' ', '').split(',')) for i in data['X'].to_list()] # Series -> list of State
        xs = game.states_to_onehot(xs)
        # change type of y from pandas.Series to list of float
        ys = data['y'].to_list()
        return xs, ys
    
    def take_step(self):
        self.root = max(self.root.edges, key = lambda x: x.W).get_out_node()
        
    def test_pipeline_performance(self, pipeline):
        # pipeline: list of action
        cv= 5
        X_train, X_test, y_train, y_test = self.game._get_dataset_from_node(MC_node(State(pipeline)))
        if self.game.art == 'C':
            list_action = ['nearestNeighborsClassifierTransform', 'svcTransform', 'randomForestClassifierTransform', 'xgbClassifierTransform']#, 'gauRBFClassifierTransform']
            X_train2, X_test2, _, _ = self.game.collect_dataset_from_actions(list_action)
        else:
            list_action = ['decisionTreeRegressorTransform', 'mlpRegressorTransform', 'nearestNeighborsRegressorTransform', 'linearRegressorTransform', 'svrTransform', 'gauDotWhiteRegressorTransform', 'gauExpRegressorTransform', 'randomForestRegressorTransform', 'xgbRegressorTransform']#'gauRBFRegressorTransform', 
            X_train2, X_test2, _, _ = self.game.collect_dataset_from_actions(list_action)
        assert(X_train.index[0] == X_train2.index[0])
        assert(X_train.index[0] == self.game.X_train.index[0])
        X_train = pd.concat([X_train, X_train2, self.game.X_train], axis = 1)
        X_test = pd.concat([X_test, X_test2, self.game.X_test], axis = 1)
        #xs = pd.concat([X_train, X_test])
        #ys = np.concatenate([y_train, y_test])
        out = []
        for i in range(cv):
            _, X_test, _, y_test = train_test_split(X_test, y_test, test_size = 0.80, random_state = i)
            tmp, _ = self.game._get_reward_for_dataset(X_train, X_test, y_train, y_test)
            out.append(tmp)
        self.logger.info('Performance of the given node is: %s'%(str(out)))
        return (np.mean(out), out)
            
        
    def _evaluate_leaf_state(self, state_leaf):
        """ not test jet
        input:
          state_leaf, type of Node
        output:
          available_actions, type of list of action
          states, type of list of State
          values, type of list of float
        """
        self.logger.info('Evaluate leaf state')
        # get index of available_actions
        available_actions = self.game.get_available_actions(state_leaf)
        # get the next state through simulation
        states = [self.game.simulate_action(state_leaf.get_state(), action) for action in available_actions]
        values = self.scorer2.predict(self.game.states_to_onehot(states))
        return available_actions, states, values
    
    def _create_edges_for_leaf_and_evaluate(self, leaf):
        """
        return created edges
        """
        self.logger.info('Evaluate leaf state with thompsom sampling')
        state_leaf = leaf.state
        # get index of available_actions
        available_actions = self.game.get_available_actions(state_leaf)
        # get the next state through simulation
        states = [self.game.simulate_action(state_leaf, action) for action in available_actions]
        # create edge for each action ##### TODO CHANGE THE PRIORI LATER
        edges = [MC_edge(action, leaf, MC_node(state), priori = [1,1]) for action, state in zip(available_actions, states)]
        values = self.scorer.predict(self.game.states_to_onehot(states))
        return edges, values
    
    def _create_edges_for_leaf_and_evaluate_with_random(self, leaf):
        """
        return created edges
        """
        self.logger.info('Evaluate leaf state with thompsom sampling')
        state_leaf = leaf.state
        # get index of available_actions
        available_actions = self.game.get_available_actions(state_leaf)
        # get the next state through simulation
        states = [self.game.simulate_action(state_leaf, action) for action in available_actions]
        # create edge for each action ##### TODO CHANGE THE PRIORI LATER
        edges = [MC_edge(action, leaf, MC_node(state), priori = [1,1]) for action, state in zip(available_actions, states)]
        values = [random.uniform(0,1) for i in states]
        return edges, values
    
    def _compare_scorers(self):
        xs, ys = self._data_preparation_for_network()
        if xs == 0:
            return 0
        lo1 = self.scorer.fit(xs, ys)
        lo2 = self.scorer2.fit(xs, ys)
        if lo1 < lo2:
            self.logger.info('Roll out scorer shows improment, copy to expand scorer')
            self.scorer2 = copy.deepcopy(self.scorer)
    
    def _data_preparation_for_network(self):
        # select data for training from the buffer
        avg_perf = self.game.buffer['y'].quantile(.7)
        pos_data = self.game.buffer[self.game.buffer['y'] > avg_perf]
        if pos_data.shape[0] < 4 or self.game.buffer[self.game.buffer['y'] <= avg_perf].shape[0] < 4:
            return 0, 0 
        neg_data = self.game.buffer[self.game.buffer['y'] <= avg_perf].sample(pos_data.shape[0], replace = True)
        data = pd.concat([pos_data, neg_data])
        # shuffle
        data = data.sample(frac = 1)
        # change type of X from pandas.Series to list of list of numpy
        xs = [State(i.replace(' ', '').split(',')) for i in data['X'].to_list()] # Series -> list of State
        xs = self.game.states_to_onehot(xs)
        # change type of y from pandas.Series to list of float
        ys = data['y'].to_list()
        return xs, ys
    
    def _update_init_performance(self, node, reward):
        self.game.update_init_performance_with_node(node, reward)
        return 1
    
    def _update_scorer(self):
        xs, ys = self._data_preparation_for_network()
        if xs == 0:
            return 0
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size = 0.33)
        # feed the network
        self.scorer.fit(X_train, y_train)
        perf = self.scorer.evaluate(X_test, y_test)
        self.logger.warn(f'Performance of scorer is: {perf}')
        return 1
    

class Data:
    """
    data class for Xs, ys
    input:
        xs: list of list of int
        ys: list of float
    """
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = torch.FloatTensor(ys)
        assert(len(self.xs) == len(self.ys))
        self.len = len(self.xs)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.xs[index],
                self.ys[index])