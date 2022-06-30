import numpy as np
import pandas as pd
import random
import copy

from state import State
###############################################################
##########            MC_node                        ##########
###############################################################

class MC_node():
    def __init__(self, state, N = 1):
        self.state = state
        self.id = self.state.get_id()
        self.N = N
        self.edges = []
        self.sorted = False # sinify whether self.edges is sorted or not, only roll out in MCT can set it to false
        self.acc = 0
        
    def add_edge(self, e):
        self.edges.append(e)
        
    def add_edges(self, es):
        self.edges.extend(es)
    
    def cut_edges_in_number(self, n):
        self.sort_edges()
        num_edges = len(self.sort_edges())
        self.edges = self.edges[:num_edges - n]
        return self.edges
    
    def cut_edges_in_percent(self, p = 0.2):
        assert(p<=1)
        self.sort_edges()
        num_edges = len(self.sort_edges())
        if num_edges > 1:
            hit_point = int((1 - p) * num_edges)
            self.edges = self.edges[:hit_point]
        return self.edges
        
    def is_leaf(self):
        if len(self.edges) == 0:
            return True
        else:
            return False
    def get_actions_of_edges(self):
        return [i.action for i in self.edges]
    
    def get_first_child(self):
        return self.edges[0].get_out_node()
    
    def get_children(self):
        return [i.get_out_node() for i in self.edges]
    
    def get_edge_with_action(self, action):
        for i in self.edges:
            if i.action == action:
                return i
        return 0
    
    def get_id(self):
        return self.id
    
    def get_id_of_edges(self):
        return [i.id for i in self.edges]
    
    def get_infor_of_edges(self):
        cols = self.get_actions_of_edges()
        cols.append('parameter')
        dat = pd.DataFrame(columns = cols)
        print(cols)
        print(f'W: {self.get_W_of_edges()}')
        print(f'N: {self.get_N_of_edges()}')
        print(f'L: {self.get_L_of_edges()}')
        print(f'W_rate: {self.get_winrate_of_edges()}')
        print(f'Value: {self.get_value_of_edges()}')
        dat.loc[0, :-1] = self.get_N_of_edges()
        dat.loc[0, 'parameter'] = 'N'
        dat.loc[1, :-1] = self.get_W_of_edges()
        dat.loc[1, 'parameter'] = 'W'
        dat.loc[2, :-1] = self.get_L_of_edges()
        dat.loc[2, 'parameter'] = 'L'
        dat.loc[3, :-1] = self.get_winrate_of_edges()
        dat.loc[3, 'parameter'] = 'Win rate'
        dat.loc[4, :-1] = self.get_value_of_edges()
        dat.loc[4, 'parameter'] = 'Value'
        return dat
            
    def get_num_edges(self):
        return len(self.deges)
    
    def get_N_of_edges(self):
        return [i.N for i in self.edges]
    
    def get_L_of_edges(self):
        return [i.L for i in self.edges]
    
    def get_value_of_edges(self):
        return [i.value for i in self.edges]
    
    def get_winrate_of_edges(self):
        return [i.get_winrate() for i in self.edges]
    
    def get_W_of_edges(self):
        return [i.W for i in self.edges]
    
    def get_N(self):
        return self.N
    
    def get_state(self):
        return self.state
    
    def reset_sorted(self):
        ### Aborted
        self.sorted = False
    
    def sort_edges_by_value(self):
        self.edges = sorted(self.edges, key = lambda x: x.get_value(), reverse = True)
        self.sorted = True
        return self.edges
    
    def sort_edges_by_winrate(self):
        self.edges = sorted(self.edges, key = lambda x: x.get_winrate(), reverse = True)
        self.sorted = True
        return self.edges
    
    def sort_edges_by_ucb(self):
        self.edges = sorted(self.edges, key = lambda x: x.get_ucb(), reverse = True)
    
    def __eq__(self, n2):
        return self.state == n2.state
    
#    def __str__(self):
#        return self.id


###############################################################
##########            MC_edge                        ##########
###############################################################



class MC_edge():
    def __init__(self, action, in_node, out_node, priori = (1, 1)):
        self.action = action
        self.in_node = in_node
        self.out_node = out_node
        self.id = in_node.get_id() + '-' + out_node.get_id()
        self.N = 0
        self.W = priori[0]
        self.L = priori[1]
        self.value = 0
    def get_in_node(self):
        return self.in_node
    def get_out_node(self):
        return self.out_node
    def get_action(self):
        return self.action
    def get_state(self):
        return (self.Q, self.U, self.W, self.N, self.P)
    def get_value(self):
        self.value = np.random.beta(self.W, self.L)
        return self.value
    def get_winrate(self):
        return self.W/(self.W + self.L)
    def get_ucb(self):
        return (self.W)/(self.W+self.L) + 1.4*np.sqrt(np.log(self.in_node.N)/self.N)
#    def __str__(self):
#        return self.id


###############################################################
##########            MCFE_tree                      ##########
###############################################################


class MCFE_tree():
    """

    """
    def __init__(self, root_state, max_depth = 3, logger = None):
        """
        input:
            root_state, type of State
        """
        # set logger
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
        # init attribute
        self.max_depth = max_depth
        self.root = MC_node(root_state)
        #self.roottree = {'root': self.root, 'tree': {}}
        self.tree = {}
        self.add_node_to_tree(self.root)
    
    def add_node_to_tree(self, node):
        self.logger.info('Add node %s to root tree'%node.get_id()) # logger
        if node.get_id() not in self.tree.keys():
            self.tree[node.get_id()] = node
            return 1
        else:
            self.logger.error("Fail to add node %s to root tree, ndoe existed"%node.get_id())
            return 0
            
    def back_fill(self, value, path):
        # update all the node and edge in the path
        #   if win W add 1 else L add 1
        #   in both case N add 1
        self.logger.info("Change value for each edge in the paths")
        for edge in path:
            # update input node value
            edge.in_node.N += 1
            edge.in_node.reset_sorted()
            # update edge value
            edge.N += 1
            if value <= 0:
                edge.L += abs(value)
            else:
                edge.W += value
    
    def expansion(self, leaf, edges, values):
        """
        Add edges only
        """
        self.logger.info("Expansion: add new edges and nodes to the tree")
        out = edges[0]
        best_score= values[0]
        for edge, value in zip(edges, values):
            # add edge to node
            flag = 0
            flag = self.add_node_to_tree(edge.get_out_node())
            if flag == 1:
                leaf.add_edge(edge)
            # get winner
            #print(best_score, value)
            #print(out.id)
            if best_score < value:
                best_score = value
                out = edge
        return out
        
    def expansion_with_ts(self, leaf, edges): # this method is ts based !!!!
        """
        Add edges to the current node and add the output nodes of the edge to the tree
        and return the expanded node
        """
        self.logger.info("Expansion: add new edges and nodes to the tree")
        for edge in edges:
            flag = 0
            flag = self.add_node_to_tree(edge.get_out_node())
            if flag == 1:
                leaf.add_edge(edge)
        leaf.sort_edges_by_value()
        return leaf.edges[0]

    def get_path_from_root(self, node):
        state = node.State.state
        edges = []
        current_node = self.root
        for i in state:
            edge = current_node.get_edge_with_action(i)
            edges.append(edge)
            current_node = edge.get_out_node()
        return edges
    
    def roll_out_random(self, node, actions):
        """
        ########### TODO maybe use tampsom sampling instead
        根据node进行展开，设定最大长度，如果大于最大长度就roll out一位，否则则roll out到最大长度
        """
        current_state = node.state
        depth = len(node.state)
        diff = self.max_depth -depth
        number = min(2, diff)
        for i in np.arange(number):
            action = random.choice(actions)
            current_state = self.simulate_action(current_state, action)
        return MC_node(current_state)
    
    def roll_out_with_scorer(self, node, available_actions, scorer, game):
        current_state = node.state
        depth = len(node.state)
        for i in np.arange(depth, self.max_depth):
            states = [game.simulate_action(current_state, action) for action in available_actions]
            values = scorer.predict(game.states_to_onehot(states))
            ind = np.argmax(values)
            current_state = game.simulate_action(current_state, available_actions[ind])
        return MC_node(current_state)
    
    def roll_out_cv(self, node, actions, scorer = None, game = None, num_cv = 10):
        """
        expand cv to roll out process, the point is, how to avoid the samme result and get good advice at the same time
        #########
        input
        #########
        node, type of MC_node
        actions, type of list of action
        num_cv, type of int
        #########
        output
        #########
        out, type of MC_node, length = num_cv
        """
        out = []
        for i in range(num_cv):
            if scorer:
                self.logger.info('scorer exist, use scorer in the roll out')
                out.append(self.roll_out_with_scorer(node, actions, scorer, game))
            else:
                self.logger.info("scorer doesn't exist, use random selection in the roll out")
                out.append(self.roll_out_random(node, actions))
        return out
    
    def selection(self, root = None):
        """
        selection with thompsom sampling
        """
        self.logger.info('Path selection with thompsom sampling')
        path = []
        current_node = root
        if root.is_leaf():
            self.logger.info('root is a leaf, return the root node')
            return root, path # the paths here are leer
        else:
            #only take one path
            self.logger.info('root is not a leaf, selection according to root')
            while not current_node.is_leaf():
                ################ TODO: change with tompson sampling method
                current_node.reset_sorted() # 因为用的是TS所以每次都要重新排序（会随机get value）
                current_node.sort_edges_by_value()
                edge = current_node.edges[0]
                path.append(edge)
                current_node = edge.get_out_node()
        return current_node, path
        
    def selection_ucb(self, root = None):
        """
        selection with UCB
        """
        self.logger.debug('Path selection with UCB')
        path = []
        current_node = root
        if root.is_leaf():
            self.logger.debug('root is a leaf, return the root node')
            return root, path
        else:
            self.logger.debug('root is not a leaf, selection according to UCB')
            while not current_node.is_leaf():
                current_node.reset_sorted()
                current_node.sort_edges_by_ucb()
                edge = current_node.edges[0]
                path.append(edge)
                current_node = edge.get_out_node()
        return current_node, path
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
    
    def set_root(self, node):
        self.root = node
        path = []