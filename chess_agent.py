import random
import numpy as np
import copy, os
from abc import abstractmethod 

class ChessAgent():
    def __init__(self):
        self.states_map = {} #0：标准state 1{标准动作:实际动作}
        self.Q_table = {} 
        self.EPSILON = 0.5
        self.ELASTIC = 0
        self.model = None
        self.curr_Q_value = None

    def set_model(self, model):
        self.model = model

    def get_Q_dict(self, state):
        state, rotate_index = self.add_state(state)
        Q = self.Q_table[state]      
        return  {self.rotate_action(k, rotate_index):Q[k] for k in Q}
    
    def add_state(self, state): 
        state_index = self.state_2_index(state)
        if state_index in self.states_map:
            return self.states_map[state_index]
        else:
            actions = self.get_valid_actions(state) 
            self.Q_table[state_index] = {act:0 for act in actions}
            rotated_states = self.get_rotate_states(copy.deepcopy(state), state_index)
            self.states_map.update(rotated_states)
            return rotated_states[state_index]          
    
    #只接受标准state和标准action 
    def update_Q_table(self, state, action, next_state, reward):   
        if next_state is None:
            self.Q_table[state][action] = reward
        else:
            vals = self.Q_table[next_state].values()
            Qmax = max(vals)
            if abs(Qmax) == 1 :
                self.Q_table[state][action] = -Qmax
            else:            
                self.Q_table[state][action] =  -sum(vals)/len(vals)  
 
    #只接受标准state, 返回标准标准action的index
    def epsilon_greedy(self, state): # ε-greedy策略
        Q = self.Q_table[state]
        Qmax = max(Q.values()) 
        if abs(Qmax)== 1:
            return None
        maxActions, otherActions, denyActions = [], [], [] 
        for index in Q:
            Q_val = Q[index]
            if Q_val >= Qmax - self.ELASTIC:
                maxActions.append(index)
            else:
                if Q_val >= -1 + self.ELASTIC:
                    otherActions.append(index) 
                else:
                    denyActions.append(index)  
        if len(denyActions) == len(Q):            
            action = random.choice(denyActions)
        else:   
            epsilon = self.EPSILON if Qmax <= 1 - self.ELASTIC else 0        
            if len(otherActions) > 0 and random.random() < epsilon:
                 action = random.choice(otherActions)
            else:
                action = random.choice(maxActions) 
        return action
    def random_policy(self, state): # ε-greedy策略
        Q = self.Q_table[state] 
        options = []
        for index, q in Q.items():
            if abs(q) == 1:
                continue
            options.append(index)
        if len(options) == 0:
            return None
        else:
            return random.choice(options) 

    def greedy(self, state, elastic, Q_table=None): # 对战时策略
        if Q_table is None:
            Q = self.get_Q_dict(state)
        else:
            state, rotate_index = self.add_state(state)
            Q = Q_table[state] if state in Q_table else self.Q_table[state]
            Q =  {self.rotate_action(k, rotate_index):Q[k] for k in Q}
        self.curr_Q_value = Q
        Qmax = max(Q.values())  
        options = []        
        for index in Q:
            if Q[index] >= Qmax - elastic:
                options.append(index) 
        index = random.choice(options)
        return self.index_2_action(index)  
    
    def load_Q_table(self, path):
        self.Q_table = np.load(os.path.join(path, "Q_table.npy"), allow_pickle=True).item()
        self.states_map = {}
        for state_index in self.Q_table:
            state = self.index_2_state(state_index)
            rotated_states = self.get_rotate_states(copy.deepcopy(state), state_index)
            self.states_map.update(rotated_states)

    def save_Q_table(self, path):
        Q_table = {}
        for k, Q in self.Q_table.items():
            if max(abs(Q)) == 0:
                continue
            Q_table[k] = Q
        self.Q_table = Q_table
        np.save(os.path.join(path, "Q_table"), Q_table)
    def Qtable_2_trainset(self, path):
        if not os.path.exists(path):
                os.mkdir(path)
        states = []
        labels = []
        for index in self.states_map:
            main_index = self.states_map[index][0]
            Q = self.Q_table[main_index]
            Qvals = Q.values()
            Qmax = max(Qvals)
            if Qmax == 0:
                continue
            r = -Qmax if abs(Qmax) == 1 else -sum(Qvals)/len(Qvals)
            labels.append(r)
            states.append(self.index_2_state(index))             
        np.save(os.path.join(path, "states"), np.array(states))
        np.save(os.path.join(path, "labels"), np.array(labels))

    @abstractmethod
    def state_2_index(self, state):
        pass
    @abstractmethod
    def index_2_state(self, index):
        pass
    @abstractmethod
    def action_2_index(self, action):
        pass
    @abstractmethod
    def index_2_action(self, index):
        pass
    @abstractmethod
    def rotate_action(self, index, rotate_index):
        pass
    @abstractmethod
    def get_valid_actions(self, state): 
        #返回形式 [0,1,2,3......]
        pass
    @abstractmethod
    def get_rotate_states(self, state, actions, state_index):
        #返回形式{state_1_index:[标准state_index, {1:3,2:5......}]}
        pass
    @abstractmethod
    def state_2_model_data(self, state):
        pass

class TicTacToeAgent(ChessAgent):
    def __init__(self):       
        #旋转映射
        ChessAgent.__init__(self)
        mat = np.rot90(np.arange(9).reshape((3,3))).reshape(-1)
        rot_map = {}
        self.state_multi_array = []
        m = 1
        for i in range(9):
            rot_map[mat[i]] = i
            self.state_multi_array.append(m)
            m *= 3
        self.rot_map = {}   
        tmp =  {i:i for i in range(9)}  
        self.rot_map[0] = tmp.copy()        
        for i in [1,2,3]:
            tmp = {a:rot_map[tmp[a]] for a in tmp}
            self.rot_map[i] = tmp.copy()

        for i in range(9):
            x, y = self.index_2_action(i)
            index = self.action_2_index((y, x))
            tmp[i] = index

        self.rot_map[4] = tmp.copy()
        for i in [5,6,7]:
            tmp = {a:rot_map[tmp[a]] for a in tmp}
            self.rot_map[i] = tmp.copy()
        
    def state_2_index(self, state):
        arr = state.reshape(-1) + 1       
        return str(int(np.dot(arr, self.state_multi_array)))
    def index_2_state(self, index):
        ret = []
        index = int(index) 
        for s in self.state_multi_array[::-1]:
            k = index // s
            ret.append(k)            
            index = index - k * s
        ret = np.array(ret)[::-1] - 1           
        return ret.reshape((3,3))

    def index_2_action(self, index):
        return (index % 3, index // 3)

    def action_2_index(self, action):
        return action[1] * 3 + action[0] 

    def get_valid_actions(self, state):  
        actions = []
        for index in range(9):
            x, y = self.index_2_action(index)
            if state[y][x] == 0:
                actions.append(index)
        return actions
   
    def get_rotate_states(self, state, state_index):        
        def rotate(state, state_index, ret, start_index):
            for i in range(3):
                state = np.rot90(state)
                index = self.state_2_index(state)
                ret[index] = [state_index, start_index + i] 

        ret = {}
        ret[state_index] = [state_index, 0] 
        rotate(state, state_index, ret, 1)
        state = state.T
        index = self.state_2_index(state)
        ret[index] = [state_index, 4] 
        rotate(state, state_index, ret, 5)
        return ret 
    def rotate_action(self, index, rotate_index):
        return self.rot_map[rotate_index][index]
    def state_2_model_data(self, state):
        return state


class FourInALineAgent(ChessAgent):
    def __init__(self):       
        ChessAgent.__init__(self)
        self.state_multi_array  = np.array([1, 2, 4, 8, 16, 32, 64])
    
    def state_2_index(self, state):       
        state, num, _ = state
        state = (state + 1) / 2
        index = []
        for i in range(7):
            index.append(chr(int(np.dot(state[:num[i], i], self.state_multi_array[:num[i]]) + self.state_multi_array[num[i]])))        
        return "".join(index) 
    def index_2_state(self, index):       
        state = []
        for c in index:
            c = ord(c)
            for i in range(6):
                if c >= self.state_multi_array[5- i]:
                    c = c - self.state_multi_array[5 - i]                    
                    break
            col = []
            for j in range(6):
                if j <= i:
                    col.append(0)
                else:                    
                    s = self.state_multi_array[5 - j]
                    k = c // s                           
                    c = c - k * s
                    col.append(k*2-1)
            state.append(col[::-1])
        return np.array(state).T
    def index_2_action(self, index):
        return index
    def action_2_index(self, action):
        return action
    def get_valid_actions(self, state):  
        num = state[1]
        actions = []
        for index in range(7):
            if num[index] < 6 :
                actions.append(index)
        return actions
    def get_rotate_states(self, state, state_index):
        ret = {}
        ret[state_index] = [state_index, 1] 
        ret[state_index[::-1]] = [state_index, -1] 
        return ret
    def rotate_action(self, index, rotate_index):
        if rotate_index == 1:
            return index
        else:
            return 6 - index

    def set_Q_value(self, Q, state, action, state_index, act_index):
        if Q != self.Q_table[state_index][action]:
            self.Q_table[state_index][action] = Q            
            mirror_state = state_index[::-1]
            if mirror_state in self.Q_table:
                self.Q_table[mirror_state][6 - action] = Q
            else:
                Q_val = self.Q_table[state_index]
                self.Q_table[mirror_state] = state_index[::-1]
    def state_2_model_data(self, state):
        if state[2] %2 == 0:
            return state[0]
        else:
            return -state[0]
