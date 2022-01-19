import numpy as np
class traversing_search():
    def __init__(self, env, agent, deepnet=None):
        self.searched_states = {}
        self.env = env
        self.agent = agent
        self.deepnet = deepnet
        self.search_tree = {}
        self.over_state_index = 1
        self.final_states = {}

    def n_step_search(self, n):
        if n < 1:
            return None
        state_index, rotate_index = self.agent.add_state(self.env.state)
        if state_index in self.searched_states:
            return self.searched_states[state_index]    
        Q = self.agent.Q_table[state_index]
        Qmax = max(Q.values())
        if abs(Qmax) == 1:
            self.searched_states[state_index] = -Qmax
            return -Qmax        
        for action, q in Q.items():       
            if q > -1:
                action_index = self.agent.rotate_action(action, rotate_index)
                real_action = self.agent.index_2_action(action_index)
                state, reward, over, info = self.env.step(real_action)                          
                if over:
                    Q[action] = reward
                    if reward == 1:                        
                        self.env.rollback()
                        self.searched_states[state_index] = -1
                        return -1
                elif n == 1:
                    next_state, _ = self.agent.add_state(self.env.state)
                    vals = self.agent.Q_table[next_state].values()
                    Qmax = max(vals)
                    if abs(Qmax) == 1:
                        Q[action] = -Qmax
                        if Qmax == -1:                            
                            self.env.rollback() 
                            self.searched_states[state_index] = -1
                            return -1
                    else:
                        Q[action] = -sum(vals) / len(vals) 
                else:    
                    Q[action] = self.n_step_search(n - 1)                              
                self.env.rollback()
        Qvals = Q.values()
        Qmax = max(Qvals)
        r = -Qmax if abs(Qmax) == 1 else -sum(Qvals)/len(Qvals)               
        self.searched_states[state_index] = r
        return  r 
    
    def deepnet_search(self, n, search_tree):
        def set_state_record(next_state, state, action):
            if next_state not in  self.Q_table_record:
                self.Q_table_record[next_state] = [[state, action]]
            else:
                self.Q_table_record[next_state].append([state, action])

        if n < 1:
            return None

        state_index, rotate_index = self.agent.add_state(self.env.state)
        if state_index in self.searched_states:
            return self.searched_states[state_index], state_index 
        Q = self.agent.Q_table[state_index]
        Qmax = max(Q.values())
        if abs(Qmax) == 1:
            self.searched_states[state_index] = -Qmax
            return -Qmax, state_index
        search_tree[state_index] = {} 
        no_object = True
        for action, q in Q.items():       
            if q > -1:
                action_index = self.agent.rotate_action(action, rotate_index)
                real_action = self.agent.index_2_action(action_index)
                state, reward, over, info = self.env.step(real_action)                          
                if over:
                    Q[action] = reward
                    if reward == 1:                        
                        self.env.rollback()
                        self.searched_states[state_index] = -1
                        return -1, state_index
                elif n == 1:
                    next_state, _ = self.agent.add_state(self.env.state)
                    vals = self.agent.Q_table[next_state].values()
                    Qmax = max(vals)
                    if abs(Qmax) == 1:
                        Q[action] = -Qmax
                        if Qmax == -1:                            
                            self.env.rollback() 
                            self.searched_states[state_index] = -1
                            return -1, state_index
                    else:  
                        self.over_state_index += 1                      
                        self.final_states[self.over_state_index] = copy.deepcopy(self.env.state)
                        set_state_record(next_state, state_index, action)                        
                        return self.over_state_index, state_index                    
                else: 
                    r, next_state =  self.deepnet_search(n - 1, search_tree[state_index]) 
                    if  type(r) != 'dict' and r <= 1:
                        Q[action] = r 
                    else:
                        set_state_record(next_state, state_index, action)
                        no_object = False                                                
                self.env.rollback()
        if no_object:
            Qvals = Q.values()
            Qmax = max(Qvals)
            r = -Qmax if abs(Qmax) == 1 else -sum(Qvals)/len(Qvals)   
            self.searched_states[state_index] = r
            return r, state_index
        else:
            return search_tree, state_index

    def deep_caculate(self, tree):
        def set_Q_value(state_index, value):
            record = self.Q_table_record[state_index]
            for state, action in record:
                self.agent.Q_table[state][action] = s

        Q_vals = []        
        for state_index, val in  tree.items():
            if type(val) == 'dict':
                v = deep_caculate(self,  val)
                set_Q_value(state_index, v)  

            elif val > 1:
                v = deep_result[val]
                set_Q_value(state_index, v)   
            else:
                v = val
            Q_vals.append(v)             
        return result