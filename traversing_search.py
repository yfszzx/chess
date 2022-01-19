import numpy as np
import copy
class traversing_search():
    STATE_INDEX = 0
    ACTIONS = 1
    def __init__(self, env, agent, set_Q=True):
        self.searched_states = {}
        self.env = env
        self.agent = agent
        self.set_Q = set_Q
        self.reset()
        

    def reset(self):   
        self.treetop_states = []
        self.treetop_states_index = []
        self.stepped_states_record = {}
        self.predict_result = None
        self.Q_table = self.agent.Q_table if self.set_Q else {}
        

    #对于神经网络预测模式（net_mode）stepped_states_record: float, int 代表最终结果； dict 代表迭代树； str代表末梢状态
    def n_step_search(self, n, net_mode):
        def break_return(state_index, value=-1, rollback=True):
            if rollback:
                self.env.rollback()
            self.searched_states[state_index] = value
            return value
        def set_Q_value(Q, val, action, next_state, return_dict=None):
            self.stepped_states_record[next_state] = val
            Q[action] = val            
            if return_dict is not None:
                return_dict[self.ACTIONS][action] = val

        def set_return_dict(return_dict, action, next_state, val):
            self.stepped_states_record[next_state] = val 
            return_dict[self.ACTIONS][action] = val                                           
            return False

        def valid_value(val, net_mode):
            return not net_mode or type(val) == int or type(val) == float

        def get_Q_values(index, value_mode):
            if self.set_Q:
                Q = self.agent.Q_table[index]
            else:
                if index not in self.Q_table:
                    Q = copy.deepcopy(self.agent.Q_table[index])
                    self.Q_table[index] = Q
                else:
                    Q = self.Q_table[index]

            Q_vals = Q.values()
            Qmax = max(Q.values())
            if value_mode:
                return Q_vals, Qmax
            else:
                return Q, Qmax

        if n < 1:
            return None

        state_index, rotate_index = self.agent.add_state(self.env.state)
        if state_index in self.searched_states:
            return self.searched_states[state_index] 

        Q, Qmax = get_Q_values(state_index, value_mode=False)       
        if abs(Qmax) == 1:
            return break_return(state_index, value=-Qmax, rollback=False)   

        no_object = True 
        return_dict =  {self.STATE_INDEX:state_index, self.ACTIONS:{}} if net_mode else None              

        for action, q in Q.items():       
            if q > -1:
                action_index = self.agent.rotate_action(action, rotate_index)
                real_action = self.agent.index_2_action(action_index)
                state, reward, over, info = self.env.step(real_action)                          
                if over:
                    Q[action] = reward
                    if reward == 1:   
                        return break_return(state_index)
                else:
                    next_state, _ = self.agent.add_state(self.env.state)
                    if next_state in self.stepped_states_record:
                        val = self.stepped_states_record[next_state]
                        if valid_value(val, net_mode):
                            Q[action] = val
                            if n == 1 and val == 1:
                                return break_return(state_index)
                        if net_mode:
                            return_dict[self.ACTIONS][action] = val 
                            if type(val) == dict or type(val) == str:
                                no_object = False
                    else:
                        if n == 1:                   
                            next_Q, Qmax = get_Q_values(next_state, True)
                            if abs(Qmax) == 1:
                                set_Q_value(Q, -Qmax, action, next_state, return_dict)
                                if Qmax == -1:                
                                    return break_return(state_index)
                            else:
                                if not net_mode:
                                    val = -sum(next_Q) / len(next_Q)
                                    set_Q_value(Q, val, action, next_state, None)
                                else:
                                    no_object = set_return_dict(return_dict, action, next_state, next_state)
                                    self.treetop_states_index.append(next_state)
                                    self.treetop_states.append(copy.deepcopy(self.agent.state_2_model_data(state)))
                        else:    
                            val = self.n_step_search(n - 1, net_mode)                             
                            if valid_value(val, net_mode):
                                set_Q_value(Q, val, action, next_state, return_dict)
                            else:                                                               
                                no_object = set_return_dict(return_dict, action, next_state, val)              
                self.env.rollback()

        if not net_mode or no_object:
            Qvals, Qmax = get_Q_values(state_index, True)         
            r = -Qmax if abs(Qmax) == 1 else -sum(Qvals)/len(Qvals)               
            return break_return(state_index, value=r, rollback=False) 
        else:
            return break_return(state_index, value=return_dict, rollback=False) 
            

    def deep_caculate(self, tree):
        state_index, actions = tree[self.STATE_INDEX], tree[self.ACTIONS] 
        for act, val in  actions.items():
            k = val
            if type(val) == dict:
                val = self.deep_caculate(val)
            elif type(val) == str:
                val = self.predict_result[val]
            self.Q_table[state_index][act] = val
        Q_vals = self.Q_table[state_index].values()
        Qmax = max(Q_vals)
        r = -Qmax if abs(Qmax) == 1 else -sum(Q_vals)/len(Q_vals) 
        return r
            
    def run(self, step_num, model=None):
        self.substep_run(step_num, model)
        if model is not None:
            result = {}
            if len(self.treetop_states) > 0:
                result = model.predict(np.array(self.treetop_states)).reshape(-1)
                result = dict(zip(self.treetop_states_index, result))
            return self.substep_renew(result)
        else:
            return self.search_result
    def substep_run(self, step_num, model):
        self.reset()
        self.search_result = self.n_step_search(step_num, model is not None)
        return self.search_result

    def substep_renew(self, predict_result):
        self.predict_result = predict_result
        if type(self.search_result) == dict:
            return self.deep_caculate(self.search_result)
        else:
            return self.search_result


import gym
import chess.chess_agent as ca
class TicTacToe_test_net():
    def __init__(self):
        env =gym.make('TicTacToe-v0')
        env.reset()
        agent = ca.TicTacToeAgent()        
        searcher = traversing_search(env, agent)
        searcher.run(10)
        self.agent = agent
    def predict(self, states, batch_size=None):
        ret = []
        for s in states:
            Q = self.agent.get_Q_dict(s)
            Qvals = Q.values()
            Qmax = max(Qvals)
            r = -Qmax * 0.999 if abs(Qmax) == 1 else -sum(Qvals)/len(Qvals)  
            ret.append(r)

        return np.array(ret)