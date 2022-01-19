import numpy as np
import copy
from chess.traversing_search import *
import numpy as np
import copy, os
from chess.traversing_search import *
class ChessTrainer():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent         

    def run_play(self, env, rollback_num, model):
        def set_reward(reward, trace):
            if len(trace) > 0:
                s, a = trace[-1]
                self.agent.Q_table[s][a] = reward
                if reward == 1 and len(trace) > 1:
                    s, a = trace[-2]
                    self.agent.Q_table[s][a] = -1

        trace = []
        state = env.state
        while True: 
            state_index, rotate_index = self.agent.add_state(state)           
            action = self.agent.random_policy(state_index)   
            if action is None: 
                Q = self.agent.Q_table[state_index]
                reward = -max(Q.values()) 
                set_reward(reward , trace)
                break
            trace.append((state_index, action)) 
            action_index = self.agent.rotate_action(action, rotate_index)
            real_action = self.agent.index_2_action(action_index)
            state, reward, over, info = env.step(real_action)  
            if over:
                set_reward(reward, trace)
                break
        searcher = None
        if rollback_num > 0 :
            for i in range(rollback_num):
                if len(trace) == 0:
                    break
                env.rollback()
                trace.pop() 
            searcher = traversing_search(env, self.agent)
            reward = searcher.substep_run(rollback_num, model)
        return searcher, trace, reward
    
    
    # def action_predict(self, env, model, Q_table=None):
    #     actions = self.agent.get_valid_actions(state) 
    #     for act in actions:
    #         state, over, reward, info = env.step()
            
    #     state, rotate_index = self.add_state(state)

    def update_predictions(self, searchers, model):
        predict_result = {}              
        for j, sch in searchers.items():
            treetop = dict(zip(sch.treetop_states_index, sch.treetop_states))
            predict_result.update(treetop)
        indexes, states = predict_result.keys(), np.array(list(predict_result.values()))
        if len(states) > 0:
            predict_result = dict(zip(indexes, model.predict(states, batch_size=256).reshape(-1))) 
        reward = {}
        for j in searchers:   
            reward[j] = searchers[j].substep_renew(predict_result)
        return reward

    def run(self, param_num, rollback_num, model=None):
        self.env.reset()        
        envs = [copy.deepcopy(self.env) for i in range(param_num)]
        searcher, traces, reward =  {}, {}, {}
        for j in range(param_num):
            searcher[j] , traces[j], reward[j] = self.run_play(envs[j], rollback_num, model)    
        if model is not None and rollback_num > 0:  
            reward = self.update_predictions(searcher, model)
        for j in range(param_num):
            next_state_index = None
            for state_index, action in traces[j][::-1]:  
                self.agent.update_Q_table(state_index, action, next_state_index, reward[j])            
                next_state_index = state_index

            
    def train(self, rounds, rollback_num, param_num, model=None, sample_path=None):   
        for i in range(rounds):     
            self.run(param_num, rollback_num, model)
        if sample_path is not None:
            if not os.path.exists(sample_path):
                os.mkdir(sample_path)
            self.Qtable_2_trainset(sample_path)

    def Qtable_2_trainset(self, path):
        self.Qtable_2_trainset(path)
        self.agent.save_Q_table(path)
        return

    def play(self, compare_agent, first=False, elastic=0, search_steps=1, compare_search_steps=1):
        self.env.reset()        
        self_turn = first
        while True: 
            agent = self.agent if self_turn else compare_agent
            steps = search_steps if self_turn else compare_search_steps
            self_turn = not self_turn                
            if agent is None:
                searcher = traversing_search(self.env, self.agent)
                searcher.run(search_steps, self.agent.model)
                self.agent.greedy(self.env.state, elastic)                 
                print(self.env.state)
                Q = self.agent.curr_Q_value
                options = [str(k) for k in Q.keys()]
                act =None
                while act not in options:                      
                    act = input(str(options ))[0]
                act = self.agent.index_2_action(int(act))
                print(self.env.player, Q, self.agent.state_2_index(self.env.state), act)
                print("***************************")
            else:
                searcher = traversing_search(self.env, agent)
                searcher.run(steps, agent.model)
                act = agent.greedy(self.env.state, elastic)   
                Q = agent.curr_Q_value
            if agent is not None:
                print(self.env.state)
                print(self.env.player, Q, self.agent.state_2_index(self.env.state), act)
                print("***************************")
            _, reward, over, info = self.env.step(act)
            if over:            
                break
        print(self.env.state)
        print("***************************")
        print(info)
        return info["winner"]
            
    def evaluate(self, rounds, compare_agent, elastic, search_steps=1, compare_search_steps=0,  param_num=10, show_fail_trace=False, result=None):     
        models = [self.agent.model, compare_agent.model]
        agents = [self.agent, compare_agent]
        search_num = [search_steps, compare_search_steps]
        self.agent.model, compare_agent.model = None, None        
        envs = [copy.deepcopy(self.env) for i in range(param_num)]
        num = rounds // param_num 
        num = num + 1 if num % 2 == 1 else num #为保证先后手数量相同
        if result is None:
            result = {0:0, -1:0, 1:0}
        else:
            result[0], result[1], result[-1] = 0, 0, 0
        for i in range(num):
            flag = i % 2 == 1
            winners = [envs[j].reset() for j in range(param_num)]              
            while True:
                agent_idx = 1 if flag else 0
                flag = not flag
                model, snum = models[agent_idx], search_num[agent_idx]
                searcher = {}          
                if snum > 0:                    
                    for j in range(param_num): 
                        if winners[j] is None:               
                            searcher[j] = traversing_search(envs[j], agents[agent_idx], set_Q=False)
                            searcher[j].substep_run(snum, model)
                    self.update_predictions(searcher, model)          
                over_flag = True
                for j in range(param_num):
                    if winners[j] is None:
                        Q_table = searcher[j].Q_table if j in searcher else None
                        act = agents[agent_idx].greedy(envs[j].state, elastic, Q_table)   
                        _, reward, over, info = envs[j].step(act)
                        if over: 
                            winners[j] = info["winner"]
                        else:
                            over_flag = False
                if over_flag:
                    for j in range(param_num):
                        result[ (1 - (i % 2) * 2) *  winners[j]] += 1
                        if  (1 - (i % 2) * 2) *  winners[j] == -1 and show_fail_trace:
                            s = "offensive" if i % 2 == 0 else "defensive"
                            print(s, envs[j].trace)                        
                    break
        self.agent.model, compare_agent.model = models
        print(result)
        return result   
        