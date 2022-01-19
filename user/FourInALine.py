import gym
import numpy as np
def diag_count(shadow_state, shadow_state_mirror):
    mat = []        
    for i in range(6):
        d1 = np.diagonal(shadow_state, offset=i)
        d2 = np.diagonal(shadow_state_mirror, offset=i)
        mat.append(d1)
        mat.append(d2)
    return mat

class FourInALine(gym.Env):
    def reset(self):
        self.state = [np.zeros([6, 7]), np.zeros(7,dtype=np.int), 0] #0:棋盘状态 1：每列棋子数 2:棋子总数
        self._shadow_state = np.zeros([6, 11]) # 用于判断斜线连接
        self._shadow_state_mirror = np.zeros([6, 11])        
        self.player = 1
        self.trace = []
        self.judge_flag = False
     
    def step(self, action):
        row = self.state[1][action]        
        self.state[0][row][action] = self.player 
        self._shadow_state[row][action + 2] = self.player
        self._shadow_state_mirror[row][8 - action]  = self.player
        self.state[1][action] += 1  
        self.state[2] += 1  
        self.trace.append(action)
        reward, done = self.judge()  
        info = {}
        if done:
            info["winner"] = self.player * reward
        self.player *= -1
        return self.state, reward, done, info

    def rollback(self):
        action = self.trace.pop()
        self.player *= -1
        self.state[1][action] -= 1  
        self.state[2] -= 1  
        row = self.state[1][action]   
        self.state[0][row][action] = 0
        self._shadow_state[row][action + 2] = 0
        self._shadow_state_mirror[row][8 - action]  = 0
        
        

    def judge(self):
        def count_in_line(a, n, axis) :
            ret = np.cumsum(a, axis=axis)
            if axis == 0:
                ret[n:] = ret[n:] - ret[:-n]
                return abs(ret[n - 1:]) 
            else:
                ret[:, n:] = ret[:, n:] - ret[:, :-n]
                return abs(ret[:, n - 1:])
                
        #判除不可能达成终局的情况，以减少计算成本       
        if not self.judge_flag:
            count = 0
            for i in self.state[1]:
                if i == 0:
                    count = 0
                else:
                    count += 1
            self.judge_flag = self.state[2] > 3 and (max(self.state[1]) > 3 or count > 3)

        if not self.judge_flag:
            return 0, False           

        if 4 in count_in_line(self.state[0], 4, 0) or 4 in count_in_line(self.state[0], 4, 1):
            return 1, True

        #对角线
        mat = diag_count(self._shadow_state, self._shadow_state_mirror)
        if 4 in count_in_line(mat, 4, 1) :
            return 1, True
        if not 0 in self.state[0]: 
            return 0, True
        return 0, False 