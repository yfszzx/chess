import gym
import numpy as np

class FourInALine(gym.Env):
    def reset(self):
        self.state = np.zeros([15, 15])        
        self._shadow_state = np.zeros([15, 29]) # 用于判断斜线连接
        self._shadow_state_mirror = np.zeros([15, 29])    
        self.state[6][6] = 1   
        self._shadow_state[6][14] = 1
        self._shadow_state_mirror[6][14] = 1
        self.player = -1
        self.trace = []
        
    def check_data(self, x, y):
        min_y = max(0, y - 4) 
        max_y = y + 4 
        return [self.state[y,max(0, x - 4) :x + 4], self.state[min_y :max_y, x], self._shadow_state[min_y:max_y , 14 + x - y], self._shadow_state[min_y:y, 28 - x - y]]
    def step(self, action):
        self.trace.append(action)
        x, y = action      
        self.state[y][x] = self.player 
        self._shadow_state[y][14 + x - y] = self.player
        self._shadow_state_mirror[y][28 - x - y]  = self.player
        
        reward, done = self.judge(x, y)  
        info = {}
        if done:
            info["winner"] = self.player * reward
        self.player *= -1
        return self.state, reward, done, info

    def rollback(self):
        action = self.trace.pop()
        x, y = action        
        self.state[y][x] = 0
        self._shadow_state[y][14 + x - y] = 0
        self._shadow_state_mirror[y][28 - x - y]  = 0

    def judge(self, x, y):
        def count_in_line(a, n, axis) :
            ret = np.cumsum(a, axis=axis)
            if axis == 0:
                ret[n:] = ret[n:] - ret[:-n]
                return abs(ret[n - 1:]) 
            else:
                ret[:, n:] = ret[:, n:] - ret[:, :-n]
                return abs(ret[:, n - 1:]) 
        state = self.check_data(x, y)
        for s in state:
            if len(s) >=5:
                v = abs(np.cumsum(s)[5:]).max()
                ret[n:] = ret[n:] - ret[:-n]
                if v == 5:
                    return 1, True            
        if not 0 in self.state: #先手负
            return -1, True
        return 0, False

    def forbid_pos(self, pos):
        act = x, y
        min_y = max(0, y - 3) 
        state = [self.state[y,max(0, x - 3) :x + 3], self.state[min_y :y + 3, x], self._shadow_state[min_y:y, 14 + x - y], self._shadow_state[min_y:y, 28 - x - y]]