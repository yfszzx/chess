import gym
import numpy as np
class TicTacToe(gym.Env):
    def reset(self):
        self.state = np.zeros([3, 3])
        self.player = 1
        self.trace = []
     
    def step(self, action):
        # action =(x,y)
        x, y = action
        self.state[y][x] = self.player 
        self.trace.append(action)
        if len(self.trace) > 4:
            reward, done = self.judge() 
        else:
            reward, done = 0, False
        info = {}
        if done:
            info["winner"] = self.player * reward
        self.player *= -1
        return self.state, reward, done, info

    def rollback(self):
        x, y = self.trace.pop()
        self.player *= -1
        self.state[y][x] = 0 
        
      
    def judge(self):
        #对角线
        diag = [np.diagonal(self.state).sum(), np.diagonal(np.rot90(self.state)).sum()]
        if 3 in diag or  -3 in diag:
            return 1, True
        #行列
        row, col = self.state.sum(axis=0), self.state.sum(axis=1)
        if 3 in row or 3 in col or -3 in row or -3 in col:
                return 1, True
        if not 0 in self.state:
            return 0, True
        return 0, False 
