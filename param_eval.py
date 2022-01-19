import gym
import random
import numpy as np
import copy
from abc import abstractmethod 
import chess.chess_agent as ca
from chess.traversing_search import *
import os
import numpy as np
import copy
import numpy as np
import copy


import os


import numpy as np
import random


def eval(queue):
    
    # env =gym.make('TicTacToe-v0')
    # agent = ca.TicTacToeAgent()
    # rnd_agent = ca.TicTacToeAgent()
    # model = TicTacToe_test_net()
    # agent.set_deepnet(model)
    # trainer = ChessTrainer(env, agent)
    # result =  trainer.evaluate(1000, rnd_agent, 0, search_steps=2, param_num=4)  
    import tensorflow as tf
    from tensorflow.keras import datasets, layers, models
    env = gym.make('FourInALine-v0')
    agent = ca.FourInALineAgent()
    rnd_agent =  ca.FourInALineAgent()    
    model =tf.keras.models.load_model("FourInALine/best_model")  

    trainer = ChessTrainer(env, agent)
    result =  trainer.evaluate(400, rnd_agent, 0, search_steps=2, param_num=4)  
    #queue.put(result, block=False)


# agent.set_deepnet(model)
# trainer = ChessTrainer(env, agent)
#result =  trainer.evaluate(1000, rnd_agent, 0, search_steps=2, param_num=4)  
if __name__ == '__main__':
    #queue = multiprocessing.Queue()
    queue = None
    t1 = multiprocessing.Process(target=eval, args=(queue,))
    t2 = multiprocessing.Process(target=eval, args=(queue,))
    #t3 = multiprocessing.Process(target=eval, args=(queue,))
    t1.start()
    t2.start()
    #t3.start()
    t1.join()
    t2.join()
    #t3.join()
    # print("test1",queue.get())
    # print("test2",queue.get())
   # print("test3",queue.get())