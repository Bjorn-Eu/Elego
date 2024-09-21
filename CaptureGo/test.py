import cProfile
import time
import training
from board import Board
from move import Move
from gamestate import GameState
from random_agent import RandomAgent
from zagent import ZNet
from zagent import ZAgent
import torch
from simpleencoder import SimpleEncoder


def test():
    random_agent = RandomAgent()
    training.play_training_game(random_agent,random_agent)



#cProfile.run('test()')
#training.play_training_game(RandomAgent(),RandomAgent(),size=9)

ragent = RandomAgent()
znet = torch.jit.load('nets\\z_net.pt')
zagent = ZAgent(ZNet(9),9,SimpleEncoder(),root_noise=True,playouts=20)
start_time = time.time()
training.play_games(1000,ragent,ragent)
print("The time was",time.time()-start_time)

start_time = time.time()
training.play_games(50,zagent,zagent)
print("The time was",time.time()-start_time)




