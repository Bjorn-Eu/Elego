import cProfile
import time
import training
from board import Board
import board
from move import Move
from gamestate import GameState
from random_agent import RandomAgent
from zagent import ZNet
from zagent import ZAgent
import torch
from simpleencoder import SimpleEncoder
from extendedencoder import ExtendedEncoder
import copy
import pstats


def test():
    random_agent = RandomAgent()
    training.play_training_game(random_agent,random_agent)

def test2():
    znet = torch.jit.load('nets\\z_net.pt')
    zagent = ZAgent(znet,9,SimpleEncoder())
    training.play_training_game(zagent,zagent)
    #training.play_games(10,zagent,zagent)

'''
cProfile.run('test2()','profile_results')
stats = pstats.Stats('profile_results')
stats.sort_stats('time').print_stats()
'''

'''
znet = torch.jit.load('nets\\z_net.pt')
zagent = ZAgent(znet,9,SimpleEncoder())
training.play_training_game(zagent,zagent)
'''




'''
gamestate = training.play_training_game(RandomAgent(),RandomAgent(),size=5)
print(gamestate.board.adjacent_liberties((Move(1,3,3))))
'''



ragent = RandomAgent()

start_time = time.time()
training.play_games(1000,ragent,ragent)
print("The time was",time.time()-start_time)




znet = torch.jit.load('nets\\znet.pt')
zagent = ZAgent(ZNet(5),5,ExtendedEncoder(size=5),root_noise=True,playouts=50)

start_time = time.time()
training.play_games(50,zagent,zagent,size=5)
print("The time was",time.time()-start_time)








