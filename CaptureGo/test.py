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
    znet = ZNet(size=5)
    zagent = ZAgent(znet,5,ExtendedEncoder(size=5))
    #training.play_training_game(zagent,zagent,size=5)
    training.play_games(1,zagent,zagent,size=5)

'''
cProfile.run('test2()','profile_results')
stats = pstats.Stats('profile_results')
stats.sort_stats('time').print_stats()
'''


'''
znet = torch.jit.load('nets\\z_net.pt')
zagent = ZAgent(znet,9,SimpleEncoder())
training.play_training_game(zagent,zagent)

znet = torch.jit.load('nets\\znet.pt')
zagent = ZAgent(znet,5,ExtendedEncoder(size=5),root_noise=False,playouts=1)
board = Board(5)
gamestate = GameState(1,board)
move = zagent.select_move(gamestate)
move.print()
'''



ragent = RandomAgent()

start_time = time.time()
training.play_games(1000,ragent,ragent)
print("The time was",time.time()-start_time)


znet = torch.jit.load('nets\\znetb6n27.pt')
zagent = ZAgent(znet,5,ExtendedEncoder(size=5),root_noise=False,playouts=500)

start_time = time.time()
training.play_games(100,zagent,RandomAgent(),size=5)
print("The time was",time.time()-start_time)

