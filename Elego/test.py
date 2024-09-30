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
import torch.nn as nn
from simpleencoder import SimpleEncoder
from extendedencoder import ExtendedEncoder
from zexperiencecollector import ZExperienceData
import copy
import pstats
import numpy as np

class WNet(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        policy = 0.04*torch.from_numpy(np.ones((5,5),dtype=np.float32))
        value = torch.tensor([0])
        return value, policy
    
def test(znet):
    zagent = ZAgent(znet,5,ExtendedEncoder(size=5),playouts=600,root_noise=False)
    #training.play_training_game(zagent,zagent,size=5)
    training.play_games(1,zagent,zagent,size=5)



'''
znet = ZNet(5)
cProfile.run('test(znet)','profile_results')
stats = pstats.Stats('profile_results')
stats.sort_stats('time').print_stats()

'''
'''

board_size = 9
encoder = ExtendedEncoder(size=board_size)
znet = ZNet(board_size)

zagent1 = ZAgent(znet,board_size,encoder,playouts=600,root_noise=False)


start_time = time.time()
training.play_games(1,zagent1,zagent1,size=board_size)
print("The time was",time.time()-start_time)
'''
board_size = 9
encoder = ExtendedEncoder(size=board_size)
board = Board(board_size)
gamestate = GameState(1,board)

np_board = encoder.encode_board(gamestate)
np_board.shape = (1,5,board_size,board_size)
print(np_board.shape)
np_boards = np.random.randint(2, size=(16384,5,board_size,board_size)).astype('float32')
print(np_boards.shape)

board_tensor = torch.from_numpy(np_board)
boards_tensor = torch.from_numpy(np_boards)
znet = ZNet(board_size)
znet.eval()


start_time = time.time()
with torch.no_grad():
    value, priors = znet(board_tensor)

value = value.item()
priors = priors.cpu().numpy()
print("The time was",time.time()-start_time)
print(priors.shape)
priors.shape = (board_size*board_size,)

print(priors.shape)


start_time = time.time()
with torch.no_grad():
    values, priors_list = znet(boards_tensor)

values = values.cpu().numpy()
priors_list = priors_list.cpu().numpy()
print("The time was",time.time()-start_time)
print(values.shape)
print(priors_list.shape)
