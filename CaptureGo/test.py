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
from zexperiencecollector import ZExperienceData
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



'''
ragent = RandomAgent()

start_time = time.time()
training.play_games(100,ragent,ragent)
print("The time was",time.time()-start_time)
'''
'''
encoder = ExtendedEncoder(size=5)
znet = torch.jit.load('nets\\znetb6n0.pt')
zagentb = ZAgent(znet,5,encoder,playouts=1)
zagentw = ZAgent(znet,5,encoder,playouts=1)
start_time = time.time()
training.play_training_games(2,zagentb,zagentw,size=5,fileindex=0)
print("The time was",time.time()-start_time)
'''
'''

datap1 = ZExperienceData()
datap1.load(f'selfplaydata\\zagentb00.npz')

datap2 = ZExperienceData()
datap2.load(f'selfplaydata\\zagentw00.npz')

print(datap1.gamestates)
print(datap1.rewards)
print(datap1.visit_counts)

print("White players:")
print(datap2.gamestates)
print(datap2.rewards)
print(datap2.visit_counts)
'''

encoder = ExtendedEncoder(size=2)
#znet = torch.jit.load('nets\\znetb6n0.pt')
znet = torch.jit.load('nets\\znet2x29.pt')

zagentb = ZAgent(znet,2,encoder,playouts=400,root_noise=False)
#training.test(size=2,fileindex=22)

board = Board(2)
#board.move(Move(1,1,1))
#board.move(Move(-1,2,1))
#board.move(Move(-1,1,2))
gamestate = GameState(1,board)
root = zagentb.select_move(gamestate,printy=True)
child = root.children[2]
print("Child 3")
for branch in child.branches:
    print("Branch index:",branch,"Prior:",child.branches[branch].prior)
    print("Branch visits",child.branches[branch].visits,"Q value:",child.expected_value(branch))
#training.fit_stuff(size=5,fileindex=0)