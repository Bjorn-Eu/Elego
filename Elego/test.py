import cProfile
import time
import training
from gameplay.board import Board
import gameplay.board as board
from gameplay.move import Move
from gameplay.gamestate import GameState
from random_agent import RandomAgent
from human_agent import HumanAgent
from network import ZNet
from zagent import ZAgent
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from encoders.simpleencoder import SimpleEncoder
from encoders.extendedencoder import ExtendedEncoder
from zexperiencecollector import ZExperienceData
from zexperiencecollector import ZExperienceCollector
from selfplay import play_training_games
import transform
import selfplay
import copy
import pstats
import numpy as np
import treelib
import gameplay.config_board as config_board
import wnet
    
def test(znet):
    zagent = ZAgent(znet,5,ExtendedEncoder(size=5),playouts=150,root_noise=False)
    for i in range(10):
        training.play_training_game(zagent,zagent,size=5)



def print_np(np_array):
    for i in range(9):
        row = ""
        for j in range(9):
            row =row+ "{:1.0f}".format((np_array[i][j]))+" "
        print(row)

def self_play(size=9,device='cpu'):
    encoder = ExtendedEncoder(size=size)
    z_net = torch.jit.load('nets9x9//znet0.pt')
    #z_net = WNet(size=size)
    z_net.to(device)
    z_agentb = ZAgent(z_net,size,encoder,root_noise=True,playouts=50,device=device,batch_size=8)
    #z_agentw = ZAgent(z_net,size

def play_training_games(N,agent1,agent2,size=9,fileindex=0):
    p1_collector = ZExperienceCollector()
    p2_collector = ZExperienceCollector()
    agent1.set_collector(p1_collector)
    #agent2.set_collector(p2_collector)
    agent1_wins = 0
    agent2_wins = 0
    for i in range (N):
        if i%5 == 0:
            print("playing game", i, "out of",N)
        gamestate = play_training_game(agent1,agent2,size)
        if gamestate.turn == 1:
            agent2_wins += 1
            p1_collector.end_episode(-1)
            #p2_collector.end_episode(1)
        elif gamestate.turn == -1:
            agent1_wins += 1
            p1_collector.end_episode(1)
            #p2_collector.end_episode(-1)
        else:
            assert(False)

    print("The winrate was",agent1_wins/N)
    datap1 = p1_collector.to_data()
    datap1.write(f'selfplaydata\\capture{fileindex}')

def play_training_game(agent1,agent2,size=9,print=False):
    board = config_board.capture()

    gamestate = GameState(1,board)
    while not gamestate.is_over():
        if gamestate.turn == 1:
            move = agent1.select_move(gamestate)
        elif gamestate.turn == -1:
            move = agent2.select_move(gamestate)
        gamestate.move(move)
        if print:
            gamestate.print_state()
    return gamestate

def fit_stuff(size=9,fileindex=0):
    znet = torch.jit.load(f'nets9x9\\capture0.pt')

    value_loss_fn = nn.functional.mse_loss
    policy_loss_fn = nn.functional.binary_cross_entropy
    batch_size = 2
    learning_rate = 1e-2
    #optimizer = torch.optim.Adam(znet.parameters(),lr=learning_rate)
    optimizer = torch.optim.SGD(znet.parameters(), lr=learning_rate,momentum=0.0)


    datap1 = ZExperienceData()
    datap1.load(f'selfplaydata\\capture0.npz')

    game_data = datap1.gamestates.astype('float32')
    game_counts = datap1.visit_counts.astype('float32')
    game_rewards = datap1.rewards.astype('float32')
    

    samples = game_data.shape[0]
    game_data.shape = (samples,5,size,size)
    game_counts.shape = (samples,size*size)
    game_rewards.shape = (samples,1)

    train_boards_tensor = torch.from_numpy(game_data)
    train_counts_tensor = torch.from_numpy(game_counts)
    train_rewards_tensor = torch.from_numpy(game_rewards)


    train_dataset = TensorDataset(train_boards_tensor,train_counts_tensor,train_rewards_tensor)
    train_dataloader = DataLoader(train_dataset,batch_size)

    for i in range(1000):
        print("epoch", i)
        training.train_loop(znet,train_dataloader,value_loss_fn,policy_loss_fn,optimizer)

    model_scripted = torch.jit.script(znet) # Export to torchscript
    model_scripted.save(f'nets9x9\\capture1.pt')


encoder = ExtendedEncoder()
board = config_board.atari_go9x9()
gamestate = GameState(1,board)
znet = torch.jit.load('nets9x9\\znet21.pt')
capture_net = wnet.WNET_ADV()#wnet.WNET_Q()#wnet.WNET_ADV()
zagent = ZAgent(znet,9,encoder,playouts=150,batch_size=8,root_noise=True)
zagent2 = ZAgent(capture_net,9,encoder,playouts=25,batch_size=1,root_noise=True)

selfplay.play_games(100,zagent,zagent2)
selfplay.play_games(100,zagent2,zagent)
#selfplay.play_training_game(zagent,zagent2,print=True)


