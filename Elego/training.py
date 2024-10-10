
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from board import Board
from move import Move
from gamestate import GameState
from network import ZNet
from zagent import ZAgent
from random_agent import RandomAgent
from zexperiencecollector import ZExperienceCollector
from zexperiencecollector import ZExperienceData
from simpleencoder import SimpleEncoder
from extendedencoder import ExtendedEncoder
import time
import copy

def play_training_games(N,agent1,agent2,size=9,fileindex=0):
    p1_collector = ZExperienceCollector()
    p2_collector = ZExperienceCollector()
    agent1.set_collector(p1_collector)
    agent2.set_collector(p2_collector)
    agent1_wins = 0
    agent2_wins = 0
    for i in range (N):
        if i%5 == 0:
            print("playing game", i, "out of",N)
        gamestate = play_training_game(agent1,agent2,size)
        if gamestate.turn == 1:
            agent2_wins += 1
            p1_collector.end_episode(-1)
            p2_collector.end_episode(1)
        elif gamestate.turn == -1:
            agent1_wins += 1
            p1_collector.end_episode(1)
            p2_collector.end_episode(-1)

    print("The winrate was",agent1_wins/N)
    

    datap1 = p1_collector.to_data()
    datap2 = p2_collector.to_data()
    datap1.write(f'selfplaydata\\zagentb{fileindex}')
    datap2.write(f'selfplaydata\\zagentw{fileindex}')


def self_play(size=9,device='cpu',fileindex=0):
    encoder = ExtendedEncoder(size=size)
    z_net = torch.jit.load(f'nets\\znet{fileindex}.pt')
    z_net.to(device)
    z_agentb = ZAgent(z_net,size,encoder,playouts=50,device=device)
    z_agentw = ZAgent(z_net,size,encoder,playouts=50,device=device)
    play_training_games(5000,z_agentb,z_agentw,size,fileindex=fileindex)

def play_games(N,agent1,agent2,size=9):
    agent1_wins = 0
    agent2_wins = 0
    for i in range (N):
        if i%10 == 0:
            print("playing game", i, "out of",N)
        gamestate = play_training_game(agent1,agent2,size)
        if gamestate.turn == 1:
            agent2_wins += 1
        elif gamestate.turn == -1:
            agent1_wins += 1
    print("The winrate was",agent1_wins/N)



def test(size=9,fileindex=0):
    encoder = ExtendedEncoder(size=size)
    z_net = torch.jit.load(f'nets9x9\\znet{fileindex}.pt')
    z_agent = ZAgent(z_net,size,encoder,root_noise=False,playouts=10)

    play_games(1000,z_agent,RandomAgent(),size)
    play_games(1000,RandomAgent(),z_agent,size)

def play_training_game(agent1,agent2,size=9):
    board = Board(size)

    gamestate = GameState(1,board)
    while not gamestate.is_over():
        if gamestate.turn == 1:
            move = agent1.select_move(gamestate)
        elif gamestate.turn == -1:
            move = agent2.select_move(gamestate)
        gamestate.move(move)
        #gamestate.print_state()
    return gamestate

def train_loop(net,dataloader,value_loss_fn,policy_loss_fn,optimizer):
    net.train()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0
    total_value_loss = 0
    total_policy_loss = 0
    for train_board, train_policy_update, train_adj_reward in dataloader:
        value, policy = net(train_board)
        loss_value = value_loss_fn(value,train_adj_reward)
        loss_policy = policy_loss_fn(policy,train_policy_update)
        loss = loss_value+loss_policy
        total_value_loss += loss_value.item()
        total_policy_loss += loss_policy.item()
        total_loss += loss.item()

        #backpropagate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"The average loss was: {total_loss/num_batches:.4f}")
    print(f"The average value loss was: {total_value_loss/num_batches:.4f}")
    print(f"The average policy loss was: {total_policy_loss/num_batches:.4f}")

def fit_stuff(size=9,fileindex=0):
    z_net = torch.jit.load(f'nets\\znet{fileindex}.pt')

    value_loss_fn = nn.MSELoss()
    policy_loss_fn = nn.CrossEntropyLoss()
    batch_size = 32
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(z_net.parameters(), lr=learning_rate,momentum=0.9)


    datap1 = ZExperienceData()
    datap1.load(f'selfplaydata\\zagentb{fileindex}.npz')

    datap2 = ZExperienceData()
    datap2.load(f'selfplaydata\\zagentw{fileindex}.npz')

    game_data = np.append((datap1.gamestates).astype('float32'),(datap2.gamestates).astype('float32'),0) 
    game_counts = np.append(datap1.visit_counts,datap2.visit_counts,0).astype('float32')
    game_rewards = np.append(datap1.rewards,datap2.rewards).astype('float32')

    samples = game_data.shape[0]
    game_data.shape = (samples,5,size,size)
    game_counts.shape = (samples,size*size)
    game_rewards.shape = (samples,1)

    train_boards_tensor = torch.from_numpy(game_data)
    train_counts_tensor = torch.from_numpy(game_counts)
    train_rewards_tensor = torch.from_numpy(game_rewards)


    train_dataset = TensorDataset(train_boards_tensor,train_counts_tensor,train_rewards_tensor)
    train_dataloader = DataLoader(train_dataset,batch_size)

    train_loop(z_net,train_dataloader,value_loss_fn,policy_loss_fn,optimizer)

    model_scripted = torch.jit.script(z_net) # Export to torchscript
    model_scripted.save(f'nets\\znet{fileindex+1}.pt')


