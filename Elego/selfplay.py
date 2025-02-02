import torch
from zagent import ZAgent
from network import ZNet
from gameplay.board import Board
from gameplay.move import Move
from gameplay.gamestate import GameState
from zexperiencecollector import ZExperienceCollector
from zexperiencecollector import ZExperienceData
from encoders.simpleencoder import SimpleEncoder
from encoders.extendedencoder import ExtendedEncoder
from random_agent import RandomAgent
import gameplay.config_board as config_board

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
        else:
            assert(False)

    print("The winrate was",agent1_wins/N)
    datap1 = p1_collector.to_data()
    datap2 = p2_collector.to_data()
    datap1.write(f'selfplaydata\\9x9b{fileindex}')
    datap2.write(f'selfplaydata\\9x9w{fileindex}')


def self_play(size=9,device='cpu',fileindex=0):
    encoder = ExtendedEncoder(size=size)
    z_net = torch.jit.load(f'nets9x9\\znet{fileindex}.pt')
    z_net.to(device)
    z_agentb = ZAgent(z_net,size,encoder,root_noise=True,playouts=300,device=device,batch_size=8)
    z_agentw = ZAgent(z_net,size,encoder,root_noise=True,playouts=300,device=device,batch_size=8)
    play_training_games(3000,z_agentb,z_agentw,size,fileindex=fileindex)

def play_games(N,agent1,agent2,size=9):
    agent1_wins = 0
    agent2_wins = 0
    for i in range (N):
        if i%20 == 0:
            print("playing game", i, "out of",N)
        gamestate = play_training_game(agent1,agent2,size)
        if gamestate.turn == 1:
            agent2_wins += 1
        elif gamestate.turn == -1:
            agent1_wins += 1
    print("The winrate was",agent1_wins/N)



def test(size=9,fileindex=0):
    encoder = ExtendedEncoder(size=size)
    znet = torch.jit.load(f'nets9x9\\znet{fileindex}.pt')
    znet2 = torch.jit.load(f'nets9x9\\znet{fileindex+1}.pt')
    z_agent = ZAgent(znet,size,encoder,root_noise=True,playouts=50,batch_size=8)
    z_agent2 = ZAgent(znet2,size,encoder,root_noise=True,playouts=50,batch_size=8)


    play_games(200,z_agent,z_agent2,size)
    play_games(200,z_agent2,z_agent,size)

def cf_nets(size=9,fileindex1=0,fileindex2=1,games=200):
    encoder = ExtendedEncoder(size=size)
    znet1 = torch.jit.load(f'nets9x9\\znet{fileindex1}.pt')
    znet2 = torch.jit.load(f'nets9x9\\znet{fileindex2}.pt')
    zagent1 = ZAgent(znet1,size,encoder,root_noise=True,playouts=250,batch_size=8)
    zagent2 = ZAgent(znet2,size,encoder,root_noise=True,playouts=250,batch_size=8)
    play_games(games,zagent2,zagent1,size)
    play_games(games,zagent1,zagent2,size)

def playout_game():
    znet = wnet.WNET_PURE()
    znet2 = wnet.WNET_PURE()
    zagent = ZAgent(znet,9,ExtendedEncoder(),playouts=5000,root_noise=False,batch_size=1)
    zagent2 = ZAgent(znet2,9,ExtendedEncoder(),playouts=5000,root_noise=False,batch_size=1)
    play_training_game(zagent,zagent2,print=True)

def play_training_game(agent1,agent2,size=9,print_state=False):
    board = config_board.random_board9x9()

    gamestate = GameState(1,board)
    while not gamestate.is_over():
        if gamestate.turn == 1:
            move = agent1.select_move(gamestate)
        elif gamestate.turn == -1:
            move = agent2.select_move(gamestate)
        gamestate.move(move)
        if print_state:
            gamestate.print_state()
    return gamestate