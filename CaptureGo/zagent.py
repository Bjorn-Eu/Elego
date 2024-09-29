import torch
import torch.nn as nn
import numpy as np
from board import Board
from gamestate import GameState
from move import Move
from agent import Agent
from branch import Branch
from random_agent import RandomAgent
from encoder import Encoder
import copy
from zexperiencecollector import ZExperienceCollector
from zexperiencecollector import ZExperienceData

class Block(nn.Module):
    def __init__(self,c=64):
        super().__init__()
        self.c = c
        self.batch1 = nn.BatchNorm2d(self.c)
        self.conv1 = nn.Conv2d(self.c,self.c,3,padding='same',bias=False)
        self.batch2 = nn.BatchNorm2d(self.c)
        self.conv2 = nn.Conv2d(self.c,self.c,3,padding='same',bias=False)
        
        
    def forward(self,input):
        x=self.batch1(input)
        x=torch.relu(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x+input
        return x

class ZNet(nn.Module):

    def __init__(self,size):
        super().__init__()
        self.c = 64
        self.c_policy = 16
        self.c_value = 16
        self.n = 6

        #input
        self.size = self.c*size*size
        self.conv0 = nn.Conv2d(5,self.c,3,padding='same',bias=False)

        #Blocks
        self.blocks = nn.ModuleList([Block() for i in range(0,self.n)])
        self.batch1 = nn.BatchNorm2d(self.c)

        #value head
        self.value_linear = nn.Linear(self.size,self.c_value)
        self.value_output = nn.Linear(self.c_value,1)

        #policy head
        self.policy_linear = nn.Linear(self.size,self.c_policy)
        self.policy_batch = nn.BatchNorm1d(self.c_policy)
        self.policy_output = nn.Linear(self.c_policy,size*size)
        self.soft_max = nn.Softmax(dim=1)


    def forward(self,x):
        #input
        x=self.conv0(x)
        
        #blocks
        for block in self.blocks:
            x = block(x)

        x=self.batch1(x)
        x=torch.relu(x)
        x=x.view(-1,self.size)

        #value output
        value = self.value_linear(x)
        value = torch.relu(value)
        value = self.value_output(value)
        value = torch.tanh(value)

        #policy output
        policy = self.policy_linear(x)
        policy = self.policy_batch(policy)
        policy = torch.relu(policy)
        policy = self.policy_output(policy)
        policy = self.soft_max(policy)

        return value, policy

class ZAgent(Agent):

    def __init__(self,net,size,encoder,root_noise=True,playouts=150,device='cpu'):
        self.net = net
        self.size = size
        self.encoder = encoder
        self.collector = None
        self.root_noise = root_noise
        self.playouts = playouts
        self.device = device

    def set_playouts(self,playouts):
        self.playouts = playouts
        
    def set_collector(self,collector):
        self.collector = collector

    def select_move(self,gamestate):
        root = self.create_node(gamestate,noise=self.root_noise)
        for i in range(self.playouts):
            node = root
            #walk down to leaf node
            next_move = node.select_branch()
            while node.has_child(next_move):

                node = node.get_child(next_move)
                if(node.is_terminal):
                    break
                else:
                    next_move = node.select_branch()

            if(node.is_terminal):
                node.update_wins(1)
            else:
                next_move_X = self.id_to_move(node.gamestate.turn,next_move)
                new_gamestate = copy.deepcopy(node.gamestate)
                new_gamestate.move(next_move_X)
                child = self.create_node(new_gamestate,next_move,node)
                node.add_child(next_move,child)
                node.update_wins(-child.value)

        mv = root.select_move()

        move = self.id_to_move(gamestate.turn,mv)

        if not (self.collector is None):
            visit_count = np.array([root.visit_count(i) for i in range(self.size*self.size)]).astype('float32')
            visit_count = visit_count/(root.visits-1) #requires at least 2 visits adds to approx 1.. 
            np_board = self.encoder.encode_board(gamestate).astype('float32')
            self.collector.record_data(np_board,visit_count)
        #return root
        
        return move

    def create_node(self,gamestate,move=None,parent=None,noise = False):
        board_np = self.encoder.encode_board(gamestate).astype('float32')
        board_np.shape = (1,5,self.size,self.size)
        board_tensor = torch.from_numpy(board_np)
        board_tensor = board_tensor.to(self.device)
        self.net.eval()

        with torch.no_grad():
            value, priors = self.net(board_tensor)

        value = value.item()
        priors = priors.cpu().numpy()
        
        priors.shape = (self.size*self.size,)
        
        if noise:
            rnd = np.random.default_rng()
            ar = [0.03*10 for i in range(self.size*self.size)]
            s = rnd.dirichlet(ar)
            priors = 0.75*priors + 0.25*s
        new_node = Node(gamestate,parent,move,priors,value)

        return new_node

    

    def id_to_move(self,turn,mv_index):
        mv_x = (mv_index % self.size)+1
        mv_y = int(mv_index/self.size)+1
        return Move(turn,mv_x,mv_y)
    
class Node():
    def __init__(self,gamestate,parent,last_move,priors,value):
        self.gamestate = gamestate
        self.parent = parent
        self.last_move = last_move
        self.visits = 1
        self.value = value
        self.size = gamestate.board.size
        self.is_terminal = gamestate.is_over()

        self.branch_values = np.zeros((self.size*self.size),dtype=np.float32)
        self.branch_priors = priors
        self.branch_visits = np.zeros((self.size*self.size),dtype=np.float32)

        self.children = {}

        self.branches = set()

        for mv_index in range(self.size*self.size):    
            mv_x = mv_index%self.size + 1
            mv_y = mv_index//self.size+1
            move = Move(gamestate.turn,mv_x,mv_y)
            if(gamestate.is_legal_move(move)):
                self.branches.add(mv_index)

    def get_total_value(self):
        return self.values

    def set_total_value(self,values):
        self.values = values

    def get_visit_count(self):
        return self.branch_visits

    def set_visit_count(self,visits):
        self.branch_visits = visits

    def select_branch(self):
        moves = np.argsort(self.score_branch())[::-1]
        for move in moves:
            if move in self.branches:
                return move

    def select_move(self):
        return np.argmax(self.branch_visits)

    def add_child(self,move,child):
        self.children[move] = child
        self.visits += 1
        self.branch_visits[move] += 1
        self.branch_values[move] -= child.value

    def has_child(self,move):
        return move in self.children

    def get_child(self,move):
        return self.children[move]
            
    def score_branch(self):
        return self.branch_Q() + self.branch_U()

    def branch_Q(self):
        return np.divide(self.branch_values, self.branch_visits,
        out=np.zeros_like(self.branch_values), where=self.branch_visits!=0)

    def branch_U(self):
        c=4.0
        return c*self.branch_priors*np.sqrt(self.visits)/(self.branch_visits+1)

    def update_wins(self,value):
        if not self.parent is None:
            parent = self.parent
            parent.visits = parent.visits + 1
            parent.branch_visits[self.last_move] += 1
            parent.branch_values[self.last_move] += value
            parent.update_wins(-value)



