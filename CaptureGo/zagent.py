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
        self.conv0 = nn.Conv2d(5,self.c,5,padding='same',bias=False)

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

    def select_move(self,gamestate,printy=False):
        root = self.create_node(gamestate,noise=self.root_noise)
        for i in range(self.playouts):
            node = root
            #walk down to leaf node
            next_move = self.select_branch(node)
            while node.has_child(next_move):

                node = node.get_child(next_move)
                if(node.is_terminal):
                    break
                else:
                    next_move = self.select_branch(node)

            if(node.is_terminal):
                node.update_wins(1)
            else:
                next_move_X = self.id_to_move(node.gamestate.turn,next_move)
                new_gamestate = copy.deepcopy(node.gamestate)
                new_gamestate.move(next_move_X)
                child = self.create_node(new_gamestate,next_move,node,printy=printy)
                node.add_child(next_move,child)
                node.branches[next_move].visits += 1
                node.update_wins(-child.value)

        mv = max(root.moves(),key=root.visit_count)
        if printy:
            for branch in root.branches:
                print("Branch index:",branch,"Prior",root.branches[branch].prior)
                print("Branch visits",root.branches[branch].visits,"Q value:",root.expected_value(branch))
        
        move = self.id_to_move(gamestate.turn,mv)
        if not (self.collector is None):
            visit_count = np.array([root.visit_count(i) for i in range(self.size*self.size)]).astype('float32')
            visit_count = visit_count/(root.visits-1) #requires at least 2 visits adds to approx 1.. 
            np_board = self.encoder.encode_board(gamestate).astype('float32')
            self.collector.record_data(np_board,visit_count)
        #return root
        
        return move

    def select_branch(self,node):
        return max(node.moves(), key=node.score_branch)


    def create_node(self,gamestate,move=None,parent=None,noise = False,printy=False):
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
        if printy:
            print("Value:",value)
            print(priors.reshape((self.size,self.size)))
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

        self.branches = {}

        for mv_index, prior in enumerate(priors):    
            mv_x = (mv_index % self.size)+1
            mv_y = mv_index//self.size+1
            move = Move(gamestate.turn,mv_x,mv_y)
            if(gamestate.is_legal_move(move)):
                self.branches[mv_index] = Branch(prior)

        self.children = {}

    def moves(self):
        return self.branches.keys()

    def add_child(self,move,child):
        self.children[move] = child
        self.branches[move].total_value -= child.value
        self.visits += 1

    def has_child(self,move):
        return move in self.children

    def get_child(self,move):
        return self.children[move]

    def prior(self,move):
        return self.branches[move].prior

    def expected_value(self,move):
        branch = self.branches[move]
        if branch.visits == 0:
            return 0.0
        else:
            return branch.total_value/branch.visits

    def visit_count(self,move):
        if move in self.branches:
            return self.branches[move].visits
        else:
            return 0
            
    def score_branch(self,move):
        c=4.0
        q = self.expected_value(move)
        p = self.prior(move)
        n = self.visit_count(move)
        return q + c*p*np.sqrt(self.visits)/(n+1)

    def update_wins(self,value):
        if not self.parent is None:
            parent = self.parent
            parent.visits = parent.visits + 1
            (parent.branches[self.last_move]).visits += 1
            (parent.branches[self.last_move]).total_value += value
            parent.update_wins(-value)



