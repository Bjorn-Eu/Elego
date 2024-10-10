import torch
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
from network import ZNet
from node import Node

class ZAgent(Agent):

    def __init__(self,net,size,encoder,root_noise=True,virtual_losses=False,playouts=150,device='cpu'):
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
        while root.visits <= self.playouts:
            #walk down to leaf node
            self.create_leaves(root,batch_size=8)
            
        mv = root.select_move()

        move = self.id_to_move(gamestate.turn,mv)

        if not (self.collector is None):
            visit_count = root.branch_visits
            visit_count = visit_count/(root.visits-1) #requires at least 2 visits adds to approx 1.. 
            np_board = self.encoder.encode_board(gamestate).astype('float32')
            self.collector.record_data(np_board,visit_count)
        return move

    #Returns None if there if an already expanded one is choosen
    def select_virtual_leaf(self,node):
        next_move = node.select_branch()
        while node.has_child(next_move):
            node = node.get_child(next_move)
            if(node.is_terminal or (node.branch_priors is None)):
                #print("Terminal or explored detected")
                break
            else:
                next_move = node.select_branch() #select new branch
        if node.is_terminal:
            #print("Found a terminal node, which we ignore and stop")
            node.update_wins(1)
            return None

        if node.branch_priors is None:
            return None

        next_move_X = self.id_to_move(node.gamestate.turn,next_move)
        new_gamestate = copy.deepcopy(node.gamestate)
        new_gamestate.move(next_move_X)
        new_node = Node(new_gamestate,node,next_move,None,1) #add virtual loss
        node.add_child(next_move,new_node)
        new_node.add_virtual_loss() #add virtual loss
        return new_node

    def create_leaves(self,node,batch_size=1):
        leaves = [None]*batch_size
        
        for i in range(batch_size):
            new_node = self.select_virtual_leaf(node)
            if(new_node == None):
                batch_size = i
                #print("Limiting batch size to", batch_size)
                break
            leaves[i] = new_node
            

        new_gamestates = np.empty((batch_size,5,self.size,self.size),dtype=np.float32)
        for i in range(batch_size):
            np_board = self.encoder.encode_board(leaves[i].gamestate)
            new_gamestates[i] = np_board

        value_list, priors_list = self.ask_model(new_gamestates)

        #undo virtual losses and
        #update new nodes with real values
        for i in range(batch_size):
            leaves[i].undo_virtual_losses()
            leaves[i].branch_priors = priors_list[i]
            leaves[i].update_wins(value_list[i])
        return leaves

    def ask_model(self,new_gamestates):
        self.net.eval()
        board_tensors = torch.from_numpy(new_gamestates)
        with torch.no_grad():
            value_list, priors_list = self.net(board_tensors)
        value_list = value_list.cpu().numpy()
        priors_list = priors_list.cpu().numpy()
        return value_list, priors_list

    def create_node(self,gamestate,move=None,parent=None,noise=False):
        board_np = self.encoder.encode_board(gamestate)
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
    




