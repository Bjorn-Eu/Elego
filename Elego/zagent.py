import torch
import numpy as np
from gameplay.board import Board
from gameplay.gamestate import GameState
from gameplay.move import Move
from agent import Agent
from random_agent import RandomAgent
import copy
from zexperiencecollector import ZExperienceCollector
from zexperiencecollector import ZExperienceData
from network import ZNet
from node import Node

class ZAgent(Agent):

    def __init__(self,net,size,encoder,root_noise=True,playouts=150,device='cpu',batch_size=8):
        self.net = net
        self.size = size
        self.encoder = encoder
        self.collector = None
        self.root_noise = root_noise
        self.playouts = playouts
        self.device = device
        self.batch_size = batch_size

    def set_playouts(self,playouts):
        self.playouts = playouts
        
    def set_collector(self,collector):
        self.collector = collector

    def select_move(self,gamestate,printout=False):
        root = self.create_node(gamestate,noise=self.root_noise)
        while root.visits <= self.playouts:
            #walk down to leaf node
            self.create_leaves(root,batch_size=self.batch_size)
            if printout and root.visits%200==0:
                np.set_printoptions(precision=2,suppress=True)
                print("Priors:")
                print(root.branch_priors.reshape(self.size,self.size))
                print(root.branch_visits.reshape(self.size,self.size))
                print("Q Value")
                print(root.branch_Q().reshape(self.size,self.size))
                print("Utility:")
                print(root.branch_U().reshape(self.size,self.size))
            
        mv = root.select_move()

        move = self.encoder.decode_move(mv,gamestate.turn)

        if not (self.collector is None):
            visit_count = root.branch_visits
            visit_count = visit_count#/(root.visits-1) #requires at least 2 visits adds to approx 1.. 
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

        #Found a node we have already explored in this batch
        #so we end the batch here
        if node.branch_priors is None:
            return None

        next_move_X = self.encoder.decode_move(next_move,node.gamestate.turn)
        new_gamestate = copy.deepcopy(node.gamestate)
        new_gamestate.move(next_move_X)
        new_node = Node(new_gamestate,node,next_move,None) 
        node.add_child(next_move,new_node)
        if new_node.is_terminal:
            new_node.update_wins(1)
            return None
        else:
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
            leaves[i].update_wins(-value_list[i])
        return leaves

    def ask_model(self,new_gamestates):
        self.net.eval()
        board_tensors = torch.from_numpy(new_gamestates)
        with torch.no_grad():
            value_list, priors_list = self.net(board_tensors)
        priors_list = torch.softmax(priors_list,dim=-1)
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

        priors = torch.softmax(priors,dim=-1)

        value = value.item()
        
        priors = priors.cpu().numpy()
        
        priors.shape = (self.size*self.size,)
        
        if noise:
            rnd = np.random.default_rng()
            ar = [0.5 for i in range(self.size*self.size)]
            s = rnd.dirichlet(ar)
            priors = 0.75*priors + 0.25*s
        new_node = Node(gamestate,parent,move,priors,value)

        return new_node

    




