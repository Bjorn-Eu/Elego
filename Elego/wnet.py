'''
Contains a few human crafted evaluation functions 
for benchmarking and debugging.
'''
import torch
import torch.nn as nn
import numpy as np
class WNET_PURE(nn.Module):
    def __init__(self,size=9):
        super().__init__()
        self.size = size
        self.scale = 1/(self.size*self.size)
    def forward(self,x):
        #batch_size = x.shape[0]
        policy = self.scale*torch.from_numpy(np.ones((self.size*self.size),dtype=np.float32))    

        value = np.array([0])
        #if(batch_size != 1):
        #    policy = np.tile(policy,8)
        #    value = np.tile(value,8)  
        value = torch.tensor(value)
        return value, policy

#Only adjucst policy with Q set to 0
class WNET_POL(nn.Module):
    def __init__(self,size=9):
        super().__init__()
        self.size = size
        self.scale = 1/(self.size*self.size)

    def forward(self,x):
        policy = self.scale*(np.ones((self.size,self.size),dtype=np.float32))
        if x.shape[0] != 0:
            np_x = x.numpy()
            np_x.shape = (5,self.size,self.size)
            #np_x = np.rot90(np_x,axes=(-2,-1))
            for i in range(1,self.size-1):
                for j in range(1,self.size-1):
                    val =  0
                    if np_x[2][j][i] == 1:
                        val = 1000
                    elif np_x[3][j][i] == 1:
                        val = 100
                    elif np_x[4][j][i] == 1:
                        val = 10
                    if val != 0:
                        policy[i+1][j] = val
                        policy[i-1][j] = val
                        policy[i][j+1] = val
                        policy[i][j-1] = val


        policy.shape = (self.size*self.size)
        
        policy = torch.from_numpy(policy)

        value = np.array([0])
        value = torch.tensor(value)
        return value, policy

#Only adjust Q to 1 if there is a capture
#and keeps policy set to uniform
class WNET_Q(nn.Module):
    def __init__(self,size=9):
        super().__init__()
        self.size = size
        self.scale = 1/(self.size*self.size)
    def forward(self,x):
        policy = self.scale*(np.ones((self.size*self.size),dtype=np.float32))        
        policy = torch.from_numpy(policy)

        value = 0
        if x.shape[0] != 0:
            np_x = x.numpy()
            np_x.shape = (5,self.size,self.size)
            for i in range(0,self.size):
                for j in range(0,self.size):
                    if np_x[2][j][i]==1 and np_x[1][j][i]==1:
                        value = 1
                        break

        value = np.array([value])
        value = torch.tensor(value)
        return value, policy

class WNET_ADV(nn.Module):
    def __init__(self,size=9):
        super().__init__()
        self.size = size
        self.scale = 1/(self.size*self.size)

    def forward(self,x):
        policy = self.scale*(np.ones((self.size,self.size),dtype=np.float32))
        value = 0
        if x.shape[0] != 0:
            np_x = x.numpy()
            np_x.shape = (5,self.size,self.size)
            
            for i in range(1,self.size-1):
                for j in range(1,self.size-1):
                    val =  0
                    if np_x[2][j][i] == 1:
                        val = 1000
                        if np_x[1][j][i]==1:
                            value = 1
                    elif np_x[3][j][i] == 1:
                        val = 100
                    elif np_x[4][j][i] == 1:
                        val = 10
                    if val != 0:
                        policy[i+1][j] = val
                        policy[i-1][j] = val
                        policy[i][j+1] = val
                        policy[i][j-1] = val

        policy.shape = (self.size*self.size)
        policy = torch.from_numpy(policy)

        value = np.array([value])
        value = torch.tensor(value)
        return value, policy


class WNet_RANDOM(nn.Module):
    def __init__(self,size=9):
        super().__init__()
        self.size = size
        self.scale = 1/(self.size*self.size)
    def forward(self,x):
        #batch_size = x.shape[0]
        #policy = self.scale*torch.from_numpy(np.ones((self.size*self.size),dtype=np.float32))
        policy = np.random.uniform(low=0.0, high=1.0, size=(self.size*self.size))
        #policy = np.arange(0,self.size*self.size,1)
        policy = policy/np.sum(policy)


        value = np.array([np.random.uniform(-1.0, 1.0)])
        #if(batch_size != 1):
        #    policy = np.tile(policy,8)
        #    value = np.tile(value,8)  

        policy = torch.from_numpy(policy)
        value = torch.tensor(value)
        return value, policy