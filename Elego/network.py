import torch
import torch.nn as nn

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
        self.n = 4

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