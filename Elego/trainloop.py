'''
implements the main trainloop consisting of:
i) generate game records by self play
ii) train the network with data
iii) evaluate strength compared to previous network
'''
import training
import selfplay
import time
from network import ZNet
import torch
import transform



def train():
    start_index = 38
    transforms = [transform.rot90,transform.rot180,transform.rot270,
    transform.flip,transform.fliprt90,transform.fliprt180,transform.fliprt270]
    for i in range(1):
        print("iteration:",i)
        index = start_index + i
        selfplay.self_play(fileindex=index)
        
        training.fit_stuff(size=9,fileindex=index)
        for tran in transforms:
            training.fit_stuff_wtr(size=9,fileindex=index,transformation=tran)
        selfplay.cf_nets(fileindex1=index,fileindex2=index+1)

train()



