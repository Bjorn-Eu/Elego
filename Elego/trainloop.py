import training
import time
from network import ZNet
import torch


def train():
    start_index = 11
    for i in range(5):
        index = start_index + i
        training.self_play(size=5,fileindex=index)
        training.fit_stuff(size=5,fileindex=index)
        
        training.test(size=5,fileindex=index)

#train()
training.test(size=9,fileindex=0)





    





