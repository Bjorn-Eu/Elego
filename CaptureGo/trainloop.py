import training
import time
from zagent import ZNet
import torch


def train():
    start_index = 27
    for i in range(0,28):
        index = start_index + i
        training.self_play(size=5,fileindex=index)

        training.fit_stuff(size=5,fileindex=index)
        
        training.test(size=5,fileindex=index)

train()





    





