import training
import time
from zagent import ZNet
import torch


def train():
    for i in range(2):
        print("Run",i)
        training.self_play(size=5)

        training.fit_stuff(size=5)
        
        training.test(size=5)

train()




    





