import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from network import ZNet
from zagent import ZAgent
from random_agent import RandomAgent
from zexperiencecollector import ZExperienceCollector
from zexperiencecollector import ZExperienceData
import transform


def train_loop(net,dataloader,value_loss_fn,policy_loss_fn,optimizer):
    net.train()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0
    total_value_loss = 0
    total_policy_loss = 0
    for train_board, train_policy_update, train_adj_reward in dataloader:
        value, policy = net(train_board)
        loss_value = 0.5*value_loss_fn(value,train_adj_reward)
        loss_policy = nn.CrossEntropyLoss()(policy,train_policy_update.softmax(dim=1))
        loss =loss_policy+loss_value

        total_value_loss += loss_value.item()
        total_policy_loss += loss_policy.item()
        total_loss += loss.item()

    
        #backpropagate
        loss.backward()

        optimizer.step()

        
        optimizer.zero_grad()
    print(f"The average loss was: {total_loss/num_batches:.4f}")
    print(f"The average value loss was: {total_value_loss/num_batches:.4f}")
    print(f"The average policy loss was: {total_policy_loss/num_batches:.4f}")

def fit_stuff(size=9,fileindex=0):
    znet = torch.jit.load(f'nets9x9\\znet{fileindex}.pt')

    value_loss_fn = nn.functional.mse_loss
    policy_loss_fn = nn.functional.binary_cross_entropy
    batch_size = 32 
    learning_rate = 1e-5
    optimizer = torch.optim.SGD(znet.parameters(), lr=learning_rate,momentum=0.0)


    datap1 = ZExperienceData()
    datap1.load(f'selfplaydata\\9x9b{fileindex}.npz')

    game_data = datap1.gamestates.astype('float32')
    game_counts = datap1.visit_counts.astype('float32')
    game_rewards = datap1.rewards.astype('float32')
    
    datap2 = ZExperienceData()
    datap2.load(f'selfplaydata\\9x9w{fileindex}.npz')

    game_data = np.append((datap1.gamestates).astype('float32'),(datap2.gamestates).astype('float32'),0) 
    game_counts = np.append(datap1.visit_counts,datap2.visit_counts,0).astype('float32')
    game_rewards = np.append(datap1.rewards,datap2.rewards).astype('float32')
    

    samples = game_data.shape[0]
    game_data.shape = (samples,5,size,size)
    game_counts.shape = (samples,size*size)
    game_rewards.shape = (samples,1)

    train_boards_tensor = torch.from_numpy(game_data)
    train_counts_tensor = torch.from_numpy(game_counts)
    train_rewards_tensor = torch.from_numpy(game_rewards)


    train_dataset = TensorDataset(train_boards_tensor,train_counts_tensor,train_rewards_tensor)
    train_dataloader = DataLoader(train_dataset,batch_size)

    for i in range(3):
        print("epoch", i)
        train_loop(znet,train_dataloader,value_loss_fn,policy_loss_fn,optimizer)

    model_scripted = torch.jit.script(znet) # Export to torchscript
    model_scripted.save(f'nets9x9\\znet{fileindex+1}.pt')

def fit_stuff_wtr(size=9,fileindex=0,transformation=None):

    znet = torch.jit.load(f'nets9x9\\znet{fileindex+1}.pt')

    value_loss_fn = nn.functional.mse_loss
    policy_loss_fn = nn.functional.binary_cross_entropy
    batch_size = 32
    learning_rate = 1e-5
    optimizer = torch.optim.SGD(znet.parameters(), lr=learning_rate,momentum=0.0)


    datap1 = ZExperienceData()
    datap1.load(f'selfplaydata\\9x9b{fileindex}.npz')

    game_data = datap1.gamestates.astype('float32')
    game_counts = datap1.visit_counts.astype('float32')
    game_rewards = datap1.rewards.astype('float32')
    
    datap2 = ZExperienceData()
    datap2.load(f'selfplaydata\\9x9w{fileindex}.npz')

    game_data = np.append((datap1.gamestates).astype('float32'),(datap2.gamestates).astype('float32'),0) 
    game_counts = np.append(datap1.visit_counts,datap2.visit_counts,0).astype('float32')
    game_rewards = np.append(datap1.rewards,datap2.rewards).astype('float32')
    

    samples = game_data.shape[0]
    game_data.shape = (samples,5,size,size)
    game_counts.shape = (samples,size*size)
    game_rewards.shape = (samples,1)

    if transformation != None:
        game_data = transformation(game_data).copy()
        game_counts.shape = (samples,size,size)
        game_counts = transformation(game_counts)
        game_counts = game_counts.reshape(samples,size*size).copy()

  

    train_boards_tensor = torch.from_numpy(game_data)
    train_counts_tensor = torch.from_numpy(game_counts)
    train_rewards_tensor = torch.from_numpy(game_rewards)


    train_dataset = TensorDataset(train_boards_tensor,train_counts_tensor,train_rewards_tensor)
    train_dataloader = DataLoader(train_dataset,batch_size,shuffle=True)

    for i in range(3):
        print("epoch", i)
        train_loop(znet,train_dataloader,value_loss_fn,policy_loss_fn,optimizer)

    model_scripted = torch.jit.script(znet) # Export to torchscript
    model_scripted.save(f'nets9x9\\znet{fileindex+1}.pt')


