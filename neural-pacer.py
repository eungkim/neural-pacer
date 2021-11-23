import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.models as models
from torch.autograd import Variable
import numpy as np


#args
parser = argparse.ArgumentParser(description='Pytorch Implementation of Neural Pacer Training')
parser.add_argument('--model', default="resnet", type=str)
parser.add_argument('--batch_size', default=64, type=int)

args = parser.parse_args()

#torch settings
torch.manual_seed(816)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#train
def train(train_loader, train_meta_loader, model_name, optim_model, pacer, optim_pacer, epochs, lr, device):

    if model_name=="resnet32":
        model = models.resnet32().to(device) #To edit, output representation and final out value
        model_meta = models.resnet32()
    elif model_name=="efficientnetb0":
        model = models.efficientnet_b0()
        model_meta = models.efficientnet_b0()
    
    for epoch in range(epochs):
        train_loss = 0
        meta_loss = 0
        for i, ((inputs, targets), (inputs_meta, targets_meta)) in enumerate(zip(train_loader, train_meta_loader)):
                #settings
                model.train()
                inputs = inputs.to(device)
                targets = targets.to(device)
                inputs_meta = inputs_meta.to(device)
                targets_meta = targets_meta.to(device)
                model_meta.load_state_dict(model.state_dict())

                #first training of model_meta
                outputs, reps = model_meta(inputs)

                cost = F.cross_entropy(outputs, targets, reduce=False)
                v_lambda = pacer(reps)
                loss_meta_0 = torch.sum(cost * v_lambda) / len(cost)
                model_meta.zero_grad()
                grads = torch.autograd.grad(loss_meta_0, (model_meta.params()), create_graph=True)
                model_meta.update_params(lr_inner=lr, source_params=grads)
                del grads

                #training of pacer
                outputs_meta = model_meta(inputs_meta)            
                loss_meta_1 = F.cross_entropy(outputs_meta, targets_meta)
                
                optim_pacer.zero_grad()
                loss_meta_1.backward()
                optim_pacer.step()

                #training of model
                outputs, reps = model(inputs)
                cost_model = F.cross_entropy(outputs, targets, reduce=False)

                with torch.no_grad():
                    v_new = pacer(reps)
                
                loss = torch.sum(cost_model * v_new) / len(cost_model)

                optim_model.zero_grad()
                loss.backward()
                optim_model.step()


                #print loss
                train_loss += loss.item()
                meta_loss += loss_meta_1.item()

                if (i + 1)%50==0:
                    print(f"Epoch: [{epoch}/{epochs}]\t Iters: [{i}]\t Loss: [{(train_loss/(i+1))}]\t MetaLoss: [{(meta_loss/(i+1))}]")

                    train_loss = 0
                    meta_loss = 0