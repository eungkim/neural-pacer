import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np

from models import ResNet32, Pacer
from dataset import build_dataset


# args
parser = argparse.ArgumentParser(description='Pytorch Implementation of Neural Pacer Training')
parser.add_argument('--name_dataset', default='cifar10', type=str)
parser.add_argument('--model', default="resnet", type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--dataset', default="cifar10", type=str)
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--lr', default=1e-3, type=int)

args = parser.parse_args()

# torch settings
torch.manual_seed(816)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# modification of https://github.com/xjtushujun/meta-weight-net

# build model
def build_model():
    model = ResNet32(args.dataset=='cifar10' and 10 or 100)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
    
    return model

# train
def train(train_loader, train_meta_loader, model, optim_model, pacer, optim_pacer, epochs, lr, device):
    for epoch in range(epochs):
        train_loss = 0
        meta_loss = 0
        for i, ((inputs, targets), (inputs_meta, targets_meta)) in enumerate(zip(train_loader, train_meta_loader)):
                # settings
                model.train()
                inputs = inputs.to(device)
                targets = targets.to(device)
                inputs_meta = inputs_meta.to(device)
                targets_meta = targets_meta.to(device)

                model_meta = build_model().cuda()
                model_meta.load_state_dict(model.state_dict())

                # first training of model_meta
                outputs, reps = model_meta(inputs)

                cost = F.cross_entropy(outputs, targets, reduce=False)
                v_lambda = pacer(reps.detach()) # gradient should not flow to model_meta by reps
                loss_meta_0 = torch.sum(cost * v_lambda) / len(cost)
                model_meta.zero_grad()
                grads = torch.autograd.grad(loss_meta_0, (model_meta.params()), create_graph=True)
                model_meta.update_params(lr_inner=lr, source_params=grads)
                del grads

                # training of pacer
                outputs_meta = model_meta(inputs_meta)            
                loss_meta_1 = F.cross_entropy(outputs_meta, targets_meta)
                
                optim_pacer.zero_grad()
                loss_meta_1.backward()
                optim_pacer.step()

                # training of model
                outputs, reps = model(inputs)
                cost_model = F.cross_entropy(outputs, targets, reduce=False)

                with torch.no_grad():
                    v_new = pacer(reps)
                
                loss = torch.sum(cost_model * v_new) / len(cost_model)

                optim_model.zero_grad()
                loss.backward()
                optim_model.step()


                # print loss
                train_loss += loss.item()
                meta_loss += loss_meta_1.item()

                if (i+1)%50==0:
                    print(f"Epoch: [{epoch}/{epochs}]\t Iters: [{i}]\t Loss: [{(train_loss/(i+1))}]\t MetaLoss: [{(meta_loss/(i+1))}]")

                    train_loss = 0
                    meta_loss = 0

def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            test_loss+=F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct+=predicted.eq(targets).sum().item()
        
        test_loss/=len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        print(f"Test Avg Loss: {test_loss} Accuracy: {accuracy}")

    return accuracy

train_loader, train_meta_loader, test_loader = build_dataset(args.name_dataset)
model = build_model()
pacer = Pacer(64, 64, 1).cuda() #to edit - representation size

optim_model = torch.optim.SGD(model.params(), args.lr, momentum=0.9, weight_decay=1e-4)
optim_pacer = torch.optim.Adam(pacer.params(), args.lr, weight_decay=1e-4)

def main():
    best_acc = 0
    train(train_loader, train_meta_loader, model, optim_model, pacer, optim_pacer, args.epochs, args.lr, device)
    test_acc = test(model=model, test_loader=test_loader)
    if test_acc>=best_acc:
        best_acc = test_acc
    
    print(f"test accuracy: {best_acc}")


if __name__=="__main__":
    main()