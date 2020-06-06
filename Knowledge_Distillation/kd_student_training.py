import os
import time
import copy
import torch
import logging
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


class Net(nn.Module):
    
    def __init__(self, num_channels, dropout):
        
        super(Net, self).__init__()
        
        self.num_channels = num_channels
        self.dropout = dropout
        
        self.conv1 = nn.Conv2d(3, self.num_channels , 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(4*4*self.num_channels*4, self.num_channels*4)
        self.f1bn = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 10)

    def forward(self, x):
        x = F.relu(self.pool(self.bn1(self.conv1(x))))
        x = F.relu(self.pool(self.bn2(self.conv2(x))))
        x = F.relu(self.pool(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 4*4*self.num_channels*4)
        #print(x.shape)
        x = F.dropout(F.relu(self.f1bn(self.fc1(x))), p = self.dropout)
        x = self.fc2(x)
        
        return x

def get_teacher_outputs(teacher_model, input_batch):

    teacher_model.eval()  
    t_out = []
    
    with torch.no_grad():
        inputs = input_batch.to(device) 
        outputs = model(inputs)
    
    t_out.append(outputs.data.cpu().numpy())
    
    return t_out

#def tLoss(self, teacher_outputs, outputs, labels):
def get_teacher_outputs(teacher_model, dataloader):

    teacher_model.eval()  
    teacher_logits = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            tinputs = nn.Upsample(size=224, mode='bilinear', align_corners=True)(inputs).to(device)
            toutputs = teacher_model(tinputs)
            teacher_logits.append(toutputs)
            
    return teacher_logits

def tLoss(teacher_outputs, outputs, labels, T, alpha, tflag=False):
    if tflag:
        KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
        return KD_loss
    else:
        return F.cross_entropy(outputs, labels)
    
def train_model(model, tmodel, tlogits, dataloaders, optimizer, device, T, alpha, num_epochs=25, KD=True):
    since = time.time()
    
    tmodel = tmodel.to(device)
    tmodel.eval()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = -1

    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        log_dict = {}
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            j = 1
            num_minibatches = len(dataloaders[phase])
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                tout = tlogits[i].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if phase == 'train':
                        loss = tLoss(tout, outputs, labels, T, alpha, KD)
                    else:
                        loss = 0.0
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        #scheduler.step()
                        optimizer.step()

                # statistics
                if phase == 'train':
                    running_loss += loss.item() * inputs.size(0)
                else:
                    running_loss = -1.0
                running_corrects += torch.sum(preds == labels.data)
                
                if j%100 == 0:
                    print('{} batch {}/{}'.format(phase, j, num_minibatches))
                j += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            log_dict[phase] = '{:.4f} {:.4f}'.format(epoch_loss, epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
        
        f = open('models/log_{}_{}.txt'.format(alpha,T), 'a+')
        f.write('{} {} {}\n'.format(epoch, log_dict['train'], log_dict['val']))
        f.close()

        print()
    #f.close()
    time_elapsed = time.time() - since
    fstring = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    print(fstring)
    print('Best val Acc: {:4f}'.format(best_acc))
    #logging.info(fstring)
    # load best model weights
    model.load_state_dict(best_model_wts)
    fout_name = "models/cifar10_w_Resnet18_KD_{}_{}_{}.pt".format(T,alpha,best_epoch)
    torch.save(model, fout_name)
    #logging.info("Model with best weights saved at: " + fout_name)
    return model, val_acc_history

def get_teacher(device):
	teacher_model = torchvision.models.resnet18(False)
	num_ftrs = teacher_model.fc.in_features
	teacher_model.fc = nn.Linear(num_ftrs, 10)
	teacher_model.load_state_dict(torch.load("models/resnet18_transferred_01011101.p"))
	teacher_model = teacher_model.to(device)
	return teacher_model

def main():
	train_transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))])
     
	val_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))])

	trainset = torchvision.datasets.CIFAR10(root='data', train=True,
                                        download=False, transform=train_transform)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=4)

	valset = torchvision.datasets.CIFAR10(root='data', train=False,
                                       download=False, transform=val_transform)
	val_loader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=4)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	data_loaders = {'train' : train_loader, 'val' : val_loader}

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	##load teacher model
	teacher_model = get_teacher(device)
	## Get teacher logits
	teacher_logits = get_teacher_outputs(teacher_model, data_loaders['train'])

	## run grid search on student
	alphas = [0.1,0.25,0.5,0.8,0.95]
	Ts = [1.0,2.0,4.0,8.0,10.0]
	for alpha in alphas:
    	for T in Ts:
        	net = Net(32, 0.5)
        	net = net.to(device)
        	optimizer = optim.Adam(net.parameters(), lr=1e-3)
        	train_model(net, teacher_model, teacher_logits, data_loaders, optimizer, device, T, alpha, num_epochs=100)

if __name__ == '__main__':

	random.seed(42)
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)
	np.random.seed(42)

	main()