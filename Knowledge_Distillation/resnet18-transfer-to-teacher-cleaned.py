import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import SubsetRandomSampler
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import copy


model_resnet18 = models.resnet18(pretrained=True)


num_classes = 10
model_resnet18.fc = nn.Linear(model_resnet18.fc.in_features, num_classes)


nn.init.normal_(model_resnet18.fc.weight, mean=0.0, std=1.0/np.sqrt(512))
nn.init.normal_(model_resnet18.fc.bias, mean=0.0, std=1.0/np.sqrt(512))



trn_set = torchvision.datasets.CIFAR10(root='./../assignment-1/datasets/', train=True,
                                        download=True, transform=transforms.ToTensor())
trn_loader = torch.utils.data.DataLoader(trn_set,  batch_size=50000, shuffle=False, num_workers=4)

tst_set = torchvision.datasets.CIFAR10(root='./../assignment-1/datasets/', train=False,
                                       download=True, transform=transforms.ToTensor())
tst_loader = torch.utils.data.DataLoader(tst_set, batch_size=10000, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



data_iter = iter(trn_loader)
tr_images, tr_labels = data_iter.next()
tr_means = tr_images.mean(dim=(0,2,3))
tr_stds = tr_images.std(dim=(0,2,3))


tr_means, tr_stds


train_transform = transforms.Compose(
     [transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))])

val_transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))])

trn_set = torchvision.datasets.CIFAR10(root='./../assignment-1/datasets/', train=True,
                                        download=False, transform=train_transform)
trn_loader = torch.utils.data.DataLoader(trn_set, batch_size=100, shuffle=True, num_workers=4)

tst_set = torchvision.datasets.CIFAR10(root='./../assignment-1/datasets/', train=False,
                                       download=False, transform=val_transform)
tst_loader = torch.utils.data.DataLoader(tst_set, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

data_loaders = {'trn': trn_loader, 'val': tst_loader}


train_transform = transforms.Compose(
    [transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))])
     
val_transform = transforms.Compose(
    [transforms.RandomResizedCrop(224),
     transforms.ToTensor(),
     transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))])

trainset = torchvision.datasets.CIFAR10(root='./../assignment-1/datasets/', train=True,
                                        download=False, transform=train_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)

valset = torchvision.datasets.CIFAR10(root='./../assignment-1/datasets/', train=False,
                                       download=False, transform=val_transform)
val_loader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

data_loaders = {'trn' : train_loader, 'val' : val_loader}


if torch.cuda.is_available():
    device = torch.device("cuda", index=0)
else:
    device = torch.device("cpu")



model_resnet18 = model_resnet18.to(device)
parameters_to_update = []
for name, parameter in model_resnet18.named_parameters():
    if parameter.requires_grad == True:
        parameters_to_update.append(parameter)
        


def cyclical_lr(step_size=800, min_lr=1e-6, max_lr=1e-1):

    lr_lambda = lambda itr: min_lr + (max_lr - min_lr) * step(itr, step_size)

    def step(itr, step_size):
        if itr%(2*step_size) < step_size:
            return (itr%(2*step_size)) / step_size
        return (2*step_size-(itr%(2*step_size)))/step_size
        
    return lr_lambda


# In[ ]:


loss_function = nn.CrossEntropyLoss()

clr = cyclical_lr(500, 1e-5, 1e-2)

optimizer = optim.Adagrad(parameters_to_update, lr=1.0)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])


def train_model(model, data_loaders, loss_function, optimizer, num_epochs, device):
    
    start = time.time()
    val_accuracies = list()
    max_val_accuracy = 0.0

    optimal_model_parameters = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        for phase in ['trn', 'val']:
            if phase == 'trn':
                model.train()
            else:
                model.eval()

            accum_loss = 0.0
            accum_correct = 0

            i = 1
            num_minibatches = len(data_loaders[phase])
            for X, y_targ in data_loaders[phase]:
                X = X.to(device)
                y_targ = y_targ.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='trn'):
                    Y_pred_as_one_hot = model(X)
                    loss = loss_function(Y_pred_as_one_hot, y_targ)
                    max_probs, y_pred = torch.max(Y_pred_as_one_hot, dim=1)

                    if phase == 'trn':
                        loss.backward()
                        scheduler.step()
                        optimizer.step()

                accum_loss += loss.item() * X.size(0)
                accum_correct += torch.sum(y_pred==y_targ.data)
                
                if i%50 == 0:
                    print('done {} batch {}/{}'.format(phase, i, num_minibatches))
                i += 1

            epoch_loss = accum_loss / len(data_loaders[phase].dataset)
            epoch_accuracy = accum_correct.double() / len(data_loaders[phase].dataset)

            print('{} loss: {:.4f}, accuracy: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

            if phase == 'val':
                val_accuracies.append(epoch_accuracy)
                if epoch_accuracy > max_val_accuracy:
                    max_val_accuracy = epoch_accuracy
                    optimal_model_parameters = copy.deepcopy(model.state_dict())
                if (epoch+1)%10 == 0:
                    torch.save(model.state_dict(), 'models/resnet18_{}_{}.p'.format(epoch, epoch_accuracy))
                    print('saved models/resnet18_{}_{}.p'.format(epoch, epoch_accuracy))
        print()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Optimal validation accuracy: {:4f}'.format(max_val_accuracy))

    model.load_state_dict(optimal_model_parameters)
    torch.save(model.state_dict(), 'models/resnet18_optimal_{}.p'.format(val_accuracies[-1]))
    
    return model, val_accuracies


num_epochs = 3

model_resnet18_trained, val_accuracies = train_model(
    model_resnet18, data_loaders, loss_function, optimizer, num_epochs, device)

