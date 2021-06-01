import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.optim as optim
import random

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import matplotlib.pyplot as plt

import os
from os import walk


transformation = transforms.Compose([
        transforms.Resize([54,85]),
        transforms.ToTensor()
    ])
    
ampt_datasets = torchvision.datasets.ImageFolder(root= '/store/user/anderht3/ampt_visuals',transform = transformation)


train_size = 7990
test_size = 1998

trainsampt, testsampt = torch.utils.data.random_split(ampt_datasets, [train_size,test_size])


batchsize = 100

trainloader = torch.utils.data.DataLoader(trainsampt, batch_size=batchsize,
                                             shuffle=True)
testloader = torch.utils.data.DataLoader(testsampt, batch_size=len(testsampt),
                                             shuffle=True)


dataiter = iter(testloader)
images, labels = dataiter.next()

#print(images.shape)

#print(images[0].shape)
#print(labels[7].item())

batch_size = 512 # Try varying this. Its on the large side for minibatches.
data_loader_train = torch.utils.data.DataLoader(trainloader, batch_size=batch_size, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(testloader, batch_size=len(testloader), shuffle=False)

def imshow(img, title):
    
    plt.figure(figsize=(batchsize * 4, 4))
    plt.axis('off')
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.title(title)
    plt.show()
    
def show_batch_images(dataloader):
    images, labels = next(iter(dataloader))
    
    img = torchvision.utils.make_grid(images)
    imshow(img, title=[str(x.item()) for x in labels])
    
    return images, labels

images, labels = show_batch_images(trainloader)

class Base_Model(nn.Module):
    def __init__(self): 
        super(Base_Model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(13770, 12000),  # 28 x 28 = 784 pixels
            nn.BatchNorm1d(12000),
            nn.ReLU(),
            nn.Linear(12000, 8400),
            nn.BatchNorm1d(8400),
            nn.ReLU(),
            nn.Linear(8400, 7200),
            nn.BatchNorm1d(7200),
            nn.ReLU(),
            nn.Linear(7200, 4800),
            nn.BatchNorm1d(4800),
            nn.ReLU(),
            nn.Linear(4800, 2400),
            nn.BatchNorm1d(2400),
            nn.ReLU(),
            nn.Linear(2400, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(),
            nn.Linear(1200, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )
             
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
#         return x
        return x

model = Base_Model()
#print(model)

def train_batch(model, x, y, optimizer, loss_fn):
    #add back in optimizer
    # Run forward calculation
    y_predict = model.forward(x)

    # Compute loss.
    loss = loss_fn(y_predict, y)

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    return loss.data.item()

##### Iterate over epochs and all batches of data
def train(model, train_loader , test_loader , optimizer, loss_fn, epochs=200):
    #add back in optimizer
    losses = list()
    losses_test = list()
    accuracy = list()
    
    batch_index = 0
    for e in range(epochs):
        for x, y in train_loader:
            loss = train_batch(model=model, x=x, y=y, optimizer=optimizer, loss_fn=loss_fn)
            losses.append(loss)

            batch_index += 1
            
        for x , y in test_loader:
            y_predict = model.forward(x)
            losst = loss_fn(y_predict, y)
            
            losses_test.append(losst.data.item())

            accuracy.append(error(y,y_predict))
#         print("Epoch: ", e+1)
#         print("Batches: ", batch_index)

    return losses , losses_test, accuracy, epochs

def error(label,y):
  count = 0
  for i in range(len(y)):
    max = y[i][0]
    found = 0
    for j in range(len(y[i])):
      if(y[i][j] > max):
        max = y[i][j]
        found = j 

    if(label[i] == found):
      count +=1
  return (count/(len(y)))


def test_batch(model, x, y):
    # run forward calculation
    y_predict = model.forward(x)

    return y, y_predict

#####
def test(model, test_loader):
    y_vectors = list()
    y_predict_vectors = list()

    batch_index = 0
    for x, y in test_loader:
        y, y_predict = test_batch(model=model, x=x, y=y)

        y_vectors.append(y.data.numpy())
        y_predict_vectors.append(y_predict.data.numpy())

        batch_index += 1

    y_predict_vector = np.concatenate(y_predict_vectors)

    return y_predict_vector


def run(data_loader_train, data_loader_test , model_to_test , weight_decay):
    
    
    # Define the hyperparameters
    learning_rate = 1e-3
    model = model_to_test()
    
    # Initialize the optimizer with above parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate , weight_decay = weight_decay)

    # Define the loss function
    loss_fn =   torch.nn.CrossEntropyLoss() # mean squared error

    # Train and get the resulting loss per iteration
    loss , loss_test, accuracy, epoch = train(model=model, train_loader=data_loader_train , test_loader = data_loader_test, 
                 optimizer=optimizer, loss_fn=loss_fn)
    
    # Test and get the resulting predicted y values
    y_predict = test(model=model, test_loader=data_loader_test)

    return loss , loss_test , y_predict, accuracy, epoch


losses, loss_test, y_predict, accuracy, epoch = run(data_loader_train=trainloader, data_loader_test=testloader ,
                        model_to_test = Base_Model,  weight_decay = 0)



def plot_loss(losses_train,loss_test, show=True):
    fig = plt.gcf()
    fig.set_size_inches(8,6)
    ax = plt.axes()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    x_loss = list(range(len(losses)))
    plt.plot(x_loss, losses_train)
    

    a = list(range(len(loss_test)))
    const = round(len(losses) / len(loss_test))
    b = [(i+1)*const for i in a]
    
    plt.plot(b,loss_test)

    if show:
        plt.show()

    plt.savefig("Loss_large_set.png")
    plt.close()

def plot_accuracy(accuracy,epochs):
    fig = plt.gcf()
    fig.set_size_inches(8,6)
    ax = plt.axes()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    x_loss = list(range(epochs))
    plt.plot(x_loss, accuracy)

    plt.savefig("Accuracy_large_set.png")
    plt.close()

plot_accuracy(accuracy,epoch)
print("Final accuracy:", accuracy[-1:])


plot_loss(losses,loss_test)
print("Final loss:", loss_test[-1:])


