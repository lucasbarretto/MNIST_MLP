import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import random
import time

device = torch.device("cuda:0")
#device = torch.device("cpu")

transform = transforms.ToTensor()

# loads train set
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=1)

# loads test set
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                          shuffle=True, num_workers=2)

# defines the neural network
class Net(nn.Module):
    def __init__(self,i,n):
        super(Net, self).__init__()
        self.l1 = nn.Linear(i,n, bias=True)
        self.l2 = nn.Linear(n,n, bias=True)
        self.l3 = nn.Linear(n,n, bias=True)
        self.l4 = nn.Linear(n,n, bias=True)
        self.l5 = nn.Linear(n,c, bias=True)
        self.softmax = nn.LogSoftmax(dim=1) 
        
    def forward(self, X):
        y = F.relu(self.l1(X.view(-1, 28*28)))
        y = F.relu(self.l2(y))
        y = F.relu(self.l3(y))
        y = F.relu(self.l4(y))
        y = self.softmax(self.l5(y))
        
        return y
 
# defines the test accuracy function  
def test(testloader, model):
    model.eval()
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            test_pred = model(images)
            test_loss = criterion(test_pred, labels)
            test_hit = torch.argmax(test_pred, dim = 1)
    
    hits = 0
    
    for i in range(len(labels)):
        if test_hit[i] == labels[i]:
            hits += 1
    
    test_per = hits/len(labels)*100
    
    return test_per, test_loss

# shows prediction samples from the test set
def sample_pred(testloader, model):
    examples = enumerate(testloader)
    batch_idx, (example_data, example_targets) = next(examples)
    
    with torch.no_grad():
        output = model(example_data.to(device))
        
    fig = plt.figure()
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    fig
    

# defines the hyperparameters
i = 28*28 # number of features
n = 500 # number of neurons in hidden layers
c = 10 # number of classes
max_epochs = 100 # maximum number of epochs

# constructs the model
model = Net(i,n)
model.to(device)

# defines criterion
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

# trains the network
start = time.time()

train_loss_data = []
test_loss_data = []
test_per_data = []

for epoch in range(max_epochs):
    total_train_loss = 0
    n = 0
    for i, data in enumerate(trainloader,0):
        x_train, z_train = data[0].to(device), data[1].to(device)
        
        # zeroes the parameter gradients
        optimizer.zero_grad()
    
        # forward propagation
        train_pred = model(x_train)
        
        # computes the loss
        train_loss = criterion(train_pred, z_train)
        
        # backpropagation and updates the weights
        train_loss.backward()
        optimizer.step()
        
        total_train_loss += train_loss
        n +=1
        
    avg_train_loss = total_train_loss/n
    train_loss_data.append(avg_train_loss.item())
    
    # tests the model
    test_per, test_loss = test(testloader, model)
    test_loss_data.append(test_loss)
    test_per_data.append(test_per)
        
    # prints loss and accuracy rate values every 10 epochs
    #if epoch % 10 == 0:
    print('epoch %d - train_loss: %.4f - test_loss: %.4f - test_per: %.4f'% 
                 (epoch, train_loss.item(), test_loss.item(), test_per))

end = time.time()
trainingTime = end - start
    
fig, (ax1, ax2) = plt.subplots(2,sharex = True)
ax1.plot(train_loss_data)
ax2.plot(test_per_data, 'tab:red')

ax1.set(ylabel='training loss')
ax2.set(xlabel='number of epochs', ylabel='accuracy (%)')
#plt.savefig('net_performance.png')