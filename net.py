import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0")

# loads test set
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
x_train = trainset.data.view(-1, 28*28).float()
z_train = trainset.targets

# loads train set
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
x_test = testset.data.view(-1, 28*28).float()
z_test = testset.targets

# defines the neural network
class Net(nn.Module):
    def __init__(self,i,n):
        super(Net, self).__init__()
        self.l1 = nn.Linear(i,n,bias=True)
        self.l2 = nn.Linear(n,n, bias=True)
        self.l3 = nn.Linear(n,n, bias=True)
        self.l4 = nn.Linear(n,n, bias=True)
        self.l5 = nn.Linear(n,c, bias=True)
        self.softmax = nn.LogSoftmax(dim=1) 
        
    def forward(self, X):
        y = F.relu(self.l1(X))
        y = F.relu(self.l2(y))
        y = F.relu(self.l3(y))
        y = F.relu(self.l4(y))
        y = self.softmax(self.l5(y))
        
        return y

# defines the test accuracy function  
def test(x_test, z_test, model):
    with torch.no_grad():
        test_pred = torch.argmax(model(x_test), dim = 1)
    
    hits = 0
    
    for i in range(len(z_test)):
        if test_pred[i] == z_test[i]:
            hits += 1
    
    return hits/len(z_test)*100

# shows prediction samples from the test set
def sample_pred(model, testset):
    fig = plt.figure(1)
    for i, img in enumerate(testset.data[7:17].to(device)):
        ax = fig.add_subplot(1,10,i+1)
        ax.set_axis_off()
        ax = plt.imshow(img.cpu())
        with torch.no_grad():
            a = model(img.view(-1, 28*28).float())
        
        print(testset.classes[torch.argmax(a, dim=1)])
    plt.show()
    

# defines the hyperparameters
i = 28*28 # number of features
n = 500 # number of neurons in hidden layers
c = 10 # number of classes
eps = 0.02 # minimum training error
max_epochs = 5000 # maximum number of epochs
min_change_rate = 0.001
max_no_change = 20

# constructs the model
model = Net(i,n)

# defines criterion
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# send the net object and inputs to the cuda device
model.to(device)
x_train, z_train = x_train.to(device), z_train.to(device)
x_test, z_test = x_test.to(device), z_test.to(device)

# trains the network
train_loss_data = []
test_per_data = []

last_train_loss = 0
no_change = 0

for epoch in range(max_epochs):
    
    # zeroes the parameter gradients
    optimizer.zero_grad()
    
    # forward propagation
    train_pred = model(x_train)
        
    # computes the loss
    train_loss = criterion(train_pred, z_train)
    #train_loss_data.append(train_loss)
        
    # backpropagation and updates the weights
    train_loss.backward()
    optimizer.step()
    
    # tests the model
    test_per = test(x_test, z_test, model)
    
    # prints loss and accuracy rate values every 10 epochs
    train_loss_data.append(train_loss.item())
    test_per_data.append(test_per)
    print('epoch %d - train_loss: %.3f - test_per: %.3f' % 
                 (epoch, train_loss.item(), test_per))
    
#    # stop conditions
#    if train_loss < eps:
#        break
#    
#    if abs(train_loss-last_train_loss) < min_change_rate:
#        no_change += 1
#        
#    if no_change > max_no_change:
#        break
#    
#    last_train_loss = train_loss
