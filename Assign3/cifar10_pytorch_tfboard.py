#CS535 Deep Learning
#Assignment3
#Name: YuWenTseng
#ID: 933652910

from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#import tensorflow as tf
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter 
# from torch.utils.tensorboard import SummaryWriter # for pytorch above or equal 1.4

class Net(nn.Module):
######Q1######
    #def __init__(self):
        #super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        #self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        #self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        #self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.fc1 = nn.Linear(64 * 8 * 8, 512)
        #self.batchNormal = nn.BatchNorm1d(512) #Q1 Add a batch normalization layer after the first fully-connected layer(fc1) 
        #self.add_Fc = nn.Linear(512,512) #Q2 adding another fully connected layer with 512 nodes
        #self.fc2 = nn.Linear(512, 10)

    #def forward(self, x):
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = self.pool(x)
        #x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        #x = self.pool(x)
        #x = x.view(-1, self.num_flat_features(x))
        #x = F.relu(self.batchNormal(self.fc1(x))) #Q1 
        #x = F.relu(self.add_Fc(x)) #Q2
        #x = self.fc2(x)
        #return x

#####Q4 - 1#######
    #def __init__(self):
        #super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        #self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.fc1 = nn.Linear(32 * 16 * 16, 512)
        #self.batchNormal = nn.BatchNorm1d(512) #Q1 Add a batch normalization layer after the first fully-connected layer(fc1) 
        #self.add_Fc = nn.Linear(512,512) #Q2 adding another fully connected layer with 512 nodes
        #self.fc2 = nn.Linear(512, 10)

    #def forward(self, x):
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = self.pool(x)
        #x = x.view(-1, self.num_flat_features(x))
        #x = F.relu(self.fc1(x))
        #x = self.batchNormal(x) #Q1 
        #x = F.relu(self.add_Fc(x)) #Q2
        #x = self.batchNormal(x)
        #x = self.fc2(x)
        #return x
###################

###########Q4-2##########
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.batchNormal = nn.BatchNorm1d(512) #Q1 Add a batch normalization layer after the first fully-connected layer(fc1) 
        self.add_Fc = nn.Linear(512,512) #Q2 adding another fully connected layer with 512 nodes
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.batchNormal(x) #Q1 
        x = F.relu(self.add_Fc(x)) #Q2
        x = self.batchNormal(x)
        x = self.fc2(x)
        return x
########################
	
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.eval() # Why would I do this?
    criterion = nn.CrossEntropyLoss(reduction='sum')
    #criterion = nn.CrossEntropyLoss(size_average=False)
    for data in dataloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    net.train() # Why would I do this?
    return total_loss / total, correct.item() / total

if __name__ == "__main__":
    BATCH_SIZE = 32 #mini_batch size
    MAX_EPOCH = 10 #maximum epoch to train

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Building model...')
    net = Net().cuda()

    #############Q2###########
    #pretrained_dict = torch.load('mytraining.pth')
    #net_dict = net.state_dict()
    #pretrained_dict = {i: v for i, v in pretrained_dict.items() if i in net_dict}
    #net_dict.update(pretrained_dict) 
    #net.load_state_dict(net_dict)
    ######################################
    
    net.train() # Why would I do this?

    writer = SummaryWriter(log_dir='./log')
    
    #writer = SummaryWriter()
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    #optimizer = optim.Adam(net.parameters(), lr=0.0001)
    #optimizer = optim.Adagrad(net.parameters(), lr=0.001) 
    #optimizer = optim.RMSprop(net.parameters(), lr = 0.001)

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader)
        test_loss, test_acc = eval_net(testloader)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))

        #tf.summary.scalar('train_loss', train_loss)	
        #tf.summary.scalar('test_loss', test_loss)
        #writer=tf.summary.FileWriter('/path/to/logs', tf.get_default_graph())
        #writer.add_scalar('train_loss', train_loss, epoch)
        #writer.add_scalar('loss', test_loss,train_loss, epoch+1)
	   #writer.add_scalars('Loss', {'train_loss':train_loss, 'test_loss':test_loss}, epoch+1)

        writer.add_scalars('Loss', {'train_loss':train_loss, 'test_loss':test_loss}, epoch+1)
        writer.add_scalars('Accuracy', {'train_acc':train_acc, 'test_acc':test_acc}, epoch+1)

    writer.close()
    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), 'mytraining.pth')
