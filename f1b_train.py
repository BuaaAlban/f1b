import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import *
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow import keras

import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import torchtext.vocab as vocab

Dic ={'NEG':0,
        'POS':1,
        'NEU':2}
    
glove = vocab.GloVe(name='42B', dim=300, cache='./')

def get_data(path):
    with open(path,'r') as f:
        x = []
        y = [] 
        lengths = []
        target_index = []
        text = f.readlines()
        for line in text:
            xx = line.split('#### #### ####')[0]
            yy = eval(line.split('#### #### ####')[1])
            length = len(xx)
            label =Dic[yy[0][2]]
            index =yy[0][1][0]
            if label!=2:
                try:
                    x.append([glove.stoi[i.lower()] for i in xx.split()])
                    y.append(Dic[yy[0][2]])
                    target_index.append(index)
                    lengths.append(length)
                except:
                    continue
    return x, y, lengths, target_index

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(X_train, y_train, train_lengths, train_target)= get_data('train.txt')
(X_test, y_test, test_lengths, test_target) = get_data('test.txt')
import pdb
#pdb.set_trace()


max_review_length = 400  #train max =367
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length) 
X_test  = sequence.pad_sequences(X_test, maxlen=max_review_length)

#train_data = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
#test_data  = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
test_data  = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)
observe_loader  = DataLoader(train_data, batch_size=1, shuffle=False)

class Model(nn.Module):
    def __init__(self, emb_size, hid_size, typernn):
        super(Model, self).__init__()
        self.glove = vocab.GloVe(name='42B', dim=300, cache='./')
        self.emb_size  = emb_size
        self.hid_size  = hid_size

        #self.rnn = nn.LSTM(self.emb_size, self.hid_size, batch_first=True) 
        if typernn=='gru':
            self.rnn = nn.GRU(self.emb_size, self.hid_size, batch_first=True) 
        elif typernn=='rnn':
            self.rnn = nn.RNN(self.emb_size, self.hid_size, batch_first=True) 

        self.fc = nn.Linear(self.hid_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.glove.vectors[x.to(torch.LongTensor())].to(DEVICE)
        x, _ = self.rnn(x)
        #import pdb
        #pdb.set_trace()
        fullstep = x
        fulloutput = self.fc(fullstep)
        fullout = self.sigmoid(fulloutput) #(B,T=100,1)

        x = x[:,-1,:]
        x = self.fc(x)
        out = self.sigmoid(x)
        return out.view(-1),fullout

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.BCELoss()
    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_, _ = model(x)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if(batch_idx + 1) % 10 == 0: # print loss
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), running_loss/10))
            running_loss = 0.0

def test(model, device, test_loader):
    model.eval()
    criterion = nn.BCELoss()
    test_loss = 0.0 
    acc = 0 
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            y_,_ = model(x)
        test_loss += criterion(y_, y)
        acc += (y.cpu().numpy().astype(int) == (y_>0.5).cpu().numpy().astype(int)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, acc, len(test_loader.dataset),
        100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset)

def observe(model, device, test_loader, observe_lengths, observe_target):
    print('starting oberve>>>>>>>>>>>')
    raw_prob = []
    avg_data = []
    max_idx = 391
    model.eval()
    #import pdb
    #pdb.set_trace()
    for batch_idx, (x, y) in tqdm(enumerate(test_loader)):
        x, y = x.to(DEVICE), y.to(DEVICE)

        seq_len = observe_lengths[batch_idx]
        target_idx =  observe_target[batch_idx]
        target_idx = target_idx-seq_len+400 #for seq pad to 400
        #max_idx = max(target_idx, max_idx)  #77 for train set
        shift = max_idx - target_idx
        print(shift)
        with torch.no_grad():
            y_, fulloutput = model(x)
            if y==0:
                raw_prob.append(1-fulloutput.squeeze().cpu().numpy())
            else:
                raw_prob.append(fulloutput.squeeze().cpu().numpy())
            tmp =[-1 for i in range(shift)]
            tmp+=raw_prob[batch_idx].tolist()
            avg_data.append(tmp)
    print('max',max_idx)
    return avg_data

def plot(avg_data, typernn, dim):
    print('ploting,,,,,,')
    max_len = 0
    y = []
    for data in avg_data:
        max_len = max(max_len,len(data))
    x = [i for i in range(max_len)]
    for i in tqdm(range(max_len)):
        tmp = []
        for data in avg_data:
            if i<len(data) and data[i]!=-1:
                tmp.append(data[i])
        y.append(sum(tmp)/len(tmp))
    assert len(y)==max_len
    plt.plot(x, y, label='prob')
    plt.legend()  #显示上面的label
    plt.xlabel('time step index') #x_label
    plt.ylabel('prob')#y_label
    plt.title('target index is shifted to 397')
    plt.savefig('./'+typernn+str(dim)+'.png')

embedding_vector_length = 300
hidden_vector_length = 32
#typernn ='gru'
typernn ='rnn'
model = Model(embedding_vector_length, hidden_vector_length, typernn).to(DEVICE)
print(model)
optimizer = optim.Adam(model.parameters())

best_acc = 0.0 
PATH = './model.best.pth'


#avg_data = observe(model, DEVICE, observe_loader, train_lengths, train_target)
for epoch in range(1,15): 
    train(model, DEVICE, train_loader, optimizer, epoch)
    acc = test(model, DEVICE, test_loader)
    if best_acc < acc: 
        best_acc = acc 
        torch.save(model.state_dict(), PATH)
    print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))

avg_data = observe(model, DEVICE, observe_loader, train_lengths, train_target)
plot(avg_data, typernn, hidden_vector_length)
