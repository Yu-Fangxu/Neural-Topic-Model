from model import NVDM, ProdLDA
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn import svm
dataset = "R8"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50

class BoWDataset(Dataset):
    def __init__(self, dataset, labels):
        self.BoW = dataset
        self.labels = labels
        
        self.word_count = []
        for i in range(len(self.BoW)):
            self.word_count.append(torch.sum(self.BoW[i]))
                
    def __getitem__(self, index):
        return self.BoW[index], self.labels[index], self.word_count[index]

    def __len__(self):
        return len(self.BoW)
        
BoW = np.load(f"temp/{dataset}.BoW.npy")
BoW = torch.from_numpy(BoW)
BoW = BoW.to(torch.float32)
labels = np.load(f"temp/{dataset}.targets.npy")

BoW_train, BoW_test, labels_train, labels_test = train_test_split(BoW, labels, test_size=0.2, random_state=0)
print(BoW_train.shape)
train_dataset = BoWDataset(BoW_train, labels_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = BoWDataset(BoW_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

model = NVDM(BoW.shape[1])
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)

model.to(device)

def train():
    for epoch in range(epochs):
        ppx_sum = 0
        kld_sum = 0
        word_count = 0
        doc_count = 0
        loss_sum = 0
        for data, _, count_batch in train_loader:
            word_count += torch.sum(count_batch)
            
            data = data.cuda()
            
            sample, logits, kld, rec_loss = model(data)
            loss = kld + rec_loss
            
            loss_sum += torch.sum(loss)
            kld_sum += torch.mean(kld)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            count_batch = torch.add(count_batch, 1e-12)
            ppx_sum += torch.sum(torch.div(loss.cpu(), count_batch))
            doc_count += len(data)
        print_ppx = torch.exp(loss_sum / word_count)
        print_ppx_perdoc = torch.exp(ppx_sum / doc_count)
        print_kld = kld_sum / len(train_loader)
        print('| Epoch train: {:d} |'.format(epoch + 1),
              '| Perplexity: {:.9f}'.format(print_ppx),
              '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
              '| KLD: {:.5}'.format(print_kld),
              '| Loss: {:.5}'.format(loss_sum))

def test():
    ppx_sum = 0
    kld_sum = 0
    word_count = 0
    doc_count = 0
    loss_sum = 0
    for data, _, count_batch in test_loader:
        word_count += torch.sum(count_batch)
        
        data = data.cuda()
        
        sample, logits, kld, rec_loss = model(data)
        loss = kld + rec_loss
        
        loss_sum += torch.sum(loss)
        kld_sum += torch.mean(kld)
        # count_batch += 1e-12
        ppx_sum += torch.sum(torch.div(loss.cpu(), count_batch))
        doc_count += len(data)
    print_ppx = torch.exp(loss_sum / word_count)
    print_ppx_perdoc = torch.exp(ppx_sum / doc_count)
    print_kld = kld_sum / len(test_loader)
    print('| Epoch test:',
          '| Perplexity: {:.9f}'.format(print_ppx),
          '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
          '| KLD: {:.5}'.format(print_kld),
          '| Loss: {:.5}'.format(loss_sum))
          
def generate_train_repr():
        samples = []
        with torch.no_grad():
            for data, _, _ in train_loader:
                
                data = data.cuda()
                
                sample, logits, kld, rec_loss = model(data)
                
                samples.append(sample)
        
        train_repr = torch.cat(samples, dim=0).cpu().numpy()
        return train_repr
        
        
def generate_test_repr():
        samples = []
        with torch.no_grad():
            for data, _, _ in test_loader:
                
                data = data.cuda()
                
                sample, logits, kld, rec_loss = model(data)
                
                samples.append(sample)
        
        test_repr = torch.cat(samples, dim=0).cpu().numpy()
        return test_repr
if __name__ == "__main__":
    train()
    test()
    # generate topic distributions used to classification
    train_repr = generate_train_repr()
    test_repr = generate_test_repr()

    clf = svm.SVC(kernel='rbf')
    
    clf.fit(train_repr, labels_train)
    
    print("Accuracy on training set:", clf.score(train_repr, labels_train))
    print("Accuracy on testing set:", clf.score(test_repr, labels_test))