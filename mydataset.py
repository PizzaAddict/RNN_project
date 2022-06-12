import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import torch
from vocab import Vocabulary
from util import MyCollate

class TextDataset(Dataset):
    #collate: dataloader가 해결
    def __init__ (self, data_dir, mode):
        self.labeldict = {'ICT':0, 'economy':1, 'education':2, 'mechanics':3}
        self.labeldict_reverse = {i:w for w,i in self.labeldict.items()}
        self.df_source = pd.read_csv(os.path.join(data_dir, mode + '.csv'))
        self.sentences = self.df_source['text'].values
        self.labels = self.df_source['label'].values
        self.vocab = Vocabulary("already",100)#vocab은 이미 만들어짐

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        numeric_sentence = self.vocab(sentence)
        numeric_label = [self.labeldict[label]]

        return torch.tensor(numeric_sentence), torch.tensor(numeric_label)

def make_data_loader(dataset, batch_size, batch_first=True, shuffle=True): #increase num_workers according to CPU
    #get pad_idx for collate fn
    pad_idx = dataset.vocab.wtoi['<PAD>']
    #define loader
    loader = DataLoader(dataset, batch_size = batch_size, shuffle=shuffle,
                        collate_fn = MyCollate(pad_idx=pad_idx, batch_first=batch_first)) #MyCollate class runs __call__ method by default
    return loader
