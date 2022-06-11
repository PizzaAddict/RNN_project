from torch.nn.utils.rnn import pad_sequence


#######################################################
#               Collate fn
#######################################################

'''
class to add padding to the batches
collat_fn in dataloader is used for post processing on a single batch. Like __getitem__ in dataset class is used on single example
'''

class MyCollate:
    def __init__(self, pad_idx, batch_first):
        self.pad_idx = pad_idx
        self.batch_first = batch_first#기본값은 True로 놔라
        
    
    #__call__: a default method
    ##   First the obj is created using MyCollate(pad_idx) in data loader
    ##   Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        #get all source indexed sentences of the batch
        source = [item[0] for item in batch] #item: (문장텐서, 라벨) tuple
        #pad them using pad_sequence method from pytorch. 
        source = pad_sequence(source, batch_first=self.batch_first, padding_value = self.pad_idx) 
        attention_pad = source != self.pad_idx
        attention_pad = attention_pad.long()#pad가 아닌 자리만 1
        #get all target indexed sentences of the batch
        target = [item[1] for item in batch] 
        #pad them using pad_sequence method from pytorch. 
        target = pad_sequence(target, batch_first=self.batch_first, padding_value = self.pad_idx)
        return (source, attention_pad), target

