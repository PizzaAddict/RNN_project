import argparse
import torch
import numpy as np
from tqdm import tqdm
from model import BaseModel
from mydataset import TextDataset, make_data_loader
# from sklearn.metrics import classification_report

def test(args, data_loader, model):
    true = np.array([])
    pred = np.array([])
    model.eval()
    for i, (text, label) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            text = text
            label = label.to(args.device)            
            output= model(text)
            
            label = label.squeeze()
            output = output.argmax(dim=-1)
            output = output.detach().cpu().numpy()
            pred = np.append(pred,output, axis=0)
            
            label = label.detach().cpu().numpy()
            true =  np.append(true,label, axis=0)
    model.train()
    return pred, true


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2022 DL Term Project #2')
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument('--batch_first', action='store_true', help="If true, then the model returns the batch first")
    parser.add_argument('--test_data',  type=str, default='test')
    args = parser.parse_args()

    """
    TODO: You MUST write the same model parameters as in the train.py file !!
    """
    # Model hyperparameters
    embedding_dim = 256 # embedding dimension
    hidden_dim = 1280  # hidden size of RNN
    dropp = 0.5
        

    # Make Test Loader
    test_dataset = TextDataset(args.data_dir, args.test_data)#vocab 내장
    args.pad_idx = test_dataset.vocab.wtoi['<PAD>']
    test_loader = make_data_loader(test_dataset, args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # instantiate model
    model = BaseModel(len(test_dataset.vocab.wtoi), hidden_dim, embedding_dim, dropp)
    model.load_state_dict(torch.load('./model.pt'))
    model = model.to(device)
    
    print(test_dataset.labeldict_reverse)#index->label
    target_names = [ w for i, w in test_dataset.labeldict_reverse.items()]
    # Test The Model
    pred, true = test(args, test_loader, model)
    
    
    accuracy = (true == pred).sum() / len(pred)
    print("Test Accuracy : {:.5f}".format(accuracy))



    ## Save result
    strFormat = '%10s%10s\n'

    with open('result.txt', 'w') as f:
        f.write('Test Accuracy : {:.5f}\n'.format(accuracy))
        f.write('true label  |  predict label \n')
        f.write('-------------------------- \n')
        for i in range(len(pred)):
            f.write(strFormat % (test_dataset.labeldict_reverse[true[i]],test_dataset.labeldict_reverse[pred[i]]))
            
  
    
    # print(classification_report(true, pred, target_names=target_names))
