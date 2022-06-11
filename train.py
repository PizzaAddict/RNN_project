import argparse
import torch
import numpy as np
from tqdm import tqdm
from model import BaseModel
from mydataset import TextDataset, make_data_loader
from os.path import exists
from test import test
def acc(pred,label):#정확도 측정
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def train(args, data_loader, test_loader, model):
    """
    TODO: Change the training code as you need. (e.g. different optimizer, different loss function, etc.)
            You can add validation code. -> This will increase the accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
    min_loss = np.Inf
    max_test_acc = 0.0
    
    for epoch in range(args.num_epochs):
        train_losses = [] 
        train_acc = 0.0
        total=0
        print(f"[Epoch {epoch+1} / {args.num_epochs}]")
        
        model.train()
        for i, (text, label) in enumerate(tqdm(data_loader)):

            label = label.to(args.device)            
            optimizer.zero_grad()

            output = model(text).to(args.device)
            
            label = label.squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/total
        
        pred, true = test(args, test_loader, model)
        epoch_test_acc = (true == pred).sum() / len(pred)
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}'.format(epoch_train_acc*100))
        print('test_accuracy : {:.3f}'.format(epoch_test_acc*100))

        # Save Model
        if epoch_test_acc > max_test_acc:
            torch.save(model.state_dict(), 'model.pt')
            print('test accuracy increased ({:.6f} --> {:.6f}).  Saving model ...'.format(max_test_acc, epoch_test_acc))
            max_test_acc = epoch_test_acc



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='2022 DL Term Project #2')
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument('--batch_first', action='store_true', help="If true, then the model returns the batch first")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument('--num_epochs', type=int, default=5, help="Number of epochs to train for (default: 5)")
    parser.add_argument('--weight_decay', type=float, default=0.0004, help="weight decay for (default: 0.0004)")
    args = parser.parse_args()

    """
    TODO: Build your model Parameters. You can change the model architecture and hyperparameters as you wish.
            (e.g. change epochs, vocab_size, hidden_dim etc.)
    """
    # Model hyperparameters
    embedding_dim = 256 # embedding dimension
    hidden_dim = 1280  # hidden size of RNN
    dropp = 0.5


    # Make Train Loader
    train_dataset = TextDataset(args.data_dir, 'train')#vocab 내장
    train_loader = make_data_loader(train_dataset, args.batch_size)
    test_dataset = TextDataset(args.data_dir, 'test')#vocab 내장
    test_loader = make_data_loader(test_dataset, 64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print("device : ", device)

    # instantiate model
    model = BaseModel(len(train_dataset.vocab.wtoi), hidden_dim, embedding_dim, dropp)
    if exists('./model.pt'):
        model.load_state_dict(torch.load('./model.pt'))
    model = model.to(device)

    # Training The Model
    train(args, train_loader,test_loader, model)