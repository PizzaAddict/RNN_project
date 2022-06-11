import torch
import torch.nn as nn
from torch.nn.functional import softmax as softmax
import torch.nn.functional as f
#-------------------------------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch.cuda.set_device(device) # change allocation of current GPU
#print ('# Current cuda device: ', torch.cuda.current_device())
HIDDEN_SIZE = 100
OUTPUT_CLASS = 4
EMBED_SIZE = 128
DROPP = 0.2
class Encoder(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE).to(device = device)#word_sizexembed size matrix. 정수->벡터 함수
        self.LSTM = nn.LSTM(input_size = EMBED_SIZE , hidden_size= HIDDEN_SIZE, batch_first=True).to(device = device)
 
    def forward(self, x):
        x = x.to(device=device)#x shape: batch*sentence_size
        x = self.embed(x).to(device = device)#x.shape = batchxsentence size(문장의 토큰 개수)x embed_size
        output, final_status = self.LSTM(x)
        return output, final_status# shape: batch x sentence_size(LSTM이 돈 횟수) x hidden_size_sx

class Attention_LSTM(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.encoder=Encoder(vocab_size)
        #self.posi1 = nn.Parameter(torch.randn(3, 300, 300).unsqueeze(0).to(device = device), requires_grad = True)
        self.decode_embed = nn.Parameter(torch.randn(1,HIDDEN_SIZE).to(device=device), requires_grad = True)
        self.hiddenstate = nn.Sequential(nn.Linear(HIDDEN_SIZE*2, HIDDEN_SIZE),nn.Dropout(DROPP),nn.BatchNorm1d(HIDDEN_SIZE), nn.Tanh(), nn.Linear(HIDDEN_SIZE, OUTPUT_CLASS)).to(device = device)
        
    def forward(self, x):#x: 숫자 텐서. shape: batch*sequence
        batch_size = x.shape[0]
        print(x.shape)
        encoder_result, _ = self.encoder(x)#encoder result: batch*sentence size*embedding
        decode_duplicated = self.decode_embed.unsqueeze(0).repeat(batch_size,1,1)    
        attention_score = torch.matmul(encoder_result,decode_duplicated.transpose(1,2))#decoder_embed를 column vector로 바꿔주었다. final shape = batch*sentence size* 1
        attention_distribution = softmax(attention_score, dim=1)
        attention_value = torch.sum(encoder_result * attention_distribution, dim=1).unsqueeze(1)#아다마르 곱 후 unsqueeze 로 batch*1*hidden_size 로 바꿔준다
        att_dec_concat = torch.cat((attention_value, decode_duplicated), dim = 2).squeeze(1)#batch*hidden_size 두배
        print(att_dec_concat.shape)
        final_vector = torch.tanh(self.hiddenstate(att_dec_concat))#batch*hidden_s
    
        return final_vector

class BaseModel(nn.Module):
    '''
    input_size -> text vocab size
    '''
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first):
        super(BaseModel, self).__init__()

        self.num_layers = num_layers
        self.batch_first = batch_first   
        self.hidden_dim = hidden_dim

        """
        TODO: Implement your own model. You can change the model architecture.
        """
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_dim, output_size)

    # the size of x in forward is (seq_length, batch_size) if batch_first=False
    def forward(self, x):
        batch_size = x.size(0) if self.batch_first else x.size(1)

        #h_0: (num_layers * num_directions, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        embedding = self.embedding(x)

        outputs, hidden = self.rnn(embedding, None)  # outputs.shape -> (sequence length, batch size, hidden size)

        outputs = outputs[:, -1, :] if self.batch_first else outputs[-1, :, :]
        output = self.fc(outputs)
        
        return output, hidden