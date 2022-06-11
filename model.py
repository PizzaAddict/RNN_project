import torch
import torch.nn as nn
from torch.nn.functional import softmax as softmax
import torch.nn.functional as f
#-------------------------------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_CLASS = 4
class Encoder(nn.Module):
    def __init__(self,vocab_size,hidden_size,embed_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size).to(device = device)#word_sizexembed size matrix. 정수->벡터 함수
        self.LSTM = nn.LSTM(input_size = embed_size , hidden_size= hidden_size, batch_first=True).to(device = device)
 
    def forward(self, x):
        x = x.to(device=device)#x shape: batch*sentence_size
        x = self.embed(x).to(device = device)#x.shape = batchxsentence size(문장의 토큰 개수)x embed_size
        output, final_status = self.LSTM(x)
        return output, final_status# shape: batch x sentence_size(LSTM이 돈 횟수) x hidden_size_sx

class BaseModel(nn.Module):
    def __init__(self,vocab_size,hidden_size,embed_size,dropp):
        super().__init__()
        self.encoder=Encoder(vocab_size,hidden_size,embed_size)
        #self.posi1 = nn.Parameter(torch.randn(3, 300, 300).unsqueeze(0).to(device = device), requires_grad = True)
        self.decode_embed = nn.Parameter(torch.randn(1,hidden_size).to(device=device), requires_grad = True)
        self.hiddenstate = nn.Sequential(nn.Linear(hidden_size*2, hidden_size),nn.Dropout(dropp),nn.BatchNorm1d(hidden_size), nn.Tanh(), nn.Linear(hidden_size, OUTPUT_CLASS)).to(device = device)
        
    def forward(self, x):#x: 숫자 텐서의 tuple. 하나는 정수이고, 하나는 float attention mask. shape: batch*sequence
        mask = x[1].to(device=device)#batchsize*sentence size
        x = x[0].to(device=device)
        batch_size = x.shape[0]
        encoder_result, _ = self.encoder(x)#encoder result: batch*sentence size*embedding
        decode_duplicated = self.decode_embed.unsqueeze(0).repeat(batch_size,1,1)    
        attention_score = torch.matmul(encoder_result,decode_duplicated.transpose(1,2))#decoder_embed를 column vector로 바꿔주었다. final shape = batch*sentence size* 1
        attention_score = attention_score.squeeze(-1)*mask+(mask != 1).float()*(-1e20)
        attention_score = attention_score.unsqueeze(-1)
        attention_distribution = softmax(attention_score, dim=1)
        attention_value = torch.sum(encoder_result * attention_distribution, dim=1).unsqueeze(1)#아다마르 곱 후 unsqueeze 로 batch*1*hidden_size 로 바꿔준다
        att_dec_concat = torch.cat((attention_value, decode_duplicated), dim = 2).squeeze(1)#batch*hidden_size 두배
        final_vector = torch.tanh(self.hiddenstate(att_dec_concat))#batch*hidden_s
    
        return final_vector
