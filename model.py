import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
#print ('# Current cuda device: ', torch.cuda.current_device())
HIDDEN_SIZE = 100
OUTPUT_CLASS = 4
EMBED_SIZE = 128
class Encoder(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE).to(device = device)#word_sizexembed size matrix. 정수->벡터 함수
        self.LSTM = nn.LSTM(input_size = EMBED_SIZE , hidden_size= HIDDEN_SIZE, batch_first=True).to(device = device)
 
    def forward(self, x):
        x = x#x shape: batch*sentence_size
        x = self.embed(x).to(device = device)#x.shape = batchxsentence size(문장의 토큰 개수)x embed_size
        output, final_status = self.LSTM(x)
        return output, final_status# shape: batch x sentence_size(LSTM이 돈 횟수) x hidden_size_sx

class Attention_LSTM(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.encoder=Encoder(vocab_size)
        self.decode_embed = nn.parameter(torch.randn(1,HIDDEN_SIZE).to(device=device), requires_grad = True)
        self.hiddenstate = nn.Sequential(nn.Linear(HIDDEN_SIZE*2, HIDDEN_SIZE), nn.BatchNorm1d(HIDDEN_SIZE), nn.Linear(HIDDEN_SIZE, OUTPUT_CLASS)).to(device = device)
        
    def forward(self, x):#x: 숫자 리스트
        encoder_result, final_encoder_status = self.encoder(x)#batch*sentence size*embedding    
        for iteration in range(1, en_sentence_size):
            attention_score = torch.matmul(encoder_result,self.decode_embed.transpose(1,2))#decoder_embed를 column vector로 바꿔주었다. final shape = batch*encoder sentence size* 1(디코더에는 한 번에 한 token만 넣어주므로..)
            #
            attention_distribution = softmax(attention_score, dim=1)
            #print("ALSTM attention_distribution:", attention_distribution)#debug
            attention_value = torch.sum(encoder_result * attention_distribution, dim=1).unsqueeze(1)#아다마르 곱 후 unsqueeze 로 batch*1*hidden_size 로 바꿔준다
            #print("ALSTM attention_value:", attention_value)#debug
            att_dec_concat = torch.cat((attention_value, decoder_result), dim = 2).squeeze(1)#batch*hidden_sie*2
            #print("ALSTM input for final Linear layer:", att_dec_concat)#debug
            final_vector = torch.tanh(self.hiddenstate(att_dec_concat))#batch*hidden_s
            a_output = self.outputLayer(final_vector)#batch*en_word_size
            #print("ALSTM output of final Linear layer:", output)#debug
            #output_softmax = softmax(output, dim=1)
            #output_lsm = logsoftmax(output,dim=1)
            #print("ALSTM Logsoftmax:", output_lsm)#debug
            
            a_output_argmax = a_output.argmax(dim=1).view(-1,1)
            current_truth = ground_truth[:,iteration]
            
            #print(iteration, output_softmax[:,current_truth].shape)
            #print(current_truth)

            if self.training == True:#what to do in training mode -> model.train()...
                if  tf == 1:
                    en_generated = torch.cat((en_generated, current_truth.unsqueeze(1)), dim=1)#teacher forcing
                else:
                    en_generated = torch.cat((en_generated, a_output_argmax), dim=1)#Non_teacher_forcing
            else:
                en_generated = torch.cat((en_generated, a_output_argmax), dim=1)
            #prob_truth = output_lsm[[idx for idx in range(output_lsm.shape[0])],current_truth].unsqueeze(1)
            if output is None:#shape = batch*sentence_size2
                output = a_output.unsqueeze(1)#batch*1*en 사전크기
                #print(output.shape)
            else:
                output = torch.cat((output, a_output.unsqueeze(1)),dim=1)
                #print(output.shape)
            #print(en_generated)
        #print("tensor size of output: ", gpusize(output))# debug
        return en_generated, output

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