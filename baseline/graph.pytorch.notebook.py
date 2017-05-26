
# coding: utf-8

# # STATIC GRAPH

# In[55]:

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable


# ## DATA

# In[56]:

import data_utils
metadata, idx_q, idx_a = data_utils.load_data('../data/')


# In[57]:

# add special symbol
i2w = metadata['idx2w'] + ['GO']
w2i = metadata['w2idx']
w2i['GO'] = len(i2w)-1


# ## Parameters

# In[282]:

batch_size = 256
L = len(idx_q[0])
vocab_size = len(i2w)
hidden_size = 256


# In[84]:

class Config:
    pass

config = Config()
config.printsize = True


# In[169]:

len(idx_q)


# 
# ## Graph

# In[444]:

def initial_state(batch_size, hidden_size):
    state = torch.zeros([batch_size, hidden_size])
    return Variable(state.cuda())

def psize(name, variable):
    if config.printsize:
        print(name, variable.size(), '\t', type(variable.data))
        
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Encoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
                
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.encode = nn.LSTMCell(hidden_size, hidden_size)
            
    def forward(self, enc_inputs, hidden, batch_size):
        input_length = enc_inputs.size()[0]
        psize('enc_inputs', enc_inputs)
        enc_embeddings = self.embed(enc_inputs)
        psize('enc_embeddings', enc_embeddings)
        enc_embeddings = enc_embeddings.view(input_length, 
                                            batch_size, 
                                            hidden_size)            #LxBxH       
                
        psize('enc_embeddings', enc_embeddings)        
        hidden, cell_state = hidden
        for i in range(enc_embeddings.size()[0]):
            hidden, cell_state = self.encode(enc_embeddings[i], (hidden, cell_state))
            
        return hidden, cell_state
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.decode = nn.LSTMCell(hidden_size, hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.project = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, outputs, hidden, batch_size):
        length = outputs.size()[0]
        psize('hidden', hidden[0]), psize('hidden', hidden[1])
        predicted_outputs = []
    
        dec_embeddings = self.embed(outputs).view(length,
                                                 batch_size,
                                                 hidden_size)           #LxBxH
    
        GO = torch.LongTensor([w2i['GO']] * batch_size).cuda()            
        GO = Variable(GO)
        psize('GO', GO)
        
        dec_input = GO
        hidden, cell_state = hidden
        for i in range(length):
            #psize('\tdec_input', dec_input)
            dec_input_emb = self.embed(dec_input)
            #psize('\tdec_input_emb', dec_input_emb)

            hidden, cell_state = self.decode(dec_input_emb, (hidden, cell_state))
            predicted_outputs.append(hidden)

            topv, topi = self.project(F.log_softmax(hidden)).topk(1)
            #psize('\ttopi', topi)
            dec_input = topi.squeeze(1)
            
        predicted_outputs = torch.stack(predicted_outputs).squeeze(1)
        psize('predicted_outputs', predicted_outputs)
              
        predicted_outputs = self.project(predicted_outputs.view(length*batch_size, hidden_size))
        psize('predicted_outputs', predicted_outputs)
        predicted_outputs = predicted_outputs.view(length, batch_size, vocab_size)
        psize('predicted_outputs', predicted_outputs)

        return predicted_outputs
    


# # TRAINING

# In[442]:

from pprint import pprint
from tqdm import tqdm
def train_epochs(epochs, encoder, decoder, eoptim, doptim, criterion, print_every=1):
    encoder.train()
    decoder.train()
    losses = []
    config.printsize = True

    for epoch in tqdm(range(epochs+1)):
        loss = train(encoder, decoder, eoptim, doptim, criterion, idx_q, idx_a,
                    print_every=print_every*100)    
        if epoch % print_every == 0:
            losses.append(loss)
            print('{} - loss: {}'.format(epoch, loss))

        
def train(encoder, decoder, eoptim, doptim, criterion, question_ids, answer_ids, print_every=100):
    input_length = len(question_ids[0])
    dataset_size = len(idx_q)
    batch_count  = dataset_size//batch_size
    for batch_index in range(batch_count):
        #l,r = batch_index * batch_size, (batch_index + 1) * batch_size
        import numpy as np
        indices = np.arange(batch_size)
        np.random.shuffle(indices)
        indices = indices * batch_count
        
        question_id, answer_id = question_ids[indices], answer_ids[indices]
        _batch_size = len(question_id)
        if _batch_size != batch_size:
            print('breaking because batch sizes do not match')
            break

        data = Variable(torch.from_numpy(question_id).long().cuda().t())
        target = Variable(torch.from_numpy(answer_id).long().cuda().t())

        eoptim.zero_grad(), doptim.zero_grad()    
        initial_hidden = initial_state(batch_size, hidden_size).cuda(), initial_state(batch_size, hidden_size).cuda()
        
        encoder_output = encoder(data, initial_hidden, _batch_size)
        decoder_output = decoder(target, encoder_output, _batch_size)
        loss = 0
        for i in range(input_length):
            logits = F.log_softmax(decoder_output[i])
            loss += criterion(logits, target[i])    
            
        loss.backward()
        eoptim.step(), doptim.step()
        config.printsize = False
        
        if batch_index % print_every == 0:
            print('\t{} - loss: {}'.format(batch_index, loss.data[0]))
        
    return loss.data[0]


encoder = Encoder(vocab_size, hidden_size)
decoder = Decoder(vocab_size, hidden_size)

encoder.cuda()
decoder.cuda()


encoder.load_state_dict(torch.load('graph.pytorch.encoder.pth'))
decoder.load_state_dict(torch.load('graph.pytorch.decoder.pth'))


criterion = nn.NLLLoss()
eoptim = optim.SGD(encoder.parameters(), lr=0.01, momentum=0.1)
doptim = optim.SGD(decoder.parameters(), lr=0.01, momentum=0.1)


encoder_test = Encoder(vocab_size, hidden_size)
decoder_test = Decoder(vocab_size, hidden_size)
encoder_test.cuda()
decoder_test.cuda()


def train_epochs_(epochs, encoder, decoder, eoptim, doptim, criterion, print_every=1, validate_every=10):
    encoder.train()
    decoder.train()
    losses = []
    config.printsize = True

    for epoch in tqdm(range(epochs+1)):
        print('--- epoch: {} ---'.format(epoch))
        loss = train(encoder, decoder, eoptim, doptim, criterion, idx_q, idx_a,
                    print_every=print_every*1000)    
        if epoch % print_every == 0:
            losses.append(loss)
            print('{} - loss: {}'.format(epoch, loss))

        torch.save(encoder.state_dict(), 'graph.pytorch.encoder.pth')
        torch.save(decoder.state_dict(), 'graph.pytorch.decoder.pth')

        encoder_test.load_state_dict(torch.load('graph.pytorch.encoder.pth'))
        decoder_test.load_state_dict(torch.load('graph.pytorch.decoder.pth'))

        if epoch % validate_every == 0:
            test_q, test_a = idx_q[-1], idx_a[-1]

            encoder_test.eval()
            decoder_test.eval()

            test_q = Variable(torch.from_numpy(test_q).long().cuda())
            test_a = Variable(torch.from_numpy(test_a).long().cuda())

            #config.printsize = True
            _batch_size = 1
            hidden = initial_state(_batch_size, hidden_size).cuda(), initial_state(_batch_size, hidden_size).cuda()
            predictions = decoder_test.predict(test_a, encoder_test(test_q, hidden, _batch_size), _batch_size)
            predictions = predictions.squeeze(1)
            predictions = F.log_softmax(predictions).max(1)[1].squeeze(1)
            
            predictions_ = decoder_test(test_a, encoder_test(test_q, hidden, _batch_size), _batch_size)
            predictions_ = predictions_.squeeze(1)
            predictions_ = F.log_softmax(predictions_).max(1)[1].squeeze(1)

            print(arr2sent(predictions.cpu().data.numpy()))
            print(arr2sent(predictions_.cpu().data.numpy()))
            print(arr2sent(test_a.cpu().data.numpy()))

def arr2sent(arr):
    return ' '.join([i2w[item] for item in arr])


if __name__ == '__main__':
    train_epochs_(1000,  encoder, decoder, eoptim, doptim, criterion, print_every=1)
