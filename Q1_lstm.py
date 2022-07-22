#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
from nltk.corpus import stopwords

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.autograd import Variable


import gensim.downloader as api # package to download text corpus
import nltk # text processing

import string

# download stopwords
nltk.download('stopwords')

# download textcorpus
data = api.load('text8')

# collect all words to be removed
stop = stopwords.words('english') + list(string.punctuation)

actual_words = []
cleaned_words = []
unique_words = set()

# remove stop words
print('removing stop words from text corpus')
for words in data:
    current_nonstop_words = [w for w in words if w not in stop]
    cleaned_words += current_nonstop_words
    actual_words += words

    for ns in current_nonstop_words:
        unique_words.add(ns)

# print statistics
print(len(actual_words), 'words BEFORE cleaning stop words and punctuations')
print(len(cleaned_words), 'words AFTER cleaning stop words and punctuations')
print('vocabulary size: ', len(unique_words))


# keeping words that occur more than 15 times
import collections
word_count = collections.Counter(cleaned_words)
data = [word for word in cleaned_words if word_count[word] > 15]

unique_words  = set(data)


# making dictionary from data
def Dictionary(words):
    word_count = collections.Counter(words)
    sorted_vocab = sorted(word_count, key=word_count.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

dict_word_to_index, dict_index_to_word  = Dictionary(data)

data_index_form =  [dict_word_to_index[word] for word in data]

class lstm(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(lstm, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(0.3)
        
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embed_out = self.embedding(x)
        lstm_out, hidden = self.lstm(embed_out, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -self.vocab_size:] 
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if torch.cuda.is_available():
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

    
    
    
##LSTM
def generate_LSTM_single_data(data_index_form, no_of_prev_words, current_index ):
    neighbour_indices = list(range(current_index-no_of_prev_words, current_index ))
    return [data_index_form[i] for i in neighbour_indices ]

def generate_LSTM_batch(data_index_form, batch_size, current_step, no_of_prev_words):
    centre_words_to_use =  list(range(current_step*batch_size, (current_step+1)*batch_size ))
    context_words = []
    centre_words = []
    for ind in centre_words_to_use:
        context_words_for_ind = generate_LSTM_single_data(data_index_form, no_of_prev_words, ind )
        context_words.append(context_words_for_ind)
        centre_words.extend([data_index_form[ind]])
    return centre_words, context_words    
    

    
vocab_size  = len(unique_words) 
net = lstm(vocab_size, vocab_size, 100, 100, 2)  
if torch.cuda.is_available():
    net = net.cuda()  
    
learning_rate = 0.001    
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


batch_size = 128
embedding_size = 100
no_of_prev_words = 4
for epoch in (range(100)):
    total_loss = 0.0
    hidden_layer = net.init_hidden(batch_size)
    no_of_steps = int(len(data_index_form)/ batch_size)
    for step in tqdm(range(no_of_steps)):
        y,x = generate_LSTM_batch(data_index_form, batch_size, step, no_of_prev_words)
        if torch.cuda.is_available():
            x = torch.tensor(x).cuda() 
            y = torch.tensor(y).cuda() 
        else:
            x = torch.tensor(x) 
            y = torch.tensor(y)           
            
        hidden_layer = tuple([each.data for each in hidden_layer])
        net.zero_grad()
        output, hidden_layer = net(x, hidden_layer)
        loss = criterion(output.squeeze(), y.long())
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        total_loss=total_loss + float(loss)
        optimizer.step()
        
    print('Epoch:', epoch, '\total_loss:', total_loss)
    torch.save(net.state_dict(), './models/model_lstm-'+str(epoch)+'vector_size-'+str(embedding_size)+'.pth')
        
        
embeddings = net.embedding.weight.to('cpu').data.numpy()
no_of_words = 100
test = list(dict_word_to_index.keys())[:no_of_words]
word2vec = embeddings[:no_of_words]

tsne = TSNE(n_components=2, random_state=0, n_iter=1000)

two_D_embedding  = tsne.fit_transform(word2vec)

def plot(embeddings, labels):
    width_in_inches = 20
    height_in_inches = 20
    plt.figure(figsize=(width_in_inches, height_in_inches))
    plt.scatter(embeddings[:,0], embeddings[:,1])
    for word in labels:
        plt.annotate(word, (embeddings[labels.index(word),0], embeddings[labels.index(word),1]))
    plt.show()
plot(two_D_embedding, test)        


# In[ ]:




