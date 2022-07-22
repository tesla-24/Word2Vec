#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
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
import pandas as pd
from tqdm import tqdm


import gensim.downloader as api 
import nltk 
from nltk.corpus import stopwords
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

# 'cleaned_words' and 'unique_words' to create a word2vec model


# Remove worts that occur less than 15 times
import collections
word_count = collections.Counter(cleaned_words)
data = [word for word in cleaned_words if word_count[word] > 15]

unique_words  = set(data)


# Creating a dictionary
def Dictionary(words):
    # gives count of each word
    word_count = collections.Counter(words)
    # sorting 
    sorted_vocab = sorted(word_count, key=word_count.get, reverse=True)
    # Creating dictionary an inverse dictionary
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

dict_word_to_index, dict_index_to_word  = Dictionary(data)
data_index_form =  [dict_word_to_index[word] for word in data]


train = pd.read_csv('train.csv', sep='\t')
test = pd.read_csv('test.csv', sep='\t')
val = pd.read_csv('val.csv', sep='\t')

# data preprocessing

def remove_punct(sent):
    return ' '.join(word.lower().replace('-',' ').replace('\\',' ').replace('/',' ').strip(string.punctuation).replace("'",'') for word in sent.split())

def only_vocab_words(sent,dictionary):
    vocab_words = []
    for word in sent.split():
        if word in dictionary:
            vocab_words.append(word)
    return ' '.join(i for i in vocab_words)


train['Phrase'] = train['Phrase'].apply(lambda x:remove_punct(x))
train['Phrase'] = train['Phrase'].apply(lambda x:only_vocab_words(x,dict_word_to_index))
train = train.replace(r'^\s*$', np.nan, regex=True)
train = train.replace(np.nan, 'skipped', regex=True)

test['Phrase'] = test['Phrase'].apply(lambda x:remove_punct(x))
test['Phrase'] = test['Phrase'].apply(lambda x:only_vocab_words(x,dict_word_to_index))
test = test.replace(r'^\s*$', np.nan, regex=True)
test = test.replace(np.nan, 'skipped', regex=True) 

val['Phrase'] = val['Phrase'].apply(lambda x:remove_punct(x))
val['Phrase'] = val['Phrase'].apply(lambda x:only_vocab_words(x,dict_word_to_index))
val = val.replace(r'^\s*$', np.nan, regex=True)
val = val.replace(np.nan, 'skipped', regex=True)

train = train.sample(frac=1)
train = train.reset_index()


class sentiment(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(sentiment, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding layer 
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear layer 
        self.fc = nn.Linear(hidden_dim, output_size)
        # sigmoid layer
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):

        batch_size = x.size(0)
        lstm_out, hidden = self.lstm(x, hidden)
        # creating a stack of all lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        # Reshaping the output
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -5:] 
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

    
def sent_to_index(sentence, sent_length, dict_word_to_index):
    index_sent = [dict_word_to_index[i] for i in sentence.split()]
    return index_sent


def word2vec(sentence, sent_len, embedding_size, embeddings, dict_word_to_index):
    arr =[]
    word_index_list = sent_to_index(sentence, sent_len, dict_word_to_index)
    empty = [[0]*embedding_size] * ( sent_len - len(word_index_list) )
    for word_index in word_index_list:
        arr.append(list(embeddings[word_index]))
    if(len(word_index_list) < sent_len):
        return empty + list(arr)
    else:
        return list(arr)[:sent_len]

def generate_batch(train, sent_len, embedding_size, batch_size, step_no, embeddings, dict_word_to_index ):
    x = []
    y = []
    for i in range(batch_size*step_no, min(batch_size*step_no+batch_size, len(train))):
        x.append(word2vec(train['Phrase'][i], sent_len, embedding_size,  embeddings, dict_word_to_index))
        y.append(train['Sentiment'][i])
    return x, y   


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

    
vocab_size  = len(unique_words) 
model_lstm = lstm(vocab_size, vocab_size, 100, 100, 2)
model_lstm.load_state_dict(torch.load('models/model_lstm-5vector_size-100.pth'))
model_lstm.eval() 
embeddings_lstm = model_lstm.embedding.weight.to('cpu').data.numpy()


vocab_size  = len(unique_words) 
net = sentiment(vocab_size, 5, 100, 100, 2)    
    
if torch.cuda.is_available():
    net = net.cuda()    
    

learning_rate = 0.001    
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


batch_size = 128
sent_length = 15 
embedding_size = 100
for epoch in (range(50)):
    total_loss = 0.0
    hidden_layer = net.init_hidden(batch_size)
    no_of_steps = int(len(train)/ batch_size)
    for step in tqdm(range(no_of_steps)):
        x,y = generate_batch(train, sent_length , embedding_size, batch_size, step, embeddings_lstm, dict_word_to_index )
        
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
        

    print('validating : - ')
    batch_size = 128
    no_of_steps = int(len(test)/batch_size)
    val_accuracy = 0
    for step in tqdm(range(no_of_steps)):
        x_val ,y_val = generate_batch(val, sent_length , embedding_size, batch_size, step, embeddings_lstm, dict_word_to_index )

        if torch.cuda.is_available():
            x_val = torch.tensor(x_val).cuda()
            y_val = torch.tensor(y_val).cuda()
        else:
            x_val = torch.tensor(x_val)
            y_val = torch.tensor(y_val)       


        hidden_layer = tuple([each.data for each in hidden_layer])
        output, hidden_layer = net(x_val, hidden_layer)
        output = torch.argmax(output, dim =1)
        val_accuracy =  val_accuracy + list(output - y_val).count(0)
    print('Epoch:', epoch, '\total_loss:', total_loss)
    print('Validation Accuracy is : ', val_accuracy/ len(val))
    torch.save(net.state_dict(), './models/model_sentiment_lstm-'+str(epoch)+'vector_size-'+str(embedding_size)+'.pth')
    
print('Testing : - ')
batch_size = 128
no_of_steps = int(len(test)/batch_size)
val_accuracy = 0
for step in tqdm(range(no_of_steps)):
    x_val ,y_val = generate_batch(test, sent_length , embedding_size, batch_size, step, embeddings_lstm, dict_word_to_index )

    if torch.cuda.is_available():
        x_val = torch.tensor(x_val).cuda()
        y_val = torch.tensor(y_val).cuda()
    else:
        x_val = torch.tensor(x_val)
        y_val = torch.tensor(y_val)       


    hidden_layer = tuple([each.data for each in hidden_layer])
    output, hidden_layer = net(x_val, hidden_layer)
    output = torch.argmax(output, dim =1)
    val_accuracy =  val_accuracy + list(output - y_val).count(0)
print('Test Accuracy is : ', val_accuracy/ len(test))


# In[ ]:




