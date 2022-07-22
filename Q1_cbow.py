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


# keeping words occuring more than 15
import collections
word_count = collections.Counter(cleaned_words)
data = [word for word in cleaned_words if word_count[word] > 15]

unique_words  = set(data)


# generating dictionary
def Dictionary(words):
    word_count = collections.Counter(words)
    sorted_vocab = sorted(word_count, key=word_count.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

dict_word_to_index, dict_index_to_word  = Dictionary(data)

data_index_form =  [dict_word_to_index[word] for word in data]

##CBOW
def generate_cbow_single_data(data_index_form, context_size, current_index ):
    neighbour_indices = list(range(current_index-context_size, current_index+context_size+1 ))
    neighbour_indices.remove(current_index)
    return [data_index_form[i] for i in neighbour_indices ]

def generate_cbow_batch(data_index_form, batch_size, current_step, context_size):
    centre_words_to_use =  list(range(current_step*batch_size, (current_step+1)*batch_size ))
    context_words = []
    centre_words = []
    for ind in centre_words_to_use:
        context_words_for_ind = generate_cbow_single_data(data_index_form, context_size, ind )
        context_words.append(context_words_for_ind)
        centre_words.extend([data_index_form[ind]])
    return centre_words, context_words


### CBOW MODEL
class CBOW(torch.nn.Module):

    def __init__(self,  context_size, vocab_size, embedding_dim=100):
        super(CBOW, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)       
        self.activation_softmax = nn.LogSoftmax(dim = -1)
        

    def forward(self, x):
        x = torch.sum( self.embedding_layer(x), 1 )
        x = self.linear1(x)
        x = self.activation_softmax(x)
        return x
    
vector_size = 100
vocab_size = len(set(data))
context_size = 2    
model = CBOW(context_size, vocab_size)

loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
if torch.cuda.is_available():
    model = model.cuda()
    
    
batch_size = 64
current_step = 0

for epoch in range(50):
    total_loss = 0.0
    no_of_steps = int(len(data_index_form)/batch_size)
    for current_step in tqdm(range(no_of_steps)):
        y, x  = generate_cbow_batch(data_index_form, batch_size, current_step, context_size)
        
        if torch.cuda.is_available():
            x = Variable(torch.LongTensor(x)).cuda()
            y = Variable(torch.LongTensor(y)).cuda()
        else:
            x = Variable(torch.LongTensor(x))
            y = Variable(torch.LongTensor(y))
        
        model.zero_grad()
        output_log_probability = model(x)
        loss = loss_function(output_log_probability ,y)
        loss.backward()
        
        optimizer.step()
        del x
        del y
        total_loss=total_loss + float(loss)
    print('Epoch:', epoch, '\tLoss:', total_loss)
    torch.save(model.state_dict(), './models/model_cbow-'+str(epoch)+'_vector_size'+str(vector_size)+'.pth')
    
    
embeddings = model.embedding_layer.weight.to('cpu').data.numpy()
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




