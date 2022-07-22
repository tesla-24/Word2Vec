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


import collections
# keeping words occuring more than 15
word_count = collections.Counter(cleaned_words)
data = [word for word in cleaned_words if word_count[word] > 15]

unique_words  = set(data)


# Making dictionary from data
def Dictionary(words):
    word_count = collections.Counter(words)
    sorted_vocab = sorted(word_count, key=word_count.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

dict_word_to_index, dict_index_to_word  = Dictionary(data)

data_index_form =  [dict_word_to_index[word] for word in data]

def generate_one_skipgram_data(data_index_form, index_of_centre_word, context_size, no_of_neighbours ):
    neighbour_word_indexes = []
    target = []
    index_to_choose_from =list(range( max(index_of_centre_word-context_size, 0), min(index_of_centre_word+context_size+1, len(data_index_form))  ))
    index_to_choose_from.remove(index_of_centre_word)
    neighbour_indexes = random.sample(index_to_choose_from, no_of_neighbours)
    neighbour_word_indexes.extend([data_index_form[i] for i in neighbour_indexes ])
    
    target.extend([1]*no_of_neighbours)
    negative_word_indexes =  list(random.sample(data_index_form, no_of_neighbours) ) 
    neighbour_word_indexes.extend(negative_word_indexes)
    target.extend([0]*no_of_neighbours)
    return neighbour_word_indexes, target


def generate_skipgram_batch(data_index_form, batch_size, index, context_size, no_of_neighbours):
    indices_to_use = list(range(index*batch_size, (index+1)*batch_size ))
    neighbours = []
    centre_words = []
    target = []
    for ind in indices_to_use:
        neighbour_indices, target_values = generate_one_skipgram_data(data_index_form, ind, context_size, no_of_neighbours )
        neighbours.extend(neighbour_indices)
        target.extend(target_values)
        centre_words.extend([data_index_form[ind]]*len(neighbour_indices))
    return centre_words, neighbours, target


class skipgram(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim=100):
        super(skipgram, self).__init__()

        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)   
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True) 
        self.lin = nn.Linear(embedding_dim,1)

    def forward(self, input_pos, context_pos ):
        if torch.cuda.is_available():
            embed_input = self.input_embeddings(torch.Tensor(input_pos).long().cuda())
            embed_context = self.context_embeddings(torch.Tensor(context_pos).long().cuda()) 
        else:
            embed_input = self.input_embeddings(torch.Tensor(input_pos).long())
            embed_context = self.context_embeddings(torch.Tensor(context_pos).long())
            
        score  = torch.mul(embed_input, embed_context)
        score = self.lin(score)
        target = F.sigmoid(score).squeeze()
        return target

vector_size = 100
vocab_size = len(set(data))
model = skipgram(vocab_size)
loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

if torch.cuda.is_available():
    model = model.cuda()
    
    
batch_size = 32
context_size = 2
no_of_neighbours = 2
for epoch in range(1, 100):
    total_loss = 0.0
    no_of_steps = int(len(data_index_form)/batch_size)
    for i in tqdm(range(no_of_steps)):
        centre_words, neighbour_words, target = generate_skipgram_batch(data_index_form, batch_size, i, context_size, no_of_neighbours)
        model_output = model(centre_words, neighbour_words)
        if torch.cuda.is_available():
            loss = loss_function(model_output,torch.Tensor(target).cuda())
        else:
            loss = loss_function(model_output,torch.Tensor(target))
        optimizer.zero_grad()
        loss.backward()
        total_loss=total_loss + float(loss)
        optimizer.step()
    torch.save(model.state_dict(), './models/model_skipgram-'+str(epoch)+'_vector_size'+str(vector_size)+'.pth')
    print('Epoch:', epoch, '\total_loss:', total_loss)
    
    
embeddings = model.context_embeddings.weight.to('cpu').data.numpy()

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




