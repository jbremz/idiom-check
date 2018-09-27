# Adapted from bit.ly/2DJ7WD8

import tensorflow as tf
from tensorflow.contrib import rnn

def corpus(file_list):
    '''
    Inputs:
    file_list - the list of paths of the text files to create the corpus
    
    Returns:
    A list of strings containing the corpus
    '''
    corpus = []

    for file_path in file_list:
        with open(file_path) as f_input:
            sample = re.sub(r'\([^)]*\)', ' ', re.sub('<[^>]+>', '', f_input.read()).replace('\n', ' '))[:20000]
            sample = re.sub(r'/[^\w\s]/gi', '', sample)
            sample = sample.replace(chr(8221),"")
            if len(sample) > 0:
                corpus.append(sample)

    return corpus

def int_encode(groups, encoder):
    '''
    Input:
    groups - array - groups of characters to be encoded
    encoder - a sklearn.preprocessing.LabelEncoder object which has been prefitted to a vocab
    
    Returns:
    int_encoded - array - the integer encoded groups
    vocab - array - an array where the index of each item corresponds to the integers in int_encoded: a lookup
    
    TODO: predefined vocab as input
    '''
    flat_groups = groups.flatten()
    try:
        int_encoded = encoder.transform(flat_groups).reshape(groups.shape)
    except ValueError:
        print(''.join(list(groups[:,1])))
    
    return int_encoded

vocab = np.array([chr(i) for i in range(2048)])
encoder = LabelEncoder().fit(vocab)

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 100
display_step = 200

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 #

