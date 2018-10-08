import re, os, glob
import numpy as np
from tqdm import tqdm_notebook as tqdm
import pandas as pd

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
            sample = f_input.read()[:20000] # only select the first 20000 characters for memory purposes
            sample = re.sub(r'[^\u0000-\u0800]', '', sample) # select only the first 2048 UTF-8 characters
            sample = re.sub(r'\([^)]*\)', ' ', re.sub('<[^>]+>', '', sample).replace('\n', ' '))
            sample = re.sub(r'/[^\w\s]/gi', '', sample)
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
    '''
    flat_groups = groups.flatten()
    try:
        int_encoded = encoder.transform(flat_groups).reshape(groups.shape)
    except ValueError:
        print(''.join(list(groups[:,1])))
    
    return int_encoded

def makeDF(data_path, samples):
    '''
    Inputs:
    data_path - the path of the txt directory containing the samples
    samples - number of text samples to take for each language

    Returns:
    A pandas dataframe with the string and the language

    '''
    languages = next(os.walk('./txt'))[1]

    strings = np.array([])
    langs = np.array([])

    for language in tqdm(languages):
        file_list = glob.glob(os.path.join(data_path, "txt", language,"*.txt"))
        the_corpus = corpus(np.random.choice(file_list, samples))
        strings = np.concatenate([strings, the_corpus])
        langs = np.concatenate([langs, [language]*samples])

    return pd.DataFrame(np.array([strings, langs]).T, columns=['string','language'])