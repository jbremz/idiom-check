import re

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
            sample = re.sub(r'\([^)]*\)', ' ', re.sub('<[^>]+>', '', f_input.read()).replace('\n', ' '))[:20000] # only select the first 20000 characters for memory purposes
            sample = re.sub(r'/[^\w\s]/gi', '', sample)
            sample = re.sub(r'[^\u0000-\u0800]', '', sample) # select only the first 2048 UTF-8 characters
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