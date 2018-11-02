import numpy as np
import re
import random
import json
import collections
import numpy as np
#import util.parameters as params
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet as wn 
import os
import pickle
import multiprocessing as mp
from itertools import islice, chain
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger

from dataset import tokenize

##params
#FIXED_PARAMETERS, config = params.load_parameters()

nltk.download('wordnet')


LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": -1
}

PADDING = "<PAD>"
POS_Tagging = [PADDING, 'WP$', 'RBS', 'SYM', 'WRB', 'IN', 'VB', 'POS', 'TO', ':', '-RRB-', '$', 'MD', 'JJ', '#', 'CD', '``', 'JJR', 'NNP', "''", 'LS', 'VBP', 'VBD', 'FW', 'RBR', 'JJS', 'DT', 'VBG', 'RP', 'NNS', 'RB', 'PDT', 'PRP$', '.', 'XX', 'NNPS', 'UH', 'EX', 'NN', 'WDT', 'VBN', 'VBZ', 'CC', ',', '-LRB-', 'PRP', 'WP']
POS_dict = {pos:i for i, pos in enumerate(POS_Tagging)}


stemmer = nltk.SnowballStemmer('english')
tt = nltk.tokenize.treebank.TreebankWordTokenizer()


def load_nli_data(path, snli=False, shuffle = True):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. 
    """
    data = []
    with open(path, encoding='utf-8') as f:
        for line in tqdm(f):
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            data.append(loaded_example)
        if shuffle:
            random.seed(1)
            random.shuffle(data)
    return data

def load_nli_data_genre(path, genre, snli=True, shuffle = True):
    """
    Load a specific genre's examples from MultiNLI, or load SNLI data and assign a "snli" genre to the examples.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. If set to true, it will overwrite the genre label for MultiNLI data.
    """
    data = []
    j = 0
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            if loaded_example["genre"] == genre:
                data.append(loaded_example)
        if shuffle:
            random.seed(1)
            random.shuffle(data)
    return data

def is_exact_match(token1, token2):
    token1 = token1.lower()
    token2 = token2.lower()
    
    token1_stem = stemmer.stem(token1)

    if token1 == token2:
        return True
    
    for synsets in wn.synsets(token2):
        for lemma in synsets.lemma_names():
            if token1_stem == stemmer.stem(lemma):
                return True
    
    if token1 == "n't" and token2 == "not":
        return True
    elif token1 == "not" and token2 == "n't":
        return True
    elif token1_stem == stemmer.stem(token2):
        return True
    return False

def is_antonyms(token1, token2):
    token1 = token1.lower()
    token2 = token2.lower()
    token1_stem = stemmer.stem(token1)
    antonym_lists_for_token2 = []
    for synsets in wn.synsets(token2):
        for lemma_synsets in [wn.synsets(l) for l in synsets.lemma_names()]:
            for lemma_syn in lemma_synsets:
                for lemma in lemma_syn.lemmas():
                    for antonym in lemma.antonyms():
                        antonym_lists_for_token2.append(antonym.name())
                        # if token1_stem == stemmer.stem(antonym.name()):
                        #     return True 
    antonym_lists_for_token2 = list(set(antonym_lists_for_token2))
    for atnm in antonym_lists_for_token2:
        if token1_stem == stemmer.stem(atnm):
            return True
    return False


#TODO: check wordnet synonyms api
def is_synonyms(token1, token2):
    token1 = token1.lower()
    token2 = token2.lower()
    token1_stem = stemmer.stem(token1)
    synonym_lists_for_token2 = []
    for synsets in wn.synsets(token2):
       for lemma_synsets in [wn.synsets(l) for l in synsets.lemma_names()]:
            for lemma_syn in lemma_synsets:
                for lemma in lemma_syn.lemmas():
                    synonym_lists_for_token2.append(lemma.name())
                        # if token1_stem == stemmer.stem(synonym.name()):
                        #     return True 
    synonym_lists_for_token2 = list(set(synonym_lists_for_token2))
    for atnm in synonym_lists_for_token2:
        if token1_stem == stemmer.stem(atnm):
            return True
    return False


def worker(dataset):
    # pat = re.compile(r'\(|\)')
    # def tokenize(string):
    #     string = re.sub(pat, '', string)
    #     return string.split()
    
    shared_content = {}
    
    for example in tqdm(dataset):
            s1_tokenize = tokenize(example['sentence1_binary_parse'])
            s2_tokenize = tokenize(example['sentence2_binary_parse'])


            s1_token_exact_match = [0] * len(s1_tokenize)
            s2_token_exact_match = [0] * len(s2_tokenize)
            s1_token_antonym = [0] * len(s1_tokenize)
            s2_token_antonym = [0] * len(s2_tokenize)
            s1_token_synonym = [0] * len(s1_tokenize)
            s2_token_synonym = [0] * len(s2_tokenize)
            for i, word in enumerate(s1_tokenize):
                matched = False
                for j, w2 in enumerate(s2_tokenize):
                    matched = is_exact_match(word, w2)
                    if matched:
                        s1_token_exact_match[i] = 1
                        s2_token_exact_match[j] = 1
                    ant = is_antonyms(word, w2)
                    if ant:
                        s1_token_antonym[i] = 1
                        s2_token_antonym[j] = 1
                    syn = is_synonyms(word,w2)
                    if syn:
                        s1_token_synonym[i] = 1
                        s2_token_synonym[j] = 1
                    #TODO: add synonym
            
            content = {}

            
            content['sentence1_token_exact_match_with_s2'] = s1_token_exact_match
            content['sentence2_token_exact_match_with_s1'] = s2_token_exact_match
            content['sentence1_token_antonym_with_s2'] = s1_token_antonym
            content['sentence2_token_antonym_with_s1'] = s2_token_antonym
            content['sentence1_token_synonym_with_s2'] = s1_token_synonym
            content['sentence2_token_synonym_with_s1'] = s2_token_synonym
       
            #TODO: syn content
            
            shared_content[example["pairID"]] = content
            # print(shared_content[example["pairID"]])
    # print(shared_content)
    return shared_content

def partition(x, n):
    return ([i for i in islice(x,j,j+n)] for j in range(0,len(x),n))


shared_files = [
    "DIIN/data/multinli_0.9/shared_dev_matched.json",
    "DIIN/data/multinli_0.9/shared_dev_mismatched.json",
    "DIIN/data/multinli_0.9/shared_train.json",
]

data_files = [
    "./DIIN/data/multinli_0.9/multinli_0.9_dev_matched.jsonl",
    "./DIIN/data/multinli_0.9/multinli_0.9_dev_mismatched.jsonl",
    "./DIIN/data/multinli_0.9/multinli_0.9_train.jsonl",
]

fsize = [500, 500, 10000]

p = mp.Pool(8)

for sf, df, fs in zip(shared_files, data_files, fsize):
    print(f"processing {df}...")
    dataset = load_nli_data(df)
    shared = {k:v for d in p.map(worker, partition(dataset, fs)) for k,v in d.items()}
    with open(sf, "w") as f:
        for k in shared:
            f.write(f"{k} {json.dumps(shared[k])}\n")
    
print("done")
            
# dataset = load_nli_data("./DIIN/data/multinli_0.9/multinli_0.9_train.jsonl")#TODO: path
# p = mp.Pool(10)
# shared = {k:v for d in p.map(worker, partition(dataset, 10000)) for k,v in d.items()}

# #save to file
# #TODO: 
# with open("./DIIN/data/multinli_0.9/shared.json", "w") as f:
#     for k in shared:
#         f.write(f"{k} {json.dumps(shared[k])}")

        
