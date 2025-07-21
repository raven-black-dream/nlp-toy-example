import csv
import os
import pandas as pd
import pickle
from sklearn.manifold import TSNE
import spacy
from tqdm.auto import tqdm

NLP = spacy.load('en_core_web_lg')

def init():
    new_embeddings = []
    embeddings = pd.read_table('datasets/glove.6B.300d.txt', 
                               sep=" ", 
                               index_col=0, 
                               header=None, 
                               quoting=csv.QUOTE_NONE)
    vocabulary = embeddings.index.to_list()
    del(embeddings)
    for word in tqdm(vocabulary):
        doc = NLP(str(word))
        vec = list(doc.vector)
        vec.insert(0, word)
        new_embeddings.append(vec)
        
    df = pd.DataFrame(new_embeddings)
    df.to_csv('datasets/embeddings.csv')
    
    
def train_tsne():
    embeddings = pd.read_csv('datasets/embeddings.csv', index_col='0')
    training = embeddings.sample(50000)
    tsne = TSNE(perplexity=30, n_jobs=-1, verbose=3, random_state=42)
    tsne.fit(training)
    pickle.dump(tsne, open('tsne.pkl', 'wb'))
    return tsne
    
    
if __name__ == "__main__":
    
    train_tsne()