# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:33:29 2023

@author: EVHARLEY
"""

from bs4 import BeautifulSoup
import os
import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from tqdm.auto import tqdm

tqdm.pandas()

class CTFIDFVectorizer(TfidfTransformer):
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)
        self._idf_diag = None

    def fit(self, X: sp.csr_matrix, n_samples: int):
        """Learn the idf vector (global term weights)

        Parameters
        ----------
        X : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts.

        """

        # Prepare input
        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        # Calculate IDF scores
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        avg_nr_samples = int(X.sum(axis=1).mean())
        idf = np.log(avg_nr_samples / df)
        self._idf_diag = sp.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=dtype)
        return self

    def transform(self, X: sp.csr_matrix, copy=True) -> sp.csr_matrix:
        """Transform a count-based matrix to c-TF-IDF

        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            a matrix of term/token counts

        Returns
        -------
        vectors : sparse matrix of shape (n_samples, n_features)

        """

        # Prepare input
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES, copy=copy)
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        n_samples, n_features = X.shape

        # idf_ being a property, the automatic attributes detection
        # does not work as usual and we need to specify the attribute
        # name:
        check_is_fitted(self, attributes=["idf_"],
                        msg='idf vector is not fitted')

        # Check if expected nr features is found
        expected_n_features = self._idf_diag.shape[0]
        if n_features != expected_n_features:
            raise ValueError("Input has n_features=%d while the model"
                             " has been trained with n_features=%d" % (
                                 n_features, expected_n_features))

        X = X * self._idf_diag

        if self.norm:
            X = normalize(X, axis=1, norm='l1', copy=False)

        return X
   

def preprocess_jobs_data():
    docs = []
    for root, subdir, files in os.walk("datasets/jobs"):
        for file in tqdm(files, desc='files'):
            temp = pd.read_csv(os.path.join(root, file))
            descs = temp['description'].to_list()
            
            for i in tqdm(range(len(descs)), desc='rows'):
                try:
                    soup = BeautifulSoup(descs[i], 'html.parser')
                    docs.append({'title': file.split('.')[0], 
                                 'description': soup.text})
                except TypeError:
                    continue
                
    df = pd.DataFrame(docs)
    df.to_csv('jds.csv')
    return df


def class_based_tfidf(df:pd.DataFrame):
    
    docs_grouped = df.groupby(['title'], as_index=False).agg({'description': ' '.join})
    count_vector = CountVectorizer().fit_transform(docs_grouped.description)
    ctfidf = CTFIDFVectorizer()
    ctfidf.fit(count_vector, df.shape[0])
    ctfidf_vector = ctfidf.transform(count_vector).toarray()
    titles = docs_grouped.title.to_list()
    data = []
    for i in range(len(titles)):
        temp = [titles[i]]
        temp.extend(ctfidf_vector[i])
        data.append(temp)
    ctfidf_df = pd.DataFrame(data)
    pickle.dump(ctfidf_df, open('ctfidf_vector.pkl', 'wb'))
    
    
    
    


if __name__ == '__main__':
    
    if not os.path.exists('jds.csv'):
        df = preprocess_jobs_data()
    else:
        df = pd.read_csv('jds.csv')
        df.dropna(inplace=True)
    output = class_based_tfidf(df)
    pickle.dump(open('cTfIdf_jds.pkl', 'wb'))
    
    
    