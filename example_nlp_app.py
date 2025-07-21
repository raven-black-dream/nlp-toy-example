import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import streamlit as st
import subprocess
import sys


subprocess.run([f"{sys.executable}", "-m" , "spacy download en_core_web_lg"])
print("Why isn't this waiting")
NLP = spacy.load("en_core_web_lg")  

@st.cache_data
def load_2D_data() -> pd.DataFrame:
    df = pd.read_pickle('reduced.pkl')
    titles = list(df.pop('title'))
    vectors = TSNE().fit_transform(df)
    data = []
    for i in range(len(titles)):
        temp = [titles[i]]
        temp.extend(list(vectors[i]))
        data.append(temp)
        
    
    return pd.DataFrame(data, columns=['title', 'x', 'y'])

@st.cache_data
def load_data() -> pd.DataFrame():
    return pd.read_pickle('reduced.pkl')
    

st.title("NLP Toy Example")  
    
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Lemmatization")
        
        with st.form('Lemmatization'):
            text1 = st.text_input('Enter a word or phrase')
            submit1 = st.form_submit_button()
            
            if submit1:
                doc1 = NLP(text1)
                st.write([token.lemma_ for token in doc1])
    with col2:
        st.header('Tokenization')
        
        with st.form('Tokenization'):
            text2 = st.text_input('Enter a word or phrase')
            submit2 = st.form_submit_button()
            
            if submit2:
                doc2 = NLP(text2)
                st.write([token for token in doc2])
                
                
    with st.form('Text Similarity'):
        st.header("Text Similarity")
        text3 = st.text_input("First word or phrase")
        text4 = st.text_input("Second word or phrase")
        submit3 = st.form_submit_button()
        
        if submit3:
            king = NLP("king")
            queen = NLP('queen')
            cat = NLP('cat')
            dog = NLP("dog")
            
            
            doc3 = NLP(text3)
            doc4 = NLP(text4)
            
            docs = [king.vector, queen.vector, dog.vector, 
                    cat.vector, doc3.vector, doc4.vector]
            
            pca = PCA(random_state=42)
            vectors = pca.fit_transform(np.array(docs))
            v1 = [v[0] for v in vectors]
            v2 = [v[1] for v in vectors]
            
            df = pd.DataFrame(
                {
                    'input': ["king", 'queen', 'cat', 'dog', text3, text4],
                    'x': v1,
                    'y': v2
                    }
                )
            
            fig = px.scatter(df, x='x', y='y', color="input")
            st.plotly_chart(fig)



with st.container():
    st.header("Job Description Data")
    data = load_2D_data()
    fig = px.scatter(data, x='x', y='y', color='title')
    st.plotly_chart(fig)
    

with st.container():
    data = load_data()
    st.header("Machine Learning Application")
    with st.form("Similarity"):
        choice = st.selectbox("Choose a Job Title", options=data['title'].to_list())
        submit4 = st.form_submit_button()
        
    if submit4:
        matrix = data.loc[:, data.columns != 'title'].copy()
        value = data.loc[data['title'] == choice, data.columns != 'title'].to_numpy()
        similarity = cosine_similarity(matrix, value)
        similarity_df = pd.DataFrame({'title': data['title'].to_list(),
                                      'similarity': list(similarity)})
        similarity_df.sort_values('similarity', ascending=False, inplace=True)
        st.dataframe(similarity_df.head(10))
        
    
    
        
    
        
        
        
    
