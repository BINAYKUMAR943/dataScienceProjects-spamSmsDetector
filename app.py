import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import streamlit as st
import pickle

def cleanDocument(document):  
    '''This function performs some cleaning operations such as removing number, extra spaces etc. to text data'''
    # Changing the text to lowercase
    document=document.lower()

    #Removing extra spaces
    document=re.sub(r' +',' ',document)

    #Removing Punctuality
    exclude = set(string.punctuation)
    document = ''.join(ch for ch in document if ch not in exclude)

    #Removing numbers
    document=document.strip()
    document=re.sub(r'\d+','',document)

    #Removing Stop words
    selected_words=[]
    stop_words=list(stopwords.words("english"))
    tokens=word_tokenize(document)
    for i in range(len(tokens)):
        if tokens[i] not in stop_words:
            selected_words.append(tokens[i])    
    return " ".join(selected_words)

# Function for lemmetization
def lemmetize(document): 
    ''' This function does lemmetizaton of doucument'''
    lem=WordNetLemmatizer()
    tokens=word_tokenize(document)
    for i in range(len(tokens)):
        lem_word=lem.lemmatize(tokens[i])
        tokens[i]=lem_word

    document=" ".join(tokens)
    return document

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('nb.pkl','rb'))
le=pickle.load(open('le.pkl','rb'))
st.title('SMS classifier')
sms=st.text_area('Please enter sms')
if st.button('Predict'):
    #clean
    cleanDocument(sms)
    #Lemmetize
    lemmetize(sms)
    #Vectorize
    sms=tfidf(sms).toarray()
    #Predict
    sms_type_nbr=model.predict(sms)
    #GetType
    sms_type_txt=le.inverse_transform(sm_type)
    st.header(sms_type_txt)