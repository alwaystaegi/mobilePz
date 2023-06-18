from googletrans import Translator
import re
import nltk
from nltk.stem import WordNetLemmatizer


from nltk.corpus import stopwords




def translate_text(text):
    translator=Translator()
    result=""
    try:
        for sentence in text.split("."):
            result=result+translator.translate(sentence,dest='ko').text
    
        print(result)
        return result
    except Exception:
        return

def replace_text(text):
    lemmatizer=WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) # Load stop words
    pers_types = ['INFP' ,'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP' ,'ISFP' ,'ENTJ', 'ISTJ','ENFJ', 'ISFJ' ,'ESTP', 'ESFP' ,'ESFJ' ,'ESTJ']
    pers_types = [p.lower() for p in pers_types]  
    
    text=text.lower()
        
    text=re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+',' ',text)
        
    text=re.sub('[^0-9a-z]',' ',text)
        
    text = " ".join([word for word in text.split() if word not in stop_words]) # Remove stop words
    #print(len(sentence))
        
    for p in pers_types:
        text = re.sub(p, '', text)
        #print(len(sentence))
        
    text = lemmatizer.lemmatize(text) # Lemmatize words
    return text