import pandas as pd
import numpy as np
import re, string, os, torch, nltk, itertools, spacy
from unidecode import unidecode
from tqdm import tqdm 
tqdm.pandas()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, DistilBertForSequenceClassification
from transformers import AdamW, get_scheduler

from spellchecker import SpellChecker
from functools import lru_cache



# ## Read data

def read_file(path):
  """
  Access and filter data, parameters are:
  ---------------------------
    - path: file to access data
    - columnas: selected columns for analysis 
    - base: year of dataset 
  """
  columnas = ['sexo','edad','des_causa_a','des_causa_b','des_causa_c','causa_a','causa_b',
'des_causa_c','causa_c']
  
  df = pd.read_csv(path, usecols = columnas)
  for x in df.columns:
    if x in df.select_dtypes(include = 'category').columns:
      df[x] = df[x].astype(str).str.lower()
  print(df.columns)
  print(df)

  #In labels count the number of codes related diagnosis 
  def count_words(causa):
    """
    causa: refers to ICD code, sometimes more than 1 code is assigned 
    """
    if isinstance(causa, str):
      return len(causa.split())
    return 0
  
  len_causa = []
  for causa in ['causa_a','causa_b','causa_c']:
    df[f'{causa}_word_count'] = df[causa].apply(count_words)
    len_causa.append(f'{causa}_word_count')  
  #filtered_df = df[df[len_causa].apply(lambda x: any(x == 1), axis=1)]
  """categories 888 and 999 were assigned for newborns or under 1 year old """
  df['edad'] = df['edad'].replace({888:0, 999:0})

  return df

#In this section we melt 
def merge_data(data):
  row1 = data.melt(
  value_vars=["des_causa_a", "des_causa_b", "des_causa_c"], 
  var_name="causa_type", 
  value_name="causa")
  
  row2 = data.melt(
  value_vars = ['causa_a','causa_b','causa_c'],
  var_name='causa_code',
  value_name = 'causa_icd')
  merged_data = pd.concat([row1.drop('causa_type', axis=1),
                    row2[['causa_icd']]], axis=1)
  return merged_data

def clean_causa_icd(df):
    """
    Clean and filter the 'causa_icd' column in the dataframe.
    """
    def removechar(text):
      if isinstance(text, str):
        text = text.lower()
        text = unidecode(text)
        punctuation = '''|!()-[]{};:'"\,<>./?@#$%^&*_~'''
        text = text.translate(str.maketrans('', '', punctuation))
        text = re.sub(r'-','', text)
        text = text.strip()
        return text
      return None
  
    df['causa_icd'] = df['causa_icd'].apply(removechar).str.strip().str.replace('[^\x00-\x7F]', '', regex=True)
    df = df[df['causa_icd'].notna() & df['causa_icd'].str.match(r'^[a-zA-Z]\d{3}$|^[a-zA-Z]\d{2}[a-zA-Z]$')]
    df = df[df['causa_icd'].isin(df.causa_icd.value_counts()[:30].index)]
    df['sexo'] = df['sexo'].replace({0:int(np.nan), 
                                     9:int(np.nan)})
    df = df[pd.Series(df['sexo']).astype(int).notna()]
    df = df.drop(columns = ['base'], axis=1)
        
    return df

nlp = spacy.load('es_core_news_md')

def clean_description(text):
  if not isinstance(text, str):
    return ""
  text = unidecode(text.lower().strip())
  punctuation = '''|!()-[]{};:'"\,<>./?@#$%^&*_~'''
  text = text.translate(str.maketrans('', '', punctuation))
  words = word_tokenize(text)
  spanish_stopwords = set(stopwords.words('spanish'))
  filtered_words = [word for word in words if word not in spanish_stopwords]
  filtered_text = ' '.join(filtered_words)
  doc = nlp(filtered_text)
  lemmatized_words = ' '.join(token.lemma_ for token in doc)
  
  return lemmatized_words

# Initialize SpellChecker once
spell = SpellChecker(language='es')
custom_words = {"covid", "sars", "cov", "sars-cov2", "covid19"}
spell.word_frequency.load_words(custom_words)

# Cache corrected words
@lru_cache(maxsize=10000)
def correct_word(word):
    if word in custom_words:
        return word
    return spell.correction(word) or word

def spellcheck_correction(text):
    words = text.split()
    corrected_words = [correct_word(word) for word in words]
    return ' '.join(corrected_words)

# Define the remove_digits function
def remove_digits(text):
    if isinstance(text, str):
        return re.sub(r'\d+', '', text)
    return text


path = '../../github/data/seedcausa.csv' ## access to data


# In[116]:


#Read data

df = clean_causa_icd(merge_data(read_file(path)))

df['causa'] = df['causa'].progress_apply(clean_description)
df['sexo'] = df.sexo.astype(int)
df['cause'] = df.loc[:,['sexo','edad','causa']].progress_apply(lambda x: ' '.join(x.astype(str)), axis=1)
df['causa'] = df['causa'].progress_apply(remove_digits).progress_apply(spellcheck_correction)
df['len_causa'] = df['causa'].apply(lambda x: len(x.split()))

print(df)

