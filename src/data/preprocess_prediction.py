import spacy, nltk, string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
from spellchecker import SpellChecker
from functools import lru_cache

nltk.download('stopwords')
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