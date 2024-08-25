from setuptools import setup, find_packages

setup(
	name = 'ML4H_ICD',
	version = '0.1',
	packages = find_packages('src'),
	package_dir = {'':'src'},
	install_requires = [
		'pandas', 'numpy', 're', 'string', 'os', 'torch', 'nltk', 
    'itertools', 'spacy', 'unidecode', 'tqdm', 'nltk.corpus', 
    'nltk.tokenize', 'sklearn.model_selection', 'sklearn.preprocessing', 
    'sklearn.ensemble', 'sklearn.metrics', 'sklearn.utils', 'tensorflow', 
	  'tensorflow.keras.preprocessing.text', 'tensorflow.keras.preprocessing.sequence', 
    'torch.utils.data', 'transformers', 'spellchecker', 'functools', 
	  'matplotlib.pyplot', 'matplotlib.ticker', 'seaborn'
	],
)
