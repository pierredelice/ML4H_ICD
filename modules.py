import sys
import subprocess
from pathlib import Path

def install_and_import(package, version=None, conda=False):
    """Install and import a package, optionally using conda."""
    try:
        # Check if the package is already installed with the correct version
        if version:
            __import__(package)
            installed_version = subprocess.check_output([sys.executable, "-m", "pip", "show", package]).decode().split('\n')
            installed_version = [line.split(": ")[1] for line in installed_version if "Version" in line][0]
            if installed_version != version:
                raise ImportError
        else:
            __import__(package)
    except ImportError:
        # Attempt to install the package with conda or pip
        if conda:
            try:
                if version:
                    subprocess.check_call(['conda', 'install', f"{package}={version}", '-y'])
                else:
                    subprocess.check_call(['conda', 'install', package, '-y'])
            except subprocess.CalledProcessError:
                print(f"Conda failed to install {package}, trying pip...")
                if version:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
                else:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            # Install using pip
            if version:
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def is_conda():
    """Check if the script is running in a conda environment."""
    return 'conda' in sys.version or 'Continuum' in sys.version

def parse_requirements(file_path):
    """Parse requirements.txt to extract package names and versions."""
    requirements = {}
    if Path(file_path).is_file():
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '==' in line:
                        package, version = line.split('==')
                        requirements[package.strip()] = version.strip()
                    else:
                        requirements[line.strip()] = None
    return requirements

def freeze_requirements(output_file='requirements.txt'):
    """Generate requirements.txt from the current environment using pip freeze."""
    with open(output_file, 'w') as f:
        subprocess.check_call([sys.executable, '-m', 'pip', 'freeze'], stdout=f)

def main():
    # Check if conda is available
    conda = is_conda()

    # Parse requirements.txt if it exists
    requirements_file = 'requirements.txt'
    requirements = parse_requirements(requirements_file) if Path(requirements_file).is_file() else {}

    # List of packages to check and import
    packages = [
        'pandas', 'numpy', 're', 'string', 'os', 'torch', 'nltk', 
        'itertools', 'spacy', 'unidecode', 'tqdm', 'nltk.corpus', 
        'nltk.tokenize', 'sklearn.model_selection', 'sklearn.preprocessing', 
        'sklearn.ensemble', 'sklearn.metrics', 'sklearn.utils', 'tensorflow', 
        'tensorflow.keras.preprocessing.text', 'tensorflow.keras.preprocessing.sequence', 
        'torch.utils.data', 'transformers', 'pyspellchecker', 'functools', 
        'matplotlib.pyplot', 'matplotlib.ticker', 'seaborn'
    ]

    for package in packages:
        pkg_name = package.split('.')[0]  # Use the base package name for installation
        version = requirements.get(pkg_name)
        install_and_import(pkg_name, version=version, conda=conda)

    # After installing and importing, generate a new requirements.txt
    freeze_requirements()

    # Additional setup or configuration after imports
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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
    from sklearn.utils.class_weight import compute_class_weight

    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    from transformers import AdamW, get_scheduler
    import torch.nn as nn
    import torch.optim as optim

    from spellchecker import SpellChecker  # This should now use pyspellchecker
    from functools import lru_cache

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import seaborn as sns

    pd.set_option('display.max_colwidth', 500)

if __name__ == "__main__":
    main()
