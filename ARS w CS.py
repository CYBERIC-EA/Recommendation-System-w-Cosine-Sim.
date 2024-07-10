import os
import numpy as np
import pandas as pd
import warnings
import scipy as sp
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Download necessary NLTK data
nltk.download('stopwords')

# Default theme and settings
pd.options.display.max_columns = None

# Handle warnings
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")

# Load dataset
anime_path = '.venv\\anime-dataset-2023.csv'
anime = pd.read_csv(anime_path)
print("Dataset Information:\n")
print(anime.info())

# Select relevant columns for recommendation
reco_df = anime[['Name', 'Synopsis']]
print("Missing Values in the Anime Dataset (%) : \n\n")
print(round(reco_df.isnull().sum().sort_values(ascending=False) / len(reco_df.index), 4) * 100)

print("\nDuplicate Values in the Anime Dataset (%) : \n\n")
print(reco_df.duplicated().sum())

# Function to remove punctuation and convert text to lowercase
def remove_punctuation_and_turn_lower(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator).lower()

# Function to clean strings
def string_cleaning(string_list):
    strings_limpas = [remove_punctuation_and_turn_lower(string) for string in string_list]
    strings_limpas_no_numbers = [re.sub(r'\d', '', string) for string in strings_limpas]
    new_list = [item for item in strings_limpas_no_numbers if item]
    tokens_without_sw = [word for word in new_list if not word in stopwords.words('english')]
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in tokens_without_sw]
    return stemmed_words

# Treating the synopsis column
reco_df['Synopsis'] = reco_df['Synopsis'].apply(lambda x: x.split() if isinstance(x, str) else [])
reco_df['Synopsis'] = reco_df['Synopsis'].apply(string_cleaning)
print(reco_df.head(3))

# Applying bag of words technique
x_bow = reco_df['Synopsis'].apply(lambda x: ' '.join(x))

# Using CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=5000)
new_column = vectorizer.fit_transform(x_bow).toarray()

# Calculation of cosine similarity between the vectors
similarities = cosine_similarity(new_column)

# Function to get anime recommendation
def get_anime_recommendation(name):
    try:
        index = anime[anime['Name'] == name].index[0]
    except IndexError:
        print(f"Anime named '{name}' not found in the dataset.")
        return
    
    normal_list = similarities[index]
    ordenated_list = sorted(enumerate(normal_list), key=lambda x: x[1], reverse=True)
    
    print('Top 5 recommendations based on the selected anime:\n')
    for i, (idx, score) in enumerate(ordenated_list[1:6], start=1):
        print(f"{i}. {reco_df.iloc[idx]['Name']}")

# Example usage
get_anime_recommendation('Baki')
