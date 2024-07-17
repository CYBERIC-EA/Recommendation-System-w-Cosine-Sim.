import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download NLTK stopwords
nltk.download('stopwords')

# Define preprocessing functions
def remove_punctuation_and_turn_lower(text):
    translator = str.maketrans('', '', string.punctuation)
    return (text.translate(translator)).lower()

def string_cleaning(string_list):
    strings_limpas = [remove_punctuation_and_turn_lower(string) for string in string_list]
    strings_limpas_no_numbers = [re.sub(r'\d', '', string) for string in strings_limpas]
    new_list = [item for item in strings_limpas_no_numbers if item]
    tokens_without_sw = [word for word in new_list if not word in stopwords.words('english')]
    ps = PorterStemmer()
    steemed_words = [ps.stem(w) for w in tokens_without_sw]
    return steemed_words

# Load dataset
animePath = 'anime-dataset-2023.csv'
anime = pd.read_csv(animePath)
anime = anime.head(1000)

# Display the dataframe
st.write("Displaying the dataframe:")
st.dataframe(anime)

# Preprocess the dataset
st.write("Preprocessing data...")
RecoDf = anime[['Name', 'Synopsis']]
RecoDf['Synopsis'] = RecoDf['Synopsis'].apply(lambda x: x.split())
RecoDf['Synopsis'] = RecoDf['Synopsis'].apply(string_cleaning)

# Apply bag of words technique
st.write("Applying bag of words...")
x_bow = RecoDf['Synopsis'].apply(lambda x: ' '.join(x))
vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=5000)
new_column = (vectorizer.fit_transform(x_bow)).toarray()

# Calculate cosine similarity
st.write("Calculating cosine similarity...")
similarities = cosine_similarity(new_column)

# Define recommendation function
def get_anime_recommendation(name):
    index = (anime.index[anime['Name'] == name])[0]
    normal_list = similarities[index]
    ordenated_list = sorted(similarities[index], reverse=True)
    ranking_list = []
    for i in range(len(ordenated_list)):
        index_new = np.where(normal_list == ordenated_list[i])[0]
        tuple_anime = (int(index_new[0]), ordenated_list[i])
        ranking_list.append(tuple_anime)
    recommendations = []
    for tuple_values in ranking_list[1:6]:
        recommendations.append(RecoDf.iloc[tuple_values[0]]['Name'])
    return recommendations

# Streamlit UI
st.title("Anime Recommendation System")
st.write("This application provides anime recommendations based on the synopsis of a selected anime.")

selected_anime = st.selectbox("Select an Anime", anime['Name'].values)

if st.button("Get Recommendations"):
    st.write("Fetching recommendations...")
    recommendations = get_anime_recommendation(selected_anime)
    st.write("Top 5 recommendations based on the selected anime:")
    for rec in recommendations:
        st.write(rec)

st.write("Done!")
