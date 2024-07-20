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
anime = anime.head(500)

# Display the dataframe
st.write("Displaying the dataframe:")
st.dataframe(anime)

# Preprocess the dataset
st.write("Preprocessing data...")
RecoDf = anime[['Name', 'Synopsis']]
RecoDf['Synopsis'] = RecoDf['Synopsis'].apply(lambda x: x.split())
RecoDf['Synopsis'] = RecoDf['Synopsis'].apply(string_cleaning)

# Display the preprocessed data
st.write("### Preprocessed Data")
st.dataframe(RecoDf)


# Apply bag of words technique
st.write("Applying bag of words...")
x_bow = RecoDf['Synopsis'].apply(lambda x: ' '.join(x))
vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=5000)
new_column = (vectorizer.fit_transform(x_bow)).toarray()

# Display the bag of words result
st.write("### Bag of Words Result")
st.write(pd.DataFrame(new_column, columns=vectorizer.get_feature_names_out()))

# Calculate cosine similarity
st.write("Calculating cosine similarity...")
similarities = cosine_similarity(new_column)

# Display a portion of the cosine similarity matrix
st.write("### Cosine Similarity Matrix (First 10 Rows and Columns)")
st.write(pd.DataFrame(similarities[:10, :10]))

# Define recommendation function
def get_anime_recommendation(name):
    st.write(f"### Process for: {name}")

    index = (anime.index[anime['Name'] == name])[0]
    st.write(f"Index of Selected Anime: {index}")

    normal_list = similarities[index]
    st.write("### Similarity Scores for Selected Anime")
    st.write(pd.DataFrame({
        'Index': range(len(normal_list)),
        'Similarity Score': normal_list
    }).sort_values(by='Similarity Score', ascending=False).reset_index(drop=True).head(10))

    ordenated_list = sorted(similarities[index], reverse=True)
    ranking_list = []
    for i in range(len(ordenated_list)):
        index_new = np.where(normal_list == ordenated_list[i])[0]
        tuple_anime = (int(index_new[0]), ordenated_list[i])
        ranking_list.append(tuple_anime)
    st.write("### Sorted Similarity Scores with Corresponding Indexes")
    st.write(pd.DataFrame({
        'Index': [item[0] for item in ranking_list],
        'Similarity Score': [item[1] for item in ranking_list]
    }).head(10))

    recommendations = []
    for tuple_values in ranking_list[1:6]:
        recommendations.append((RecoDf.iloc[tuple_values[0]]['Name'], tuple_values[1]))
    st.write("### Top 5 Recommendations")
    for rec, score in recommendations:
        st.write(f"{rec} (Similarity Score: {score})")
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
