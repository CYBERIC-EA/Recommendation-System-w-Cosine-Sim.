# Recommendation-System-w-Cosine-Sim.

Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space. It is calculated as the cosine of the angle between the two vectors. The cosine similarity is particularly used in high-dimensional positive spaces where the magnitude of the vectors does not matter.

The formula for cosine similarity between two vectors A and B is:
cosine similarity= cos(θ) = A⋅B / ∥A∥∥B∥

​
Where:

𝐴
⋅
𝐵
is the dot product of vectors A and B.

∥
𝐴
∥
and 
∥
𝐵
∥
are the magnitudes of vectors A and B, respectively.
​

The workflow is;

Preprocessing: Clean and tokenize the text data (anime synopses).

Vectorization: Convert the cleaned text into numerical vectors using CountVectorizer.

Similarity Calculation: Compute the pairwise cosine similarity between the vectors representing the synopses.


# Consider two synopses:

Synopsis A: "Death Note is a psychological thriller"

Synopsis B: "A psychological thriller about Death Note"

After vectorization, the document-term matrix might look something like this (simplified example):


![table](https://github.com/user-attachments/assets/0cc1caa7-6ae1-45b4-bc6c-37504c72de3e)

  


Then, the cosine similarity between these vectors would be calculated. If they are very similar, the cosine similarity value will be close to 1.


# recommendation

Identify the Index: Find the index of the given anime in the dataset.

Similarity Scores: Retrieve the similarity scores for that anime with all other anime.

Ranking: Sort the similarity scores in descending order.

Recommendations: Extract the top 5 most similar anime based on the similarity scores. 
