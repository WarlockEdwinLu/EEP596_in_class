import os
import gdown
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

#  Google Drive File ID for GloVe embeddings (Replace with your actual File ID)
GOOGLE_DRIVE_FILE_ID = "1-BPA4eerIHEIzvrjty-Cqr2BJNsLw_ON"

#  Path to store the embeddings
file_path = "embeddings/glove.6B.300d.txt"
os.makedirs("embeddings", exist_ok=True)

#  Check if the file exists, if not, download it
if not os.path.exists(file_path):
    st.info("Downloading GloVe embeddings from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", file_path, quiet=False)
else:
    st.success("GloVe embeddings already exist.")

#  Load GloVe embeddings
class Embeddings:
    def __init__(self):
        self.drive_path = "embeddings"

    def load_glove_embeddings(self, embedding_dimension=300):
        file_path = os.path.join(self.drive_path, "glove.6B.300d.txt")
        embeddings_dict = {}
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype="float32")
                    if len(vector) == embedding_dimension:
                        embeddings_dict[word] = vector
            return embeddings_dict
        except FileNotFoundError:
            st.error("GloVe embeddings file not found! Please download and place it in the 'embeddings' folder.")
            return {}

#  Define the Search Class
class Search:
    def __init__(self, embeddings_dict):
        self.embeddings_dict = embeddings_dict
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def cosine_similarity(self, x, y):
        return np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-3)

    def get_sentence_embedding(self, sentence):
        return self.model.encode(sentence)

    def get_topK_similar_categories(self, sentence, categories):
        sentence_embedding = self.get_sentence_embedding(sentence)
        category_scores = {}
        for category in categories:
            category_embedding = self.get_sentence_embedding(category)
            similarity = self.cosine_similarity(sentence_embedding, category_embedding)
            category_scores[category] = similarity
        return dict(sorted(category_scores.items(), key=lambda x: x[1], reverse=True))

#  Load embeddings and initialize Search class
st.title(" Semantic Search Web App")
st.write("Enter a sentence below to find its closest semantic categories.")

embeddings = Embeddings()
embeddings_dict = embeddings.load_glove_embeddings()
search = Search(embeddings_dict)

#  User input
input_text = st.text_input("Enter your sentence:")
categories = ["Flowers", "Colors", "Cars", "Weather", "Food"]

if input_text:
    category_scores = search.get_topK_similar_categories(input_text, categories)

    st.subheader(" Results:")
    for category, score in category_scores.items():
        st.write(f"**{category}:** {score:.4f}")

    #  Display as a bar chart
    st.subheader(" Category Similarity Scores")
    st.bar_chart(category_scores)
