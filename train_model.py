import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# LOAD DATASET
data = pd.read_csv("dataset/dataset.csv")

X = data['text']
y = data['label']

# VECTORIZATION (ubah teks ke angka)
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# MODEL NAIVE BAYES
model = MultinomialNB()
model.fit(X_vec, y)

# SIMPAN MODEL
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model berhasil disimpan!")