import pickle
with open('./models/spam_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
print(type(vectorizer))  # Should print <class 'sklearn.feature_extraction.text.CountVectorizer'>
