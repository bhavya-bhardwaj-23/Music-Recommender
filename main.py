import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

df = pd.read_csv("C:/Users/nachi/Downloads/archive/songdata.csv")
df = df.sample(n=5000).drop('link', axis=1).reset_index(drop=True)

df['text'] = df['text'].str.lower().replace(r'[^\s\w]', '').replace(r'[\n]', '', regex=True)

ps = PorterStemmer()


def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [ps.stem(w) for w in tokens]

    return " ".join(stemming)


df['text'] = df['text'].apply(lambda x: tokenization(x))

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
matrix = tfidf_vectorizer.fit_transform(df['text'])
similarity = cosine_similarity(matrix)


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


def enhance_similarity(song_tokens, similarity_matrix, threshold=0.2):
    enhanced_similarity = np.zeros(similarity_matrix.shape[0])

    for token in song_tokens:
        synonyms = get_synonyms(token)
        for syn in synonyms:
            if syn in tfidf_vectorizer.vocabulary_:
                syn_index = tfidf_vectorizer.vocabulary_[syn]
                if syn_index < similarity_matrix.shape[1]:
                    enhanced_similarity += similarity_matrix[:, syn_index]

    enhanced_similarity /= len(song_tokens)  # Normalize by the number of tokens
    enhanced_similarity[enhanced_similarity >= threshold] = 1  # Threshold for enhanced similarity
    return enhanced_similarity





def recommendation(song_df):
    idx = df[df['song'] == song_df].index[0]

    song_tokens = nltk.word_tokenize(tokenization(df.iloc[idx]['text']))
    enhanced_similarity = enhance_similarity(song_tokens, similarity)

    distances = sorted(list(enumerate(enhanced_similarity)), reverse=True, key=lambda x: x[1])

    songs = []
    for m_id in distances[1:21]:
        songs.append(df.iloc[m_id[0]].song)

    return songs


songss = input("enter the song name: ")
print(recommendation(songss))
print()

# pickle.dump(df, open('df.pkl', 'rb'))
# pickle.dump(similarity, open('similarity.pkl', 'rb'))
