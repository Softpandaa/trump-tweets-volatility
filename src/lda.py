"""LDA and LSA on the most positive and the most negative tweet of the event sessions, reported in the appendix."""
import re
import string

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import config

STOP = set(stopwords.words("english")) | set(config.STOPWORDS_EXTRA)
LEMMA = WordNetLemmatizer()


def lemmatize(word):
    """WordNet lemma, except that media is kept rather than read as the plural of medium."""
    return word if word == "media" else LEMMA.lemmatize(word)


def clean(text):
    """Lower case, punctuation, digits and stop words removed, words lemmatized, stop words removed again
    after lemmatizing."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    for ch in ["“", "”", "’"]:
        text = text.replace(ch, " ")
    text = text.replace("➡️", "").replace("american", "america").replace("united states", "america")
    text = re.sub("[0-9]+", "", text)
    words = (lemmatize(w) for w in text.split() if w not in STOP)
    return " ".join(w for w in words if w not in STOP)


def corpus(ext):
    """Most negative tweet of each session with one at or below -THRESHOLD, then most positive tweet of each
    session with one at or above THRESHOLD, each in session order."""
    neg = ext.loc[ext["neg"] <= -config.THRESHOLD, "neg_text"]
    pos = ext.loc[ext["pos"] >= config.THRESHOLD, "pos_text"]
    return [clean(t) for t in pd.concat([neg, pos]).dropna()]


def key_words(dtm, vocab, labels, k):
    """The KEY_WORDS most frequent words among the documents assigned to each topic."""
    out = []
    for topic in range(k):
        counts = np.asarray(dtm[labels == topic].sum(axis=0)).ravel()
        out.append([vocab[i] for i in np.argsort(counts)[::-1][:config.KEY_WORDS] if counts[i] > 0])
    return out


def fit(docs, k, model="lda"):
    vec = CountVectorizer(stop_words="english")
    dtm = vec.fit_transform(docs)
    if model == "lda":
        est = LatentDirichletAllocation(n_components=k, learning_method="online", random_state=config.SEED)
    else:
        est = TruncatedSVD(n_components=k, random_state=config.SEED)
    weights = est.fit_transform(dtm)
    labels = weights.argmax(axis=1)
    return weights, labels, key_words(dtm, vec.get_feature_names_out(), labels, k)


def coherence(docs, words):
    """c_v coherence of the key words of the topics that hold at least one document, over the tokenized
    documents."""
    analyzer = CountVectorizer(stop_words="english").build_analyzer()
    texts = [analyzer(d) for d in docs]
    return CoherenceModel(topics=[w for w in words if w], texts=texts, dictionary=Dictionary(texts), coherence="c_v").get_coherence()


def scan(docs):
    return pd.Series({k: coherence(docs, fit(docs, k)[2]) for k in config.LDA_GRID})


def tsne(weights):
    """Planar t-SNE coordinates."""
    return TSNE(random_state=config.SEED, **config.TSNE).fit_transform(weights)
