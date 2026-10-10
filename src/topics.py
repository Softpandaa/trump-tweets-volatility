"""BERTopic on the cleaned tweets; run as a script to rebuild data/tweet_topics.csv."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import config
import tweets

STAMP = "%Y-%m-%d %H:%M"   # tweet time as written to data/tweet_topics.csv


def documents(tw):
    """Cleaned tweets and the mask of those not left empty by cleaning."""
    docs = np.array([tweets.clean(t) for t in tw["text"]], dtype=object)
    return docs, np.array([len(d.split()) > 0 for d in docs])


def embed(docs):
    from sentence_transformers import SentenceTransformer
    return (SentenceTransformer(config.EMBEDDING)
            .encode(docs, show_progress_bar=True).astype(np.float32))


def fit(docs, embeddings):
    """Topic of each document."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    model = BERTopic(umap_model=UMAP(n_neighbors=config.N_NEIGHBORS,
                                     n_components=config.N_COMPONENTS,
                                     min_dist=config.MIN_DIST, random_state=config.SEED),
                     hdbscan_model=HDBSCAN(min_cluster_size=config.MIN_CLUSTER_SIZE,
                                           min_samples=config.MIN_SAMPLES))
    return np.asarray(model.fit_transform(docs, embeddings=embeddings)[0])


def weighted_words(docs, labels, n):
    """Top n words of each topic by BERTopic's class-based TF-IDF, outlier topic excluded."""
    from bertopic.vectorizers import ClassTfidfTransformer
    joined = pd.DataFrame({"doc": docs, "topic": labels}).groupby("topic")["doc"].agg(" ".join)
    vectorizer = CountVectorizer()
    counts_ = vectorizer.fit_transform([d or "emptydoc" for d in joined])
    weights = ClassTfidfTransformer().fit_transform(counts_).toarray()
    vocab = vectorizer.get_feature_names_out()
    return {t: list(vocab[np.argsort(-w, kind="stable")[:n]])
            for t, w in zip(joined.index, weights) if t != -1}


def load():
    """Topic of each tweet, in the order of tweets.load()."""
    tw = tweets.load()
    frozen = pd.read_csv(config.TOPICS)
    if len(frozen) != len(tw) or (frozen["date"] != tw.index.strftime(STAMP)).any():
        raise ValueError("data/tweet_topics.csv does not match data/trump_tweets.csv")
    return frozen["topic"].to_numpy()


def counts(labels, session):
    """N_{j,t} of equation (4): tweets per session and topic, outlier topic -1 dropped."""
    df = pd.get_dummies(labels, dtype=float)
    df.index = session.to_numpy()
    df = df.drop(columns=-1, errors="ignore").groupby(level=0).sum()
    df.columns = [f"topic_{c}" for c in df.columns]
    return df


if __name__ == "__main__":
    tw = tweets.load()
    docs, text = documents(tw)
    if not config.EMBEDDINGS.exists():
        np.save(config.EMBEDDINGS, embed(list(docs)))
    labels = np.full(len(docs), -1)   # tweets left empty by cleaning stay in the outlier topic
    labels[text] = fit(list(docs[text]), np.load(config.EMBEDDINGS)[text])
    out = pd.DataFrame({"date": tw.index.strftime(STAMP), "topic": labels})
    out.to_csv(config.TOPICS, index=False)
