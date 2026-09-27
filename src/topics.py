"""BERTopic on the cleaned tweets.

Run once as a script. It embeds the tweets if data/embeddings.npy is absent, which needs the model from
HuggingFace, then clusters them and writes the topic of every tweet to data/. Tweets left empty by cleaning,
mostly links only, share one embedding and are assigned to the outlier topic -1."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import config
import tweets


def embed(docs):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBEDDING).encode(docs, show_progress_bar=True).astype(np.float32)


def fit(docs, embeddings):
    """Topic of each document."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    model = BERTopic(umap_model=UMAP(n_neighbors=config.N_NEIGHBORS, n_components=config.N_COMPONENTS,
                                     min_dist=config.MIN_DIST, random_state=config.SEED),
                     hdbscan_model=HDBSCAN(min_cluster_size=config.MIN_CLUSTER_SIZE, min_samples=config.MIN_SAMPLES))
    return np.asarray(model.fit_transform(docs, embeddings=embeddings)[0])


def weighted_words(docs, labels, n):
    """The n words of largest class-based TF-IDF weight of each topic, computed as BERTopic computes them, outlier
    topic excluded. The representative words of the report are chosen among them."""
    from bertopic.vectorizers import ClassTfidfTransformer
    joined = pd.DataFrame({"doc": docs, "topic": labels}).groupby("topic")["doc"].agg(" ".join)
    vectorizer = CountVectorizer()
    counts_ = vectorizer.fit_transform([d or "emptydoc" for d in joined])
    weights = ClassTfidfTransformer().fit_transform(counts_).toarray()
    vocab = vectorizer.get_feature_names_out()
    return {t: list(vocab[np.argsort(-w, kind="stable")[:n]]) for t, w in zip(joined.index, weights) if t != -1}


def load():
    """Topic of each tweet from 2016, in the order of tweets.load()."""
    tw = tweets.load()
    frozen = pd.read_csv(config.TOPICS)
    if len(frozen) != len(tw) or (frozen["date"] != tw.index.strftime("%Y-%m-%d %H:%M")).any():
        raise ValueError("data/tweet_topics.csv does not match data/trump_tweets.csv")
    return frozen["topic"].to_numpy()


def counts(labels, session):
    """Tweets per trading session and topic, N_{j,t}, outlier topic -1 dropped, columns topic_0, topic_1, ...
    labels and session are in the order of tweets.load()."""
    df = pd.get_dummies(labels, dtype=float)
    df.index = session.to_numpy()
    df = df.drop(columns=-1, errors="ignore").groupby(level=0).sum()
    df.columns = [f"topic_{c}" for c in df.columns]
    return df


if __name__ == "__main__":
    tw = tweets.load()
    docs = np.array([tweets.clean(t) for t in tw["text"]], dtype=object)
    if not config.EMBEDDINGS.exists():
        np.save(config.EMBEDDINGS, embed(list(docs)))
    text = np.array([len(d.split()) > 0 for d in docs])
    labels = np.full(len(docs), -1)
    labels[text] = fit(list(docs[text]), np.load(config.EMBEDDINGS)[text])
    pd.DataFrame({"date": tw.index.strftime("%Y-%m-%d %H:%M"), "topic": labels}).to_csv(config.TOPICS, index=False)
