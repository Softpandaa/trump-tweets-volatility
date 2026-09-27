"""Tweets: loading, cleaning, VADER sentiment and trading sessions."""
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config

URL = re.compile(r"https?://\S+|www\.\S+")
STOP = set(stopwords.words("english")) | set(config.STOPWORDS_EXTRA)


def load(all_years=False):
    """Tweets indexed by UTC timestamp, from FIRST_YEAR with links removed unless all_years."""
    raw = pd.read_csv(config.TWEETS, index_col=0)
    tw = pd.DataFrame({"text": raw["text"].astype(str)})
    tw.index = pd.to_datetime(tw.index, format="%m/%d/%Y %H:%M")
    if not all_years:
        tw = tw[tw.index.year >= config.FIRST_YEAR].copy()
        tw["text"] = [URL.sub("", t) for t in tw["text"]]
    return tw


def daily_counts(tw):
    """Tweets per calendar day in ET, days without a tweet included as zero."""
    et = tw.index.tz_localize("UTC").tz_convert("America/New_York")
    counts = pd.Series(1, index=pd.to_datetime(et.date)).groupby(level=0).sum()
    return counts.reindex(pd.date_range(counts.index[0], counts.index[-1]), fill_value=0)


def et_minutes(tw):
    """Minutes after midnight Eastern Time (ET) at which each tweet was posted."""
    et = tw.index.tz_localize("UTC").tz_convert("America/New_York")
    return pd.Series(et.hour * 60 + et.minute, index=tw.index)


def sessions(tw, days):
    """Trading session of each tweet, the first session whose VIX close falls after the tweet.

    days are the trading dates. A tweet stamped at the close minute is assigned to the next session."""
    days = pd.DatetimeIndex(days)
    et = tw.index.tz_localize("UTC").tz_convert("America/New_York")
    closes = days.tz_localize("America/New_York") + pd.Timedelta(minutes=config.CLOSE_ET)
    pos = closes.searchsorted(et, side="right")
    if (pos == len(days)).any():
        raise ValueError("tweets after the last trading close")
    return pd.Series(days[pos], index=tw.index, name="session")


def clean(text):
    """Lower case, punctuation, stop words and digits removed, a few merges. Used for word counts, word clouds and
    BERTopic, and kept as in the 2024 study because data/embeddings.npy was computed on its output."""
    text = "".join(c.lower() for c in text if c not in string.punctuation)
    text = text.replace("’", " ").replace("➡️", "")
    text = " ".join(w for w in text.split() if w not in STOP)
    for old, new in [("american", "america"), ("unit state", "america"), ("fake news", "fake_news"),
                     (" i ", " "), (" u ", " "), (" it ", " "), (" k ", " ")]:
        text = text.replace(old, new)
    text = re.sub("[0-9]+", "", text)
    return text.replace("  ", " ")


def top_words(texts, n):
    """Most frequent words, ties in order of first appearance."""
    counts = sorted(Counter(" ".join(texts).split()).items(), key=lambda x: x[1], reverse=True)
    return pd.Series(dict(counts[:n]))


def scores(tw):
    """VADER compound score of each tweet."""
    analyzer = SentimentIntensityAnalyzer()
    return pd.Series([analyzer.polarity_scores(t)["compound"] for t in tw["text"]], index=tw.index)


def extreme_sessions(score, session):
    """Indicator D_t of a session carrying a tweet with |compound| at or above THRESHOLD."""
    hit = pd.Series(score.abs().to_numpy() >= config.THRESHOLD, index=session.to_numpy())
    return hit.groupby(level=0).any().astype(float).rename("D")


def extremes(tw, score, session):
    """Most positive and most negative tweet of each session, scores floored at zero, last tweet on ties."""
    df = pd.DataFrame({"text": tw["text"].to_numpy(), "score": score.to_numpy()}, index=session.to_numpy())
    rows = []
    for day, g in df.groupby(level=0, sort=True):
        s = g["score"].to_numpy()
        hi, lo = len(s) - 1 - s[::-1].argmax(), len(s) - 1 - s[::-1].argmin()
        rows.append((day, max(s[hi], 0.0), g["text"].iloc[hi] if s[hi] >= 0 else np.nan,
                     min(s[lo], 0.0), g["text"].iloc[lo] if s[lo] <= 0 else np.nan))
    return pd.DataFrame(rows, columns=["session", "pos", "pos_text", "neg", "neg_text"]).set_index("session")
