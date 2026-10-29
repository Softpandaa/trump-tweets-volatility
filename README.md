# Do Trump's Tweets Spur Market Volatility?

This study examines the daily association between Donald Trump's tweets and the CBOE Volatility Index (VIX) from January 2016 to January 2021. Each tweet is scored with VADER and assigned a topic with BERTopic, and an event regression relates the daily log change in the VIX in the sessions around each tweet to its sentiment and topic.

## Findings

Sessions with a tweet of extreme sentiment show a log change in the VIX 0.95% higher than in other sessions, about 0.14 index points, but the difference does not extend to later sessions. Among the 19 topics extracted by BERTopic, only a few coefficients are significant, and their signs change across horizons. The tweets on the coronavirus came after the VIX had risen and those on election fraud came as it fell, which is consistent with tweets that comment on news rather than create it. Nevertheless, the response to a tweet can hardly be separated from the response to the news it comments on at a daily frequency. A natural extension is to work on an intraday dataset and test whether trading or hedging strategies triggered by the tweets are profitable.

## Layout

```
data/     committed tweets, prices, embeddings and topic labels
src/      analysis modules, every parameter declared once in config.py
report.pdf
```

The report is distributed as a compiled PDF. Its typesetting source is not included.

## Data

`data/trump_tweets.csv` is the [Trump's Tweets](https://www.kaggle.com/datasets/rishidamarla/trumps-tweets) dataset of Damarla (2021) on Kaggle. `data/prices.csv` is the daily VIX close from Yahoo Finance over January 2016 to January 2021. `data/embeddings.npy` is the all-MiniLM-L6-v2 sentence embedding of each cleaned tweet from 2016, and `data/tweet_topics.csv` is the BERTopic topic of each tweet used in the report.

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
python src/plots.py
```

This prints every number quoted in the report and writes the five figures it uses to `latex/figures/`, which is created on first run and is not tracked.
