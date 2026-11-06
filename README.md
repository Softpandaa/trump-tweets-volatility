# Do Trump's Tweets Spur Market Volatility?

This study examines the daily association between Donald Trump's tweets and the CBOE Volatility Index (VIX) from January 2016 to January 2021. Each tweet is scored with VADER and assigned a topic with BERTopic, and an event regression relates the daily log change in the VIX in the sessions around each tweet to its sentiment and topic.

## Findings

Sessions with a tweet of extreme sentiment, which also carry more tweets, show a log change in the VIX 0.95% higher than other sessions, about 0.14 index points at the median VIX. Coefficients of later sessions are not significant, and among the 19 BERTopic topics only a few are, with signs that change across horizons.

Tweets on the coronavirus came after the VIX had risen and those on election fraud as it fell, consistent with tweets that comment on news rather than create it. At daily frequency, however, the response to a tweet can hardly be separated from the response to that news. Whether trading on the tweets pays is a question for intraday data.

## Layout

```
data/     committed tweets, prices, embeddings and topic labels
src/      analysis modules, every parameter declared once in config.py
report.pdf
```

The report is distributed as a compiled PDF. Its typesetting source is not included.

## Data

`data/trump_tweets.csv` is the [Trump's Tweets](https://www.kaggle.com/datasets/rishidamarla/trumps-tweets) dataset of Damarla (2021) on Kaggle. `data/prices.csv` is the daily VIX close from Yahoo Finance for January 2016 to January 2021. `data/embeddings.npy` is the all-MiniLM-L6-v2 sentence embedding of each cleaned tweet from 2016 onward, and `data/tweet_topics.csv` is the BERTopic topic of each tweet, as fitted by `src/topics.py`.

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
python src/plots.py
```

This prints every number quoted in the report and draws its five figures.
