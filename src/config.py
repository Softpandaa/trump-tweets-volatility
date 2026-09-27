"""Parameters of the study. Every constant is declared here once."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TWEETS = ROOT / "data" / "trump_tweets.csv"      # timestamps in UTC
PRICES = ROOT / "data" / "prices.csv"            # daily VIX close
EMBEDDINGS = ROOT / "data" / "embeddings.npy"    # all-MiniLM-L6-v2 embedding of each cleaned tweet from 2016
TOPICS = ROOT / "data" / "tweet_topics.csv"      # BERTopic topic of each tweet, from topics.py
FIG_DIR = ROOT / "latex" / "figures"

FIRST_YEAR = 2016             # tweets posted before this year are dropped
ELECTION = "2020-11-03"       # presidential election day, for the timing of topic tweets
OPEN_ET = 9 * 60 + 30         # VIX regular hours in minutes after midnight Eastern Time (ET), open
CLOSE_ET = 16 * 60 + 15       # and close, the time of the daily VIX close
LAGS = range(-3, 4)           # horizons k in r_{t+k}, event session t
SENTIMENT_CUTS = [0.5, 0.9]   # |compound| thresholds tabulated in the report
THRESHOLD = 0.9               # |compound| at or above this marks an extreme tweet
HAC_LAGS = 5                  # Newey-West lags
ACF_LAGS = 20                 # lags in the autocorrelation panel
TOP_WORDS = 6                 # most frequent words quoted in the report
CLOUD_WORDS = 80              # words in each word cloud
STOPWORDS_EXTRA = ["many", "more", "much", "would", "said", "say", "also", "what", "time", "done",
                   "thing", "see", "come", "work", "really", "want", "one", "make", "look", "best",
                   "amp"]

# BERTopic
EMBEDDING = "all-MiniLM-L6-v2"
N_NEIGHBORS = 8
N_COMPONENTS = 3
MIN_DIST = 0.05
MIN_CLUSTER_SIZE = 80
MIN_SAMPLES = 40
SEED = 0
WEIGHTED_WORDS = 20           # words of largest weight per topic, among which the report's five are chosen

# LDA and LSA, appendix only
LDA_GRID = range(3, 19)       # numbers of topics scanned by coherence
LDA_TOPICS = 11               # maximizes the coherence over LDA_GRID
LSA_TOPICS = 7
KEY_WORDS = 3                 # words that label a topic and enter its coherence
TSNE = dict(n_components=2, perplexity=50, learning_rate=100, max_iter=2000, angle=0.75)
