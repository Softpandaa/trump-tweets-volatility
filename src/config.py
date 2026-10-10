"""Parameters of the study, each declared once."""
from pathlib import Path

# Paths -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
TWEETS = ROOT / "data" / "trump_tweets.csv"      # timestamps in UTC
PRICES = ROOT / "data" / "prices.csv"            # daily VIX close
EMBEDDINGS = ROOT / "data" / "embeddings.npy"    # one row per cleaned tweet from FIRST_YEAR
TOPICS = ROOT / "data" / "tweet_topics.csv"      # written by topics.py
FIG_DIR = ROOT / "latex" / "figures"

# Sample ----------------------------------------------------------------------
FIRST_YEAR = 2016
ELECTION = "2020-11-03"
OPEN_ET = 9 * 60 + 30         # VIX regular hours, minutes after midnight ET
CLOSE_ET = 16 * 60 + 15       # time of the daily VIX close
ET = "America/New_York"

# Sentiment and regression ----------------------------------------------------
LAGS = range(-3, 4)           # horizons k in r_{t+k}, event session t
SENTIMENT_CUTS = [0.5, 0.9]   # |compound| cut-offs of Table 2
THRESHOLD = 0.9               # theta of D_t, equation (2)
HAC_LAGS = 5                  # Newey-West lags
ACF_LAGS = 20                 # autocorrelation panel

# Words -----------------------------------------------------------------------
TOP_WORDS = 6                 # most frequent words quoted in the report
CLOUD_WORDS = 80
STOPWORDS_EXTRA = ["many", "more", "much", "would", "said", "say", "also", "what", "time", "done",
                   "thing", "see", "come", "work", "really", "want", "one", "make", "look", "best",
                   "amp"]

# BERTopic --------------------------------------------------------------------
EMBEDDING = "all-MiniLM-L6-v2"
N_NEIGHBORS = 8
N_COMPONENTS = 3
MIN_DIST = 0.05
MIN_CLUSTER_SIZE = 80
MIN_SAMPLES = 40
SEED = 0                      # also LDA, LSA, t-SNE and the word clouds
WEIGHTED_WORDS = 20           # per topic; the report quotes five of them

# LDA and LSA (appendix) ------------------------------------------------------
LDA_GRID = range(3, 19)       # numbers of topics scanned by coherence
LDA_TOPICS = 11               # K of Table B.1
LSA_TOPICS = 7
KEY_WORDS = 3                 # words that label a topic and enter its coherence
TSNE = dict(n_components=2, perplexity=50, learning_rate=100, max_iter=2000, angle=0.75)
