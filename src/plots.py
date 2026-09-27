"""Single entry point. Prints every number quoted in the report and writes the figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import stats
from statsmodels.tsa.stattools import acf
from wordcloud import WordCloud

import config
import lda
import regression
import topics
import tweets

OKABE = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73", "orange": "#E69F00",
         "purple": "#CC79A7", "sky": "#56B4E9", "black": "#000000"}
DASHED = (0, (5, 2))
MARKERS = ["o", "^", "s"]

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(config.FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def year_axis(ax):
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)


def figure_tweets(tw_all, minutes):
    monthly = tweets.daily_counts(tw_all).resample("MS").mean()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
    axes[0].plot(monthly, color=OKABE["blue"], linewidth=1.0)
    axes[0].set_ylabel("Tweets per day")
    year_axis(axes[0])
    axes[1].hist(minutes / 60, bins=np.arange(0, 24.25, 0.25), color=OKABE["blue"], linewidth=0)
    for edge in [config.OPEN_ET / 60, config.CLOSE_ET / 60]:
        axes[1].axvline(edge, color=OKABE["vermillion"], linestyle=DASHED, linewidth=1.2)
    axes[1].set_xticks(range(0, 25, 3))
    axes[1].set_xlim(0, 24)
    axes[1].set_xlabel("Hour of the day, ET")
    axes[1].set_ylabel("Number of tweets")
    fig.tight_layout()
    save(fig, "tweets.png")


def figure_vix(vix, r):
    """VIX close over time, histogram of the daily log change, and autocorrelation of the change and of its
    absolute value."""
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.2))
    axes[0].plot(vix, color=OKABE["blue"], linewidth=0.8)
    axes[0].set_ylabel("VIX close")
    year_axis(axes[0])
    axes[1].hist(r, bins=60, color=OKABE["blue"], linewidth=0)
    axes[1].set_xlabel("$r_t$")
    axes[1].set_ylabel("Number of sessions")
    lags = np.arange(1, config.ACF_LAGS + 1)
    band = 1.96 / np.sqrt(len(r))
    for series, label, colour, marker, shift in [(r, "$r_t$", "blue", "o", -0.15),
                                                  (r.abs(), "$|r_t|$", "vermillion", "^", 0.15)]:
        rho = acf(series, nlags=config.ACF_LAGS)[1:]
        axes[2].vlines(lags + shift, 0, rho, color=OKABE[colour], linewidth=1.0)
        axes[2].plot(lags + shift, rho, marker, color=OKABE[colour], markersize=2.5, label=label)
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].fill_between([0.5, config.ACF_LAGS + 0.5], -band, band, color=OKABE["black"], alpha=0.1, linewidth=0)
    axes[2].set_xticks([1, 5, 10, 15, 20])
    axes[2].set_xlabel("Lags")
    axes[2].set_ylabel("Autocorrelation")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    save(fig, "vix.png")


def figure_clouds(ext):
    font = font_manager.findfont(font_manager.FontProperties(family="serif"))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.0))
    for ax, col in zip(axes, ["pos", "neg"]):
        sel = ext[col] >= config.THRESHOLD if col == "pos" else ext[col] <= -config.THRESHOLD
        text = " ".join(tweets.clean(t) for t in ext.loc[sel, f"{col}_text"].dropna())
        cloud = WordCloud(max_words=config.CLOUD_WORDS, background_color="white", font_path=font,
                          random_state=config.SEED, width=800, height=500).generate(text)
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
    fig.tight_layout()
    save(fig, "word_clouds.png")


def figure_lda(scan, maps):
    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    ax.plot(scan.index, scan.to_numpy(), color=OKABE["blue"], marker="o", markersize=4, linewidth=1.2)
    ax.plot(config.LDA_TOPICS, scan[config.LDA_TOPICS], "o", color=OKABE["vermillion"], markersize=7)
    ax.set_xlabel("Number of topics $K$")
    ax.set_ylabel("Coherence $C_V$")
    save(fig, "lda_coherence.png")

    hues = [OKABE[c] for c in ["blue", "vermillion", "green", "orange", "purple", "sky", "black"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 5.4))
    for ax, title, (xy, labels, words) in zip(axes, [f"LSA, {config.LSA_TOPICS} topics",
                                                     f"LDA, {config.LDA_TOPICS} topics"], maps):
        ax.set_title(title)
        sizes = np.bincount(labels, minlength=len(words))
        for rank, t in enumerate(np.argsort(-sizes, kind="stable")):   # larger topics get the first colours
            pts = xy[labels == t]
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=6, color=hues[rank % len(hues)],
                           marker=MARKERS[rank // len(hues)], alpha=0.7, linewidths=0,
                           label=f"{t}: {' '.join(words[t])}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=8, frameon=False,
                  markerscale=1.8, handletextpad=0.2)
    fig.tight_layout()
    save(fig, "tsne.png")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def stars(p):
    return "***" if p <= 0.01 else "**" if p <= 0.05 else "*" if p <= 0.1 else ""


def latex_row(name, coef, pval):
    """One table row, coefficients to two decimals with stars from the Newey-West p-value."""
    cells = [f"${coef[k]:.2f}^{{{stars(pval[k])}}}$".replace("^{}", "") for k in coef.index]
    return f"{name} & " + " & ".join(cells) + " \\\\"


def main():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(config.PRICES, index_col=0, parse_dates=True)
    vix = prices["VIX_close"]
    r = regression.vix_change(prices)

    # Data, Table A.1 and Table 1
    tw_all = tweets.load(all_years=True)
    tw = tweets.load()
    per_year = tweets.daily_counts(tw_all)
    print(f"Tweets in the file {len(tw_all):,}, by ET year", per_year.groupby(per_year.index.year).sum().to_dict())
    print(f"Kept from {config.FIRST_YEAR}: {len(tw):,}, {tw.index[0]:%d %b %Y} to {tw.index[-1]:%d %b %Y}, "
          f"{tweets.daily_counts(tw).mean():.2f} per calendar day")
    minutes = tweets.et_minutes(tw)
    session = tweets.sessions(tw, prices.index)
    et_date = pd.to_datetime(tw.index.tz_localize("UTC").tz_convert("America/New_York").date)
    trading = et_date.isin(prices.index)
    before, after = minutes < config.OPEN_ET, minutes >= config.CLOSE_ET
    print(f"On a trading day before the open {pct((trading & before).mean())}, during VIX regular hours "
          f"{pct((trading & ~before & ~after).mean())}, after the close {pct((trading & after).mean())}; "
          f"on a non-trading day {pct((~trading).mean())}")
    per_session = session.value_counts()
    print(f"Sessions {len(prices):,}, {prices.index[0]:%d %b %Y} to {prices.index[-1]:%d %b %Y}, with a tweet "
          f"{len(per_session):,}, tweets per session with a tweet {per_session.mean():.2f}, median {per_session.median():.0f}")
    print("Most frequent words", list(tweets.top_words([tweets.clean(t) for t in tw["text"]], config.TOP_WORDS).index))
    print(f"VIX peak {vix.max():.2f} on {vix.idxmax():%d %b %Y}")
    for name, x in [("V_t", vix), ("r_t", r)]:
        print(f"  {name}: mean {x.mean():.3g}, median {x.median():.3g}, sd {x.std():.3g}, min {x.min():.3g}, "
              f"max {x.max():.3g}, skewness {stats.skew(x):.3g}, excess kurtosis {stats.kurtosis(x):.3g}, "
              f"autocorrelation at lag 1 {acf(x, nlags=1)[1]:.3f}")
    rho_abs = acf(r.abs(), nlags=config.ACF_LAGS)[1:]
    print(f"Autocorrelation of |r_t| at lags 1 to 10 {np.round(rho_abs[:10], 3).tolist()}, band {1.96 / np.sqrt(len(r)):.3f}")

    # Sentiment, Table 2 and Table 3
    score = tweets.scores(tw)
    for c in config.SENTIMENT_CUTS:
        pos, neg = int((score >= c).sum()), int((score <= -c).sum())
        print(f"  |s| >= {c}: positive {pos:,}, neutral {len(score) - pos - neg:,}, negative {neg:,}, "
              f"polar {pct((pos + neg) / len(score), 0)}")
    D = tweets.extreme_sessions(score, session).reindex(prices.index).fillna(0.0)
    print(f"P(D_t+1 = 1 | D_t = 1) {D.shift(-1)[D == 1].mean():.2f} against P(D = 1) {D.mean():.2f}")
    coef, pval = regression.event_regression(r, D.to_frame())
    print("  " + latex_row("$\\phi_k$", coef.loc["lag"], pval.loc["lag"]))
    print("  " + latex_row("$\\gamma_k$", coef.loc["D"], pval.loc["D"]))
    g = coef.loc["D", 0]
    print(f"gamma_0 {g:.2f}: {g / 100 * vix.median():.2f} index points at the median VIX, "
          f"{pct(np.exp(2 * g / 100) - 1)} in variance")

    # Topics, Table 4 and Table 5
    labels = topics.load()
    docs = np.array([tweets.clean(t) for t in tw["text"]], dtype=object)
    text = np.array([len(d.split()) > 0 for d in docs])
    size = pd.Series(labels).value_counts()
    for j, w in topics.weighted_words(list(docs[text]), labels[text], config.WEIGHTED_WORDS).items():
        print(f"  Topic {j}, {size[j]} tweets, words of largest weight: {', '.join(w)}")
    N = topics.counts(labels, session).reindex(prices.index).fillna(0.0)
    coef, pval = regression.event_regression(r, N)
    print("  " + latex_row("$\\phi_k$", coef.loc["lag"], pval.loc["lag"]))
    for j, row in enumerate(N.columns):
        print("  " + latex_row(str(j), coef.loc[row], pval.loc[row]))
    election = pd.Timestamp(config.ELECTION)
    for name, date in [("election day", election), ("VIX peak", vix.idxmax())]:
        share = pd.Series(et_date > date, index=labels).groupby(level=0).mean().drop(-1)
        print(f"Share of each topic's tweets posted after the {name} {date:%d %b %Y}:", share.round(2).to_dict())
    print(f"VIX close on election day {vix[election]:.2f}, in the session of the last tweet "
          f"{session.iloc[-1]:%d %b %Y} {vix[session.iloc[-1]]:.2f}")

    # LDA appendix, Figure B.1, Table B.1 and Figure B.2
    ext = tweets.extremes(tw, score, session)
    corpus = lda.corpus(ext)
    scan = lda.scan(corpus)
    print(f"LDA corpus {len(corpus)} tweets; coherence " + ", ".join(f"{k} {v:.3f}" for k, v in scan.items()))
    maps = []
    for model, k in [("lsa", config.LSA_TOPICS), ("lda", config.LDA_TOPICS)]:
        weights, lab, words_k = lda.fit(corpus, k, model)
        maps.append((lda.tsne(weights), lab, words_k))
        print(f"{model.upper()} {k} topics")
        for t, w in enumerate(words_k):
            print(f"  {t}: {int((lab == t).sum())} {' '.join(w)}")

    figure_tweets(tw_all, minutes)
    figure_vix(vix, r)
    figure_clouds(ext)
    figure_lda(scan, maps)


if __name__ == "__main__":
    main()
