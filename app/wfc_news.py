import warnings
warnings.filterwarnings("ignore")

import re
import datetime
import nltk
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class wfc_feed:
    """Fetch news for a ticker via yfinance and score sentiment with VADER.

    Each article dict contains:
        title     - headline text
        summary   - article summary (falls back to title if unavailable)
        url       - link to original source
        publisher - news outlet name
        published - formatted date string
        pos / neg / neu / compound - VADER sentiment scores
    """

    def __init__(self, ticker):
        self.ticker   = ticker
        self.articles = []
        self._sia     = SentimentIntensityAnalyzer()

    def get_news(self):
        news = yf.Ticker(self.ticker).news or []
        for item in news:
            # yfinance >= 0.2.50 nests everything under item['content']
            c = item.get('content') or item

            title = c.get('title', '').strip()
            if not title:
                continue

            # summary: plain text preferred; strip HTML tags if needed
            raw_summary = c.get('summary') or c.get('description') or ''
            summary = re.sub(r'<[^>]+>', '', raw_summary).strip() or title

            # URL
            url = (c.get('canonicalUrl') or c.get('clickThroughUrl') or {}).get('url', '')
            if not url:
                url = c.get('link', '')

            # publisher
            publisher = (c.get('provider') or {}).get('displayName', '') or c.get('publisher', '')

            # publish date
            pub_raw = c.get('pubDate') or c.get('displayTime') or ''
            if pub_raw:
                try:
                    dt = datetime.datetime.fromisoformat(pub_raw.replace('Z', '+00:00'))
                    published = dt.strftime('%Y-%m-%d %H:%M')
                except ValueError:
                    published = pub_raw
            else:
                ts = c.get('providerPublishTime', 0)
                published = (datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
                             if ts else '')

            scores = self._sia.polarity_scores(title)
            self.articles.append({
                'title':     title,
                'summary':   summary,
                'url':       url,
                'publisher': publisher,
                'published': published,
                'pos':       scores['pos'],
                'neg':       scores['neg'],
                'neu':       scores['neu'],
                'compound':  scores['compound'],
            })
        self.articles.sort(key=lambda a: a['published'], reverse=True)

    # ------------------------------------------------------------------ totals

    def _totals(self):
        pos = sum(a['pos'] for a in self.articles)
        neg = sum(a['neg'] for a in self.articles)
        neu = sum(a['neu'] for a in self.articles)
        return pos, neg, neu

    def get_pos(self):
        return self._totals()[0]

    def get_neg(self):
        return self._totals()[1]

    def get_neu(self):
        return self._totals()[2]

    # -------------------------------------------------------------- filtering

    def get_headlines(self, sentiment=None, n=5):
        """Return up to n articles sorted by compound score.

        sentiment: 'positive' | 'negative' | None (returns all sorted by date)
        """
        if sentiment == 'positive':
            filtered = sorted(self.articles, key=lambda a: a['compound'], reverse=True)
        elif sentiment == 'negative':
            filtered = sorted(self.articles, key=lambda a: a['compound'])
        else:
            filtered = self.articles
        return filtered[:n]

    def overall_sentiment(self):
        """Return 'positive', 'negative', or 'neutral' based on aggregated scores."""
        if not self.articles:
            return 'neutral'
        pos, neg, neu = self._totals()
        if pos > neg and pos > neu:
            return 'positive'
        if neg > pos and neg > neu:
            return 'negative'
        return 'neutral'

    # ---------------------------------------------------------------- display

    def show_summary(self, ticker=None):
        label = ticker or self.ticker
        pos, neg, neu = self._totals()
        total = len(self.articles)
        overall = self.overall_sentiment()
        print(f"\n{label}  ({total} articles)  overall: {overall.upper()}")
        print(f"  Positive: {pos:.2f}  Negative: {neg:.2f}  Neutral: {neu:.2f}")
        if not self.articles:
            return
        print("\n  Top Positive Headlines:")
        for a in self.get_headlines('positive', 3):
            print(f"    [{a['compound']:+.2f}] {a['title']}")
            if a['url']:
                print(f"           {a['url']}")
        print("\n  Top Negative Headlines:")
        for a in self.get_headlines('negative', 3):
            print(f"    [{a['compound']:+.2f}] {a['title']}")
            if a['url']:
                print(f"           {a['url']}")
