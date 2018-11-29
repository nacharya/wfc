import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import feedparser
import urllib
import nltk

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pandas as pd

class wfc_feed:

    def __init__(self, ticker):
        nasdaq_url = "http://articlefeeds.nasdaq.com/nasdaq/symbols?symbol="
        self.newsurls = {
            ticker :             nasdaq_url+ticker
        }
        self.a_headlines = []
        self.ticker = ticker
        self.rdr = None

    def get_headlines(self, rss_url ):
        headlines = []
        feed = feedparser.parse( rss_url )
        for newsitem in feed['items']:
            headlines.append(newsitem['title'])
        return headlines

    def get_news(self):
        for key,url in self.newsurls.items():
            self.a_headlines.extend( self.get_headlines( url ) )
        self.rdr = wfc_read(self.a_headlines)
        self.rdr.qsenti()
        x = min(self.rdr.pos, self.rdr.neu, self.rdr.neg)
        z = max(self.rdr.pos, self.rdr.neu, self.rdr.neg)
        y = ( self.rdr.pos + self.rdr.neu + self.rdr.neg) - (x + z)
        #print("pos: " + str(rdr.pos) + "\tneu: " + str(rdr.neu) + "\tneg: " + str(rdr.neg) + \
        #        "\t[" + str(x) + "\t" + str(y) + "\t" + str(z) + "]")

    def get_pos(self):
        return self.rdr.pos
    def get_neu(self):
        return self.rdr.neu
    def get_neg(self):
        return self.rdr.neg
    def senti(self, level):
        return self.rdr.sentiment(level)

    def show_summary(self,ticker):
        print(ticker + "\t+ve: " + str(self.get_pos()) + "\t-ve: " + str(self.get_neg()) \
                + "\t~neutral: " + str(self.get_neu()))


    def get_titles(self):
        return self.a_headlines

    def show(self):
        if (self.cmd == "head"):
            for hl in self.a_headlines:
                print(hl)
        else:
            for buf in self.a_newsbuf:
                print(buf)

    def get_content(self, rss_url):
        content_buffer = []
        feed = feedparser.parse( rss_url )
        for entry in feed['entries']:
            content = urllib.request.urlopen(entry['link']).read()
            html = content.decode("utf-8")
            soup = BeautifulSoup(html,"html5lib")
            text = soup.get_text(strip=True)
            content_buffer.append(text)
        return content_buffer


class wfc_read():
    def __init__(self, content):
        self.content = content
        self.pos = 0.0
        self.neg = 0.0
        self.neu = 0.0

    def sentiment(self, level):
        sid = SentimentIntensityAnalyzer()
        results = []
        for hl in self.content:
            pol_score = sid.polarity_scores(hl)
            pol_score['headline'] = hl
            results.append(pol_score)

        for it in results:
            if ((level < 0) and (it['neg'] > 0)):
                print(it['headline'])
            elif ( (level > 1) and (it['pos'] > 0) ):
                print(it['headlinei'])
            elif ( (level == 0) and (it['neu'] > 0) ):
                print(it['headline'])

    def qsenti(self):
        sid = SentimentIntensityAnalyzer()
        results = []
        for hl in self.content:
            pol_score = sid.polarity_scores(hl)
            pol_score['headline'] = hl
            results.append(pol_score)

        for it in results:
            self.pos += it['pos']
            self.neg += it['neg']
            self.neu += it['neu']

