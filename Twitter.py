import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk import WordPunctTokenizer
from wordcloud import WordCloud, STOPWORDS
import nest_asyncio
import twint
import pandas as pd
import re
import nltk;
from PIL import Image

nltk.download('stopwords')
from pprint import pprint
import numpy as np
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# spacy for lemmatization
import spacy
# Plotting tools
import pyLDAvis
import matplotlib.pyplot as plt
from datetime import time

nest_asyncio.apply()

# DATA COLLECTION AND EXPLORATION
# DATA SCRAPING LEGAL BASED ON 2021 robots.txt FROM TWITTER
# Allow: /*?lang=
# Allow: /hashtag/*?src=
# Allow: /search?q=%23
# Allow: /i/api/
search_terms = ["Blockchain agrifood", "Blockchain food",
                "Blockchain beverage", "Blockchain agriculture", "Smart Contracts agrifood",
                "Smart Contracts food", "Smart Contracts beverage", "Smart Contracts agriculture",
                "Blockchain 4.0", "Blockchain industry 4.0"]

tweetsFull = []


def agrifoodsearch():
    c = twint.Config()
    for j in range(len(search_terms)):
        c.Search = search_terms[j]
        c.Store_object = True
        c.Min_likes = 5
        c.Store_object_tweets_list = tweetsFull
        c.Since = "2021-01-01 00:00:00"
        c.Limit = 20  # 20 default value, 3200 max value
        c.Lang = "en"
        twint.run.Search(c)


agrifoodsearch()

# DATA COLLECTION AND CLEAN
token = WordPunctTokenizer()


# Remove any special characters and http link, finally convert all in lowercase
def cleaning_tweets(t):
    pattern_mentions = '@[A-Za-z0–9_]+'
    pattern_hashtag = '#(\w+)'
    link_specialchar = [
        r'(https?://)?(www\.)?(\w+\.)?(\w+)(\.\w+)(/.+)?',
        '[^A-Za-z0-9]+'
    ]
    pattern_link_special_char = re.compile('|'.join(link_specialchar))
    del_amp = BeautifulSoup(t, 'lxml')
    del_amp_text = del_amp.get_text()
    text_tweet_no_mentions = re.sub(pattern_mentions, ' ', del_amp_text)
    text_tweet_no_hashtag = re.sub(pattern_hashtag, ' ', text_tweet_no_mentions)
    text_tweet_no_link_spec_char = re.sub(pattern_link_special_char, ' ', text_tweet_no_hashtag)
    lower_case = text_tweet_no_link_spec_char.lower()
    words = token.tokenize(lower_case)
    result_words = [x for x in words if len(x) > 2]
    return (" ".join(result_words)).strip()


def get_hashtag(t):
    lower_case = t.lower()
    return re.findall('#(\w+)', lower_case)


tweets_text = []
tweets_hashtag = []
tweets_replies = []
tweets_retweets = []
tweets_likes = []
tweets_time = []

for i in range(len(tweetsFull)):
    tweets_text.append(cleaning_tweets(tweetsFull[i].tweet))
    tweets_hashtag.extend(get_hashtag(tweetsFull[i].tweet))
    tweets_replies.append(tweetsFull[i].replies_count)
    tweets_retweets.append(tweetsFull[i].retweets_count)
    tweets_likes.append(tweetsFull[i].likes_count)
    tweets_time.append(tweetsFull[i].timestamp)

# DATA FORMATTING
df = pd.DataFrame(list(
    zip(tweets_text, tweets_replies, tweets_retweets, tweets_likes,
        (pd.to_datetime(tweets_time, format='%H:%M:%S')).time)),
    columns=['Tweet', 'Replies_Count', 'Retweets_Count', 'Likes_Count', 'Timestamp'])

tweetsString = pd.Series(df.Tweet.values).str.cat(sep=' ')
stopwords = set(STOPWORDS)
stopwords.update(["blockchain", "eth", "bnb", "btc", "food", "int", "one", "xdc"])
wordcloud = WordCloud(width=1600, stopwords=stopwords, height=800, max_font_size=200, max_words=50, collocations=False,
                      background_color='white').generate(tweetsString)

wordcloud_hash = WordCloud(width=1600, stopwords=stopwords, height=800, max_font_size=200, max_words=50,
                           collocations=False,
                           background_color='black').generate(" ".join(tweets_hashtag).strip())

plt.figure(figsize=(40, 30))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

plt.figure(figsize=(40, 30))
plt.imshow(wordcloud_hash, interpolation='bilinear')
plt.axis("off")
plt.show()

# df.to_excel("C:\\Users\\gianl\\Downloads\\Twitter.xlsx")

# # NLTK Stop words
# from nltk.corpus import stopwords
# stop_words = stopwords.words('english')
# stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
#
# print(df.Tweet.unique())
# df.head()

# plt.plot(df.Tweet.values, df.Likes_Count.values)
# plt.show()

# text =  df.Tweet.values
# wordcloud = WordCloud(
#     width = 3000,
#     height = 2000,
#     background_color = 'white',
#     stopwords = STOPWORDS).generate(str(text))
# fig = plt.figure(
#     figsize = (40, 30),
#     facecolor = 'k',
#     edgecolor = 'k')
# plt.imshow(wordcloud, interpolation = 'bilinear')
# plt.axis('off')
# plt.tight_layout(pad=0)
# plt.show()
