import re
import matplotlib.pyplot as plt
import nest_asyncio
import nltk
import pandas as pd
import seaborn as sns
import twint
import pyLDAvis.gensim_models
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from bs4 import BeautifulSoup
from nltk import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

nest_asyncio.apply()

# DATA COLLECTION AND EXPLORATION
# DATA SCRAPING LEGAL BASED ON 2021 robots.txt FROM TWITTER
# Allow: /*?lang=
# Allow: /hashtag/*?src=
# Allow: /search?q=%23
# Allow: /i/api/
search_terms = ["Blockchain", "Blockchain agrifood", "Blockchain food",
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
        c.Limit = 3000  # 20 default value, 3200 max value
        c.Lang = "en"
        twint.run.Search(c)


agrifoodsearch()

# DATA COLLECTION AND CLEAN
token = WordPunctTokenizer()

# Remove any special characters, link/url, hashtag, words less than 2 letters, convert all in lowercase
# lemmatization, remove stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(["blockchain", "eth", "bnb", "btc", "food", "int", "one", "xdc"])
lemmatizer = WordNetLemmatizer()


def cleaning_tweets(t):
    pattern_mentions = '@[A-Za-z0–9_]+'
    pattern_hashtag = '#(\\w+)'
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
    lemmas = []
    for word in words:
        lemmas.append(lemmatizer.lemmatize(word))
    result_lemmas = [x for x in lemmas if len(x) > 2]
    result_after_stopwords = []
    for final in result_lemmas:
        if final not in stop_words:
            result_after_stopwords.append(final)
    return (" ".join(result_after_stopwords)).strip()


# Take only hashtags, remove words less than 2 letters, convert all in lowercase, lemmatization, remove stopwords
def get_hashtag(t):
    lower_case = t.lower()
    hashtags = re.findall('#(\\w+)', lower_case)
    lemmas = []
    for hashtag in hashtags:
        lemmas.append(lemmatizer.lemmatize(hashtag))
    result_lemmas = [x for x in lemmas if len(x) > 2]
    result_after_stopwords = []
    for final in result_lemmas:
        if final not in stop_words:
            result_after_stopwords.append(final)
    return result_after_stopwords


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

df_hashtag = pd.DataFrame(tweets_hashtag)

hashtag_chart_values = df_hashtag[0].value_counts()
noun_values = []

for i in range(len(df.Tweet.values)):
    nouns = token.tokenize(df.Tweet.values[i])
    noun_values.extend(nouns)

df_noun = pd.DataFrame(noun_values)
noun_chart_values = df_noun[0].value_counts()

tweetsString = pd.Series(df.Tweet.values).str.cat(sep=' ')
wordcloud = WordCloud(width=1600, height=800, max_font_size=200, max_words=50, collocations=False,
                      background_color='white').generate(tweetsString)

wordcloud_hash = WordCloud(width=1600, height=800, max_font_size=200, max_words=50,
                           collocations=False,
                           background_color='white').generate(" ".join(tweets_hashtag).strip())

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('WordCloud Noun Agrifood')
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_hash, interpolation='bilinear')
plt.axis("off")
plt.title('WordCloud Hashtag Agrifood')
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=noun_chart_values[:20].values, y=noun_chart_values[:20].index, alpha=0.8)
plt.title('Top 20 Noun Agrifood')
plt.ylabel('Top 20 Noun from Tweet', fontsize=12)
plt.xlabel('Count of Noun', fontsize=12)
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=hashtag_chart_values[:20].values, y=hashtag_chart_values[:20].index, alpha=0.8)
plt.title('Top 20 Hashtag Agrifood')
plt.ylabel('Hashtag from Tweet', fontsize=12)
plt.xlabel('Count of Hashtag', fontsize=12)
plt.show()

# df.to_excel("C:\\Users\\gianl\\Downloads\\Twitter.xlsx")

words_lda = []
for word_lda in tweets_text:
    words_lda.append(token.tokenize(word_lda))

# The two main inputs to the LDA topic model are the dictionary(id2word) and the corpus.

# Create Dictionary
id2word = corpora.Dictionary(words_lda)

# Create Corpus
texts = words_lda

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View, (0,1) means that word id=0 occurs 1 time in the document, (1,3) means that word id=1 occurs 3 times in the
# document
print(corpus[:1])

# Build LDA model
# Alpha is an hyperparameter that affects sparsity of the topics.
# chunksize is the number of documents to be used in each training chunk.
# update_every determines how often the model parameters should be updated
# and passes is the total number of training passes.
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=20,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

# Print the Keyword in the 20 topics
print(lda_model.print_topics())

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.


# Visualize the topics
visualisation = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds='mmds')
pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')
