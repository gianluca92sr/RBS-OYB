import re
import matplotlib.pyplot as plt
import nest_asyncio
import pandas as pd
import seaborn as sns
import twint
import pyLDAvis.gensim_models
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from bs4 import BeautifulSoup
from nltk import WordPunctTokenizer
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

nest_asyncio.apply()

# DATA COLLECTION AND EXPLORATION
# DATA SCRAPING LEGAL BASED ON 2021 robots.txt FROM TWITTER
# Allow: /*?lang=
# Allow: /hashtag/*?src=
# Allow: /search?q=%23
# Allow: /i/api/
search_terms = ["agriculture", "agrifood" "blockchain agrifood", "blockchain food",
                "blockchain beverage", "blockchain agriculture", "smart contracts agrifood",
                "smart contracts food", "smart contracts beverage", "smart contracts agriculture",
                "blockchain 4.0", "blockchain industry 4.0"]

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

# Remove any special characters, link/url, hashtag, words less than 2 letters, convert all in lowercase
# lemmatization, remove stopwords
my_stop_words = STOPWORDS.union(
    set(["blockchain", "eth", "bnb", "btc", "food", "int", "one", "xdc", "industry", "agrifood", "agriculture"]))
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
        if final not in my_stop_words:
            if not final.isnumeric():
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
        if final not in my_stop_words:
            if not final.isnumeric():
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
choerence_values = []
def choose_the_best_lda(x):
    temp_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                    id2word=id2word,
                                    num_topics=x,
                                    random_state=100,
                                    update_every=1,
                                    chunksize=100,
                                    passes=10,
                                    alpha='auto',
                                    eta='auto',
                                    per_word_topics=True)

    return temp_model


perplexity = None
number_of_topic_to_choose = None
perplexity_value_series = []
perplexity_index_series = []
# Choose the best lda_model based on the number of topics between 5 and 20. At the first iteration the base perplexity
# is set, then it is checked against the temp_perplexity of the other number of topics and also we check
# if this difference is over 10% of perplexity DECREASE (so we check with <) to avoid overfitting (V2−V1)/|V1|×100
for i in range(5, 20):
    temp_model = choose_the_best_lda(i)
    temp_perplexity = temp_model.log_perplexity(corpus)
    perplexity_value_series.append(temp_perplexity)
    perplexity_index_series.append(i)
    # u_mass approach measures how much a common word appearing within a topic is a good predictor for a less common word in the topic.
    choerence_values.append(CoherenceModel(model=temp_model, dictionary=id2word, corpus=corpus, coherence='u_mass')
                            .get_coherence())
    if(i == 5):
        perplexity = temp_perplexity
        number_of_topic_to_choose = i
    else:
        if(temp_perplexity < perplexity and ((temp_perplexity - perplexity)/abs(perplexity) * 100) < -10):
            perplexity = temp_perplexity
            number_of_topic_to_choose = i


y = pd.Series(perplexity_value_series)
x = pd.Series(perplexity_index_series)
plt.ylabel("Perplexity")
plt.xlabel("N° of Topics")
plt.title("Perplexity / N° of Topics")
p = plt.plot(x.values, y.values)

x2 = pd.Series(choerence_values)
plt.ylabel("Choerence")
plt.xlabel("N° of Topics")
plt.title("Choerence / N° of Topics")
p2 = plt.plot(x2.index, x2.values)
plt.show()
# # Build LDA model
# # Alpha is an hyperparameter that affects sparsity of the topics.
# # chunksize is the number of documents to be used in each training chunk.
# # update_every determines how often the model parameters should be updated
# # and passes is the total number of training passes.
# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                             id2word=id2word,
#                                             num_topics=10,
#                                             random_state=100,
#                                             update_every=1,
#                                             chunksize=100,
#                                             passes=10,
#                                             alpha='auto',
#                                             per_word_topics=True)
#
# # Print the Keyword in the 20 topics
# print(lda_model.print_topics())
#
# # Compute Coherence Score
# coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='u_mass')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ', coherence_lda)
#
# # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
# print('\nPerplexity: ', lda_model.log_perplexity(corpus))
#
#
# # Visualize the topics
# # Red bars represent the frequency of a word in the specific topic (FWS).
# # Gray bars represent the frequency of a word in all topics (FWA).
# # Lowering Lambda value adds more weight to the ratio FWS / FWA. Doing this allow us to see more clearly
# # the more valuable words for that specific topic and characterize that topic
# visualisation = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds='mmds')
# pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')
x= []
