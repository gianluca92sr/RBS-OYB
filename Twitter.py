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


# WORKAROUND - all in main to avoid code duplication while getting coherence value with c_v coherence method
def main():
    nest_asyncio.apply()

    tweetsFull = []

    # DATA COLLECTION AND EXPLORATION
    # DATA SCRAPING LEGAL BASED ON 2021 robots.txt FROM TWITTER
    # Allow: /*?lang=
    # Allow: /hashtag/*?src=
    # Allow: /search?q=%23
    # Allow: /i/api/

    # AGRIFOOD KEYWORDS
    # search_terms = ["blockchain agrifood", "blockchain food",
    #                 "blockchain beverage", "blockchain agriculture", "smart contracts agrifood",
    #                 "smart contracts food", "smart contracts beverage", "smart contracts agriculture"]

    # ENERGY + UTILITIES KEYWORDS
    # search_terms = ["blockchain energy", "blockchain utility",
    #                 "smart contracts energy", "smart contracts utility"]

    # IOT KEYWORDS
    # search_terms = ["blockchain iot", "blockchain internet of things",
    #                 "blockchain industry 4.0", "smart contracts iot",
    #                 "smart contracts internet of things", "smart contracts industry 4.0"]

    # HEALTHCARE KEYWORDS
    search_terms = ["blockchain healthcare", "blockchain medical",
                    "blockchain health management", "blockchain medical management", "blockchain pharma",
                    "blockchain clinic", "blockchain patient", "smart contracts healthcare"]



    def tweetsSearch():
        c = twint.Config()
        for j in range(len(search_terms)):
            c.Search = search_terms[j]
            c.Store_object = True
            # c.Min_likes = 5
            c.Store_object_tweets_list = tweetsFull
            c.Since = "2020-01-01"
            c.Limit = 1000000  # 20 default value, 3200 max value
            c.Lang = "en"
            c.Debug = True
            twint.run.Search(c)

    # low tweets retrieve problem solved decommented ('query_source', 'typed_query') in url.py
    tweetsSearch()

    print(len(tweetsFull))

    # DATA COLLECTION AND CLEAN
    token = WordPunctTokenizer()

    # Remove any special characters, link/url, hashtag, words less than 2 letters, convert all in lowercase
    # lemmatization, remove stopwords
    my_stop_words = STOPWORDS.union(
        set(["blockchain", "food", "int", "one", "xdc", "industry", "agrifood", "agriculture",
             "technology", "smart", "contract", "agricultural", "case", "tech", "agri", "like", "chain",
             "medical", "patient", "healthcare", "pharma", "management"]))
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

    # Build LDA model
    # Alpha is an hyperparameter that affects sparsity of the topics.
    # chunksize is the number of documents to be used in each training chunk.
    # update_every determines how often the model parameters should be updated
    # and passes is the total number of training passes.
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

    coherence = None
    number_of_topic_to_choose = None
    coherence_values = []
    coherence_index = []
    lda_model = None

    # Choose the best lda_model based on the number of topics between 5 and 20. Coherence higher is better
    for i in range(2, 10):
        temp_model = choose_the_best_lda(i)
        # u_mass approach measures how much a common word appearing within a topic is a good predictor for a less common
        # word in the topic.
        # temp_coherence = CoherenceModel(model=temp_model, corpus=corpus, dictionary=id2word, coherence='u_mass').get_coherence()
        # c_v from 0 to 1, the latter is better
        temp_coherence = CoherenceModel(model=temp_model, texts=texts, dictionary=id2word,
                                        coherence='c_v').get_coherence()
        coherence_values.append(temp_coherence)
        coherence_index.append(i)

        if i == 2:
            coherence = temp_coherence
            number_of_topic_to_choose = i
            lda_model = temp_model
        else:
            if temp_coherence > coherence:
                coherence = temp_coherence
                number_of_topic_to_choose = i
                lda_model = temp_model

    values = pd.Series(coherence_values)
    indexes = pd.Series(coherence_index)
    plt.ylabel("Coherence")
    plt.xlabel("N° of Topics")
    plt.title("Coherence / N° of Topics")
    plt.legend("N° of Topics to choose: " + str(number_of_topic_to_choose))
    plt.plot(indexes.values, values.values)

    plt.show()

    # Visualize the topics
    # Red bars represent the frequency of a word in the specific topic (FWS).
    # Gray bars represent the frequency of a word in all topics (FWA).
    # Lowering Lambda value adds more weight to the ratio FWS / FWA. Doing this allow us to see more clearly
    # the more valuable words for that specific topic and characterize that topic
    visualisation = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds='mmds')
    pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')

    import os
    # Setup mallet environment change it according to your drive
    os.environ.update({'MALLET_HOME': r'C:/Users/gianl/Desktop/Master/Progetto/mallet-2.0.8'})
    # Setup mallet path change it according to your drive
    mallet_path = 'C:/Users/gianl/Desktop/Master/Progetto/mallet-2.0.8/bin/mallet'

    def choose_the_best_mallet_lda(x):
        temp_model_mallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus,
                                                             num_topics=x,
                                                             id2word=id2word)

        return temp_model_mallet

    coherence_mallet = None
    number_of_topic_to_choose_mallet = None
    coherence_values_mallet = []
    coherence_index_mallet = []
    lda_model_mallet = None

    # Choose the best lda_model based on the number of topics between 5 and 20. Coherence higher is better
    for i in range(2, 10):
        temp_model_mallet = choose_the_best_mallet_lda(i)
        # c_v from 0 to 1, the latter is better
        temp_coherence_mallet = CoherenceModel(model=temp_model_mallet, texts=texts, dictionary=id2word,
                                               coherence='c_v').get_coherence()
        coherence_values_mallet.append(temp_coherence_mallet)
        coherence_index_mallet.append(i)

        if i == 2:
            coherence_mallet = temp_coherence_mallet
            number_of_topic_to_choose_mallet = i
            lda_model_mallet = temp_model_mallet
        else:
            if temp_coherence_mallet > coherence_mallet:
                coherence_mallet = temp_coherence_mallet
                number_of_topic_to_choose_mallet = i
                lda_model_mallet = temp_model_mallet

    values = pd.Series(coherence_values_mallet)
    indexes = pd.Series(coherence_index_mallet)
    plt.ylabel("Coherence")
    plt.xlabel("N° of Topics")
    plt.title("Coherence / N° of Topics")
    plt.legend("N° of Topics to choose: " + str(number_of_topic_to_choose_mallet))
    plt.plot(indexes.values, values.values)

    plt.show()

    mallet_lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_model_mallet)
    visualisationMallet = pyLDAvis.gensim_models.prepare(mallet_lda_model, corpus, id2word, mds='mmds')
    pyLDAvis.save_html(visualisationMallet, 'LDA_Mallet_Visualization.html')


if __name__ == "__main__":
    main()
