import re
import nltk
import pickle
#import gensim
#import gensim.corpora as corpora
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import pandas as pd
import xgboost as xgb

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')


def text_cleaning(text):
    """
    Remove figures, punctuation.

    Args:
        text(String): Row text to clean

    Returns:
       res(string): Cleaned text
    """

    pattern = re.compile(r'[^\w]|[\d_]')

    try:
        res = re.sub(pattern, " ", text).lower()
    except TypeError:
        return text

    return res


def tokenize(text):
    """
    Tokenize words of a text.

    Args:

        text(String): Row text

    Returns

        res(list): Tokenized string.
    """

    stop_words = set(stopwords.words('english'))

    try:
        res = word_tokenize(text, language='english')
    except TypeError:
        return text

    res = [token for token in res if token not in stop_words]
    return res


def lemmatizing(list):
    """
    Transform tokens into lems

    Args:
        tokens(list): List of tokens

    Returns:
        lemmatized(list): List of lemmatized tokens
    """

    lemma = WordNetLemmatizer()
    lemmatized = [lemma.lemmatize(word) for word in list]

    return lemmatized


def filter_2type_wd(tokens):
    """
    Filter singular nouns

    Args:
        tokens(list): A list o tokens


    Returns:

        res(list): Filtered token list
    """

    res = nltk.pos_tag(tokens, tagset='universal')

    res = [token[0]
           for token in res if token[1] == 'NOUN' or token[1] == 'VERB']

    return res


def open_model():
    supervised_model = "/Users/tatiana/OpenClass/models/knn_model.pkl"
    ml_model = "/Users/tatiana/OpenClass/models/ml_model.pkl"
    tfidf_model = "/Users/tatiana/OpenClass/models/tfidf_model.pkl"
    pca_model = "/Users/tatiana/OpenClass/models/pca_model.pkl"
    vocabulary = "/Users/tatiana/OpenClass/models/vocabulary.pkl"

    supervised_model = pickle.load(open(supervised_model, 'rb'))
    ml_model = pickle.load(open(ml_model, 'rb'))
    tfidf_model = pickle.load(open(tfidf_model, 'rb'))
    pca_model = pickle.load(open(pca_model, 'rb'))
    vocabulary = pickle.load(open(vocabulary, 'rb'))
    return supervised_model, ml_model, tfidf_model, pca_model, vocabulary


def open_model_tfidf():
    tfidf_model = "/Users/tatiana/OpenClass/models/tfidf_model.pkl"
    tfidf_model = pickle.load(open(tfidf_model, 'rb'))
    return tfidf_model


def open_vocabulary():
    vocabulary = "/Users/tatiana/OpenClass/models/vocabulary.pkl"
    vocabulary = pickle.load(open(vocabulary, 'rb'))
    return vocabulary


def open_pca():
    pca_model = "/Users/tatiana/OpenClass/models/pca_model.pkl"
    pca_model = pickle.load(open(pca_model, 'rb'))
    return pca_model


def open_supervised_model():
    supervised_model = "/Users/tatiana/OpenClass/models/knn_model.pkl"
    supervised_model = pickle.load(open(supervised_model, 'rb'))
    return supervised_model


def open_supervised_model_SVM():
    supervised_model_SVM = "/Users/tatiana/OpenClass/models/svm_model.pkl"
    supervised_model_SVM = pickle.load(open(supervised_model_SVM, 'rb'))
    return supervised_model_SVM


def open_supervised_model_XGB():
    #supervised_model_XGB = "/Users/tatiana/OpenClass/models/xgb_model_tf_idf.pkl"
    #supervised_model_XGB = pickle.load(open(supervised_model_XGB, 'rb'))
    supervised_model_XGB = xgb.XGBClassifier()
    supervised_model_XGB.load_model(
        '/Users/tatiana/OpenClass/models/xgb_model_tf_idf.json')
    return supervised_model_XGB


def open_ml_model():
    ml_model = "/Users/tatiana/OpenClass/models/ml_model.pkl"
    ml_model = pickle.load(open(ml_model, 'rb'))
    return ml_model


def predict_tags(text):
    """
    Predict tags according to a lemmatized text using a supervied model.

    Args:
        supervised_model(): Used mode to get prediction
        mlb_model(): Used model to detransform
    Returns:
        res(list): List of predicted tags
    """
    input_vector = open_model_tfidf().transform(text)
    input_vector = pd.DataFrame(input_vector.toarray())
    input_vector = open_pca().transform(input_vector)
    res = open_supervised_model().predict(input_vector)
    res = open_ml_model().inverse_transform(res)
    res = list(
        {tag for tag_list in res for tag in tag_list if (len(tag_list) != 0)})
    res = [tag for tag in res if tag in text]

    return res


def predict_tags_SVM(text):
    """
    Predict tags according to a lemmatized text using a supervied model.

    Args:
        supervised_model(): Used mode to get prediction
        mlb_model(): Used model to detransform
    Returns:
        res(list): List of predicted tags
    """
    input_vector = open_model_tfidf().transform(text)
    input_vector = pd.DataFrame(input_vector.toarray())
    input_vector = open_pca().transform(input_vector)
    res = open_supervised_model_SVM().predict(input_vector)
    res = open_ml_model().inverse_transform(res)
    res = list(
        {tag for tag_list in res for tag in tag_list if (len(tag_list) != 0)})
    res = [tag for tag in res if tag in text]

    return res


def predict_tags_XGB(text):
    """
    Predict tags according to a lemmatized text using a supervied model.

    Args:
        supervised_model(): Used mode to get prediction
        mlb_model(): Used model to detransform
    Returns:
        res(list): List of predicted tags
    """
    input_vector = open_model_tfidf().transform(text)
    input_vector = pd.DataFrame(input_vector.toarray())
    input_vector = open_pca().transform(input_vector)
    res = open_supervised_model_XGB().predict(input_vector)
    res = open_ml_model().inverse_transform(res)
    res = list(
        {tag for tag_list in res for tag in tag_list if (len(tag_list) != 0)})
    res = [tag for tag in res if tag in text]

    return res


def open_lda_dictionary():
    lda_dic = "/Users/tatiana/OpenClass/models/dictionary.pkl"
    lda_dic = pickle.load(open(lda_dic, 'rb'))
    return lda_dic


def open_lda_model():
    lda_model = "/Users/tatiana/OpenClass/models/lda_model.pkl"
    lda_model = pickle.load(open(lda_model, 'rb'))
    return lda_model


# def pred_tags_lda(text):

#    corpus_new = open_lda_dictionary().doc2bow(text)
#    topics = open_lda_model().get_document_topics(corpus_new)

    # find most relevant topic according to probability
#    relevant_topic = topics[0][0]
#    relevant_topic_prob = topics[0][1]

#    for i in range(len(topics)):
#        if topics[i][1] > relevant_topic_prob:
#            relevant_topic = topics[i][0]
#            relevant_topic_prob = topics[i][1]

    # retrieve associated to topic tags present in submited text
#    potential_tags = open_lda_model().get_topic_terms(topicid=relevant_topic, topn=20)

#   relevant_tags = [open_lda_dictionary()[tag[0]]
#                     for tag in potential_tags if open_lda_dictionary()[tag[0]] in text]

#    return relevant_tags

def pred_tags_lda(texts):

    id2word = open_lda_dictionary()
    corpus_new = open_lda_model().id2word.doc2bow(texts)
    topics, word_topics, phi_values = open_lda_model().get_document_topics(
        corpus_new, per_word_topics=True)

    # find most relevant topic according to probability
    relevant_topic = topics[0][0]

    relevant_topic_prob = topics[0][1]

    for i in range(len(topics)):
        if topics[i][1] > relevant_topic_prob:
            relevant_topic = topics[i][0]
            relevant_topic_prob = topics[i][1]

    # retrieve associated to topic tags present in submited text
    potential_tags = open_lda_model().get_topic_terms(
        topicid=relevant_topic, topn=100)

    full_relevant_tags = [open_lda_dictionary()[tag[0]]
                          for tag in potential_tags]  # if id2word[tag[0]] in text]

    relevant_tags = full_relevant_tags[0:5]

    return relevant_tags
