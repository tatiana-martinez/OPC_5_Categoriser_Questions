import re
import nltk
import pickle
import joblib
import sklearn
import numpy as np
import collections
import operator
import os
#import os

import gensim
#import gensim.corpora as corpora

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize

import pandas as pd
import xgboost as xgb

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('stopwords')


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


def convert_text_bert(clean_question):
    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in clean_question]
    return sentences


def tokens_text_bert(sentences):
    tokenizer = open_model_bert()
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    return tokenized_texts


def input_id_bert(tokenized_texts):
    tokenizer = open_model_bert()
    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x)
                 for x in tokenized_texts]
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=128,
                              dtype="long", truncating="post", padding="post")
    # df
    df_input_ids = pd.DataFrame(input_ids)
    return df_input_ids


# def open_model():
#    absolute_path = os.path.dirname(__file__)
#    relative_path_supervised_model = "../models/knn_model.pkl"
#    supervised_model = os.path.join(
#        absolute_path, relative_path_supervised_model)
#    #supervised_model = path+"../models/knn_model.pkl"

#    relative_path_ml_model = "../models/ml_model.pkl"
#    ml_model = os.path.join(absolute_path, relative_path_ml_model)
    #ml_model = path+"../models/ml_model.pkl"

#    relative_path_tfidf_model = "../models/tfidf_model.pkl"
#    tfidf_model = os.path.join(absolute_path, relative_path_tfidf_model)
    #tfidf_model = path+"../models/tfidf_model.pkl"

#    relative_path_pca_model = "../models/pca_model.pkl"
#   pca_model = os.path.join(absolute_path, relative_path_pca_model)
    #pca_model = path+"../models/pca_model.pkl"

#    relative_path_vocabulary = "../models/vocabulary.pkl"
#    vocabulary = os.path.join(absolute_path, relative_path_vocabulary)
    #vocabulary = "../models/vocabulary.pkl"

#    supervised_model = pickle.load(open(supervised_model, 'rb'))
#    ml_model = pickle.load(open(ml_model, 'rb'))
#    tfidf_model = pickle.load(open(tfidf_model, 'rb'))
#    pca_model = pickle.load(open(pca_model, 'rb'))
#    vocabulary = pickle.load(open(vocabulary, 'rb'))
#    return supervised_model, ml_model, tfidf_model, pca_model, vocabulary


def open_model_tfidf():
    absolute_path = os.path.dirname(__file__)
    relative_path_tfidf_model = "../models/tfidf_model.pkl"
    tfidf_model = os.path.join(absolute_path, relative_path_tfidf_model)

    #tfidf_model = path+"../models/tfidf_model.pkl"
    tfidf_model = pickle.load(open(tfidf_model, 'rb'))
    return tfidf_model


def open_model_bert():
    absolute_path = os.path.dirname(__file__)
    relative_path_bert_model = '../models/bert_model'
    bert_model = os.path.join(absolute_path, relative_path_bert_model)

    bert_model = torch.load(bert_model)
    return bert_model


def open_vocabulary():
    absolute_path = os.path.dirname(__file__)
    relative_path_vocabulary = "../models/vocabulary.pkl"
    vocabulary = os.path.join(absolute_path, relative_path_vocabulary)

    #vocabulary = path+"../models/vocabulary.pkl"
    vocabulary = pickle.load(open(vocabulary, 'rb'))
    return vocabulary


def open_vocabulary_bert():
    absolute_path = os.path.dirname(__file__)
    relative_path_vocabulary_bert = "../models/bert_vocab.pkl"
    vocabulary_bert = os.path.join(
        absolute_path, relative_path_vocabulary_bert)

    #vocabulary_bert = path+"../models/bert_vocab.pkl"
    vocabulary_bert = pickle.load(open(vocabulary_bert, 'rb'))
    return vocabulary_bert


def open_pca():
    absolute_path = os.path.dirname(__file__)
    relative_path_pca_model = "../models/pca_model.pkl"
    pca_model = os.path.join(absolute_path, relative_path_pca_model)

    #pca_model = path+"../models/pca_model.pkl"
    pca_model = pickle.load(open(pca_model, 'rb'))
    return pca_model


def open_pca_xgb_model():
    absolute_path = os.path.dirname(__file__)
    relative_path_pca_xgb_model = "../models/pca_tfidf_xgb_model.pkl"
    pca_xgb_model = os.path.join(absolute_path, relative_path_pca_xgb_model)

    #pca_xgb_model = path+"../models/pca_tfidf_xgb_model.pkl"
    pca_xgb_model = pickle.load(open(pca_xgb_model, 'rb'))
    return pca_xgb_model


def open_pca_bert():
    absolute_path = os.path.dirname(__file__)
    relative_path_pca_xgb_model = "../models/pca_bert_model.pkl"
    pca_bert_model = os.path.join(absolute_path, relative_path_pca_xgb_model)

    #pca_bert_model = path+"../models/pca_bert_model.pkl"
    pca_bert_model = pickle.load(open(pca_bert_model, 'rb'))
    return pca_bert_model


def open_supervised_model():
    absolute_path = os.path.dirname(__file__)
    relative_path_supervised_model = "../models/knn_model.pkl"
    supervised_model = os.path.join(
        absolute_path, relative_path_supervised_model)

    #supervised_model = path+"../models/knn_model.pkl"
    supervised_model = pickle.load(open(supervised_model, 'rb'))
    return supervised_model


def open_supervised_model_knn_bert():
    absolute_path = os.path.dirname(__file__)
    relative_path_supervised_model_knn_bert = "../models/knn_model_bert1.pkl"
    supervised_model_knn_bert = os.path.join(
        absolute_path, relative_path_supervised_model_knn_bert)

    #supervised_model_knn_bert = path+"../models/knn_model_bert1.pkl"
    supervised_model_knn_bert = pickle.load(
        open(supervised_model_knn_bert, 'rb'))
    return supervised_model_knn_bert


def open_supervised_model_SVM():
    absolute_path = os.path.dirname(__file__)
    relative_path_supervised_model_SVM = "../models/svm_model.pkl"
    supervised_model_SVM = os.path.join(
        absolute_path, relative_path_supervised_model_SVM)

    #supervised_model_SVM = path+"../models/svm_model.pkl"
    supervised_model_SVM = pickle.load(open(supervised_model_SVM, 'rb'))
    return supervised_model_SVM


def open_supervised_model_svm_bert():
    absolute_path = os.path.dirname(__file__)
    relative_path_supervised_model_svm_bert = "../models/svm_model_bert.pkl"
    supervised_model_svm_bert = os.path.join(
        absolute_path, relative_path_supervised_model_svm_bert)

    #supervised_model_svm_bert = path+"../models/svm_model_bert.pkl"
    supervised_model_svm_bert = pickle.load(
        open(supervised_model_svm_bert, 'rb'))
    return supervised_model_svm_bert


def open_supervised_model_XGB():
    absolute_path = os.path.dirname(__file__)
    relative_path_supervised_model_XGB = "../models/xgb_model_tf_idf_jb.joblib"
    supervised_model_XGB = os.path.join(
        absolute_path, relative_path_supervised_model_XGB)

    #supervised_model_XGB = path+"../models/xgb_model_tf_idf_jb.joblib"
    supervised_model_XGB = joblib.load(open(supervised_model_XGB, 'rb'))
    return supervised_model_XGB


def open_supervised_model_xgb_bert():
    absolute_path = os.path.dirname(__file__)
    relative_path_supervised_model_XGB_bert = "../models/ovc_xgb_bert.pkl"
    supervised_model_XGB_bert = os.path.join(
        absolute_path, relative_path_supervised_model_XGB_bert)

    #supervised_model_XGB_bert = path+"../models/ovc_xgb_bert.pkl"
    supervised_model_XGB_bert = pickle.load(
        open(supervised_model_XGB_bert, 'rb'))
    return supervised_model_XGB_bert


def open_ml_model():
    absolute_path = os.path.dirname(__file__)
    relative_path_ml_model = "../models/ml_model.pkl"
    ml_model = os.path.join(
        absolute_path, relative_path_ml_model)

    #ml_model = path+"../models/ml_model.pkl"
    ml_model = pickle.load(open(ml_model, 'rb'))
    return ml_model


def top_200_tags():
    absolute_path = os.path.dirname(__file__)
    relative_path_df_top_200_tags = "../data/df_top200_tags.csv"
    top_200_tags = os.path.join(absolute_path, relative_path_df_top_200_tags)

    #top_200_tags = absolute_path+"../data/df_top200_tags.csv"
    df_top_200_tags = pd.read_csv(top_200_tags)
    return df_top_200_tags


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


def pca_transform_knn_bert(df):
    input_vector = open_pca_bert().transform(df)
    return input_vector


def predict_tags_knn_bert(input_vector):
    """
    Predict tags according to a lemmatized text using a supervied model.

    Args:
        supervised_model(): Used mode to get prediction
        mlb_model(): Used model to detransform
    Returns:
        res(list): List of predicted tags
    """
    res = open_supervised_model_knn_bert().predict(input_vector)
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


def predict_alltags_svm_bert(df, clean_question):
    """
    Predict tags according to a lemmatized text using a supervied model.

    Args:
        supervised_model(): Used mode to get prediction
        mlb_model(): Used model to detransform
    Returns:
        res(list): List of predicted tags
    """
    model_svm_bert = open_supervised_model_svm_bert()
    model_ml = open_ml_model()
    input_vector = open_pca_bert().transform(df)
    res = model_svm_bert.predict(input_vector)
    res = model_ml.inverse_transform(res)

    res = res[0:5]
    # flat list of res
    flat_list = [item for sublist in res for item in sublist]

    # tags in flag list and in clean_question
    tags_flat_clean = [tag for tag in flat_list if tag in clean_question]

    len_tags_flat_clean = len(tags_flat_clean)

    nb = 5-len_tags_flat_clean
    flat_list[0:nb]
    if nb > 0 & nb < 5:
        tags = tags_flat_clean + flat_list[0:nb]
        print("cas1")
        print(nb)
        print(tags)
    elif nb == 0:
        tags = tags_flat_clean[0:5]
        print("cas2")
        print(nb)
        print(tags)
    else:
        tags = flat_list[0:5]
        print("cas3")
        print(nb)
        print(tags)

    return tags


def tfidf_pca_XGB(text):

    model_tfidf = open_model_tfidf()
    model_pca_xgb = open_pca_xgb_model()

    input_vector = model_tfidf.transform(text)
    input_vector = pd.DataFrame(input_vector.toarray())
    input_vector = model_pca_xgb.transform(input_vector)
    return input_vector


def predict_proba_XGB(text):
    model_XGB = open_supervised_model_XGB()
    ovc_xgb_preds_proba = model_XGB.predict_proba(text)
    return ovc_xgb_preds_proba


def predict_5tags_XGB(predict_proba_XGB, top_200_tags):
    n = 5
    x = predict_proba_XGB.shape[0]
    model_XGB = open_supervised_model_XGB()

    top_n_lables_idx = np.argsort(-predict_proba_XGB, axis=1)[:, : n]
    top_n_lables_proba = np.round(-np.sort(-predict_proba_XGB), 3)[:, : n]
    top_n_labels = [model_XGB.classes_[i] for i in top_n_lables_idx]

    top200_tags_T = top_200_tags.T
    top_n_tags = [[top200_tags_T.iloc[0, top_n_labels[j][i]]
                   for i in range(0, n)] for j in range(0, x)]

    flat_1_tags = [item[0] for item in top_n_tags]
    flat_1_lables_proba = [item[0] for item in top_n_lables_proba]

    dict_tags = dict(zip(flat_1_tags, flat_1_lables_proba))
    sorted_dict_tags = dict(
        sorted(dict_tags.items(), key=operator.itemgetter(1), reverse=True))

    return sorted_dict_tags


def predict_tags_XGB(sorted_dict_tags):
    lst_tags = list(sorted_dict_tags.keys())
    return lst_tags


def predict_tags_XGB_bert(df):
    """
    Predict tags according to a lemmatized text using a supervied model.

    Args:
        supervised_model(): Used mode to get prediction
        mlb_model(): Used model to detransform
    Returns:
        res(list): List of predicted tags
    """
    model_xgb_bert = open_supervised_model_xgb_bert()
    model_ml = open_ml_model()
    input_vector = open_pca_bert().transform(df)
    res = model_xgb_bert.predict(input_vector)
    res = model_ml.inverse_transform(res)
    res = list(
        {tag for tag_list in res for tag in tag_list if (len(tag_list) != 0)})
    res = [tag for tag in res if tag in text]
    return res


def open_lda_dictionary():
    absolute_path = os.path.dirname(__file__)
    relative_path_lda_dic = '../models/dictionary.pkl'
    lda_dic = os.path.join(absolute_path, relative_path_lda_dic)

    lda_dic = pickle.load(open(lda_dic, 'rb'))
    return lda_dic


def open_lda_model():
    absolute_path = os.path.dirname(__file__)
    relative_path_lda_model = "../models/lda_model.pkl"

    lda_model = os.path.join(absolute_path, relative_path_lda_model)
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
                          for tag in potential_tags]

    relevant_tags = full_relevant_tags[0: 5]

    return relevant_tags
