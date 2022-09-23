import streamlit as st
import pandas as pd
from helper import text_cleaning
from helper import tokenize
from helper import lemmatizing
from helper import filter_2type_wd
from helper import open_model
from helper import predict_tags
from helper import open_lda_dictionary
from helper import open_lda_model
from helper import pred_tags_lda
from helper import predict_tags_SVM
from helper import predict_tags_XGB

st.title("Bienvenue sur votre outil de suggestion de tags!")

question = st.text_input(
    "1. Veuillez saisir votre question dans la zone qui suit :", '')
#st.write("**Votre question :**", question)

pred_type = ['Unsupervised', 'Supervised_KNN',
             'Supervised_SVM', 'Supervised_XGBoost']

pred = st.radio("2. Choix du type de prédiction :", pred_type)
#st.write("**Vous avez choisi :**", pred)

val_button = st.button('Valider')

if pred == 'Supervised_KNN' and val_button:

    clean_question = text_cleaning(question)
    st.write('Votre question "cleaned" : ', clean_question)

    token_clean_question = tokenize(clean_question)
    st.write('Votre question "cleaned" et "tokenized" : ', token_clean_question)

    lemma_token_clean_question = lemmatizing(token_clean_question)
    st.write('Votre question "cleaned", "tokenized" et "lemmatized" : ',
             lemma_token_clean_question)

    filter_lemma_token_clean_question = filter_2type_wd(
        lemma_token_clean_question)
    st.write('Votre question "cleaned", "tokenized", "lemmatized" et "filtered" : ',
             filter_lemma_token_clean_question)

    supervised_model_predict_tags = predict_tags(
        filter_lemma_token_clean_question)
    st.write('Voici une proposition de tags en rapport avec votre question : ',
             supervised_model_predict_tags)

if pred == 'Supervised_SVM' and val_button:

    clean_question = text_cleaning(question)
    st.write('Votre question "cleaned" : ', clean_question)

    token_clean_question = tokenize(clean_question)
    st.write('Votre question "cleaned" et "tokenized" : ', token_clean_question)

    lemma_token_clean_question = lemmatizing(token_clean_question)
    st.write('Votre question "cleaned", "tokenized" et "lemmatized" : ',
             lemma_token_clean_question)

    filter_lemma_token_clean_question = filter_2type_wd(
        lemma_token_clean_question)
    st.write('Votre question "cleaned", "tokenized", "lemmatized" et "filtered" : ',
             filter_lemma_token_clean_question)

    supervised_model_predict_tags_SVM = predict_tags_SVM(
        filter_lemma_token_clean_question)
    st.write('Voici une proposition de tags en rapport avec votre question : ',
             supervised_model_predict_tags_SVM)

if pred == 'Supervised_XGBoost' and val_button:

    clean_question = text_cleaning(question)
    st.write('Votre question "cleaned" : ', clean_question)

    token_clean_question = tokenize(clean_question)
    st.write('Votre question "cleaned" et "tokenized" : ', token_clean_question)

    lemma_token_clean_question = lemmatizing(token_clean_question)
    st.write('Votre question "cleaned", "tokenized" et "lemmatized" : ',
             lemma_token_clean_question)

    filter_lemma_token_clean_question = filter_2type_wd(
        lemma_token_clean_question)
    st.write('Votre question "cleaned", "tokenized", "lemmatized" et "filtered" : ',
             filter_lemma_token_clean_question)

    supervised_model_predict_tags_XGB = predict_tags_XGB(
        filter_lemma_token_clean_question)
    st.write('Voici une proposition de tags en rapport avec votre question : ',
             supervised_model_predict_tags_XGB)

if pred == 'Unsupervised' and val_button:

    clean_question = text_cleaning(question)
    st.write('Votre question "cleaned" : ', clean_question)

    token_clean_question = tokenize(clean_question)
    st.write('Votre question "cleaned" et "tokenized" : ', token_clean_question)

    lemma_token_clean_question = lemmatizing(token_clean_question)
    st.write('Votre question "cleaned", "tokenized" et "lemmatized" : ',
             lemma_token_clean_question)

    filter_lemma_token_clean_question = filter_2type_wd(
        lemma_token_clean_question)
    st.write('Votre question "cleaned", "tokenized", "lemmatized" et "filtered" : ',
             filter_lemma_token_clean_question)

    unsupervised_model_predict_tags = pred_tags_lda(
        filter_lemma_token_clean_question)
    st.write('Voici une proposition de tags en rapport avec votre question : ',
             unsupervised_model_predict_tags)

#st.write('en construction')
