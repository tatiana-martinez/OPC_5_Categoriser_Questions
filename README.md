# OPC_5_Categoriser_Questions

Application de suggestion de tags. Projet OpenClassRoom en partenariat avec CentraleSupelec.

Après avoir entraîné des modèles en mode supervisé et non supervisé à l’aide de données issues de stackexchange (de stackoverflow), l’application propose tout d’abord de saisir sa question telle que l'utilisateur voudrait la soumettre à la communauté sur stackoverflow. 
Les modèles appelées par l'application ont été entraînés grâce aux textes des titres des questions majoritairement postées de 2009 à 2018.
L’application donne la possibilité de choisir entre plusieurs modèles et chaque modèle fournit sa liste de tags relativement à la question saisie. Avant de soumettre les suggestions de tags, en dernière étape, l’application restitue les étapes de prétraitement de la question.

L’application dont les sources sont dans ce dossier github ont été déployées sur Streamlit Cloud à l’adresse suivante : https://tags-suggestion.streamlitapp.com/