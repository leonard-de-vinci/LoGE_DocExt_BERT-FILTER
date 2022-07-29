# -*- coding: utf-8 -*-

import numpy as np
import re
import string
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

import os
from elasticsearch import Elasticsearch
from time import sleep
from datetime import datetime,timezone

from gensim.models import KeyedVectors
import sklearn.metrics.pairwise as ps
from sklearn.decomposition import PCA
from wikipedia2vec import Wikipedia2Vec

# --------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Connexion au cluster ES ------------------------------------------
# --------------------------------------------------------------------------------------------------------------


es_password=os.environ.get("ELASTIC_PASSWORD")#permet de récuperer la variable d'environnement 'ELASTIC_PASSWORD'
elastic = Elasticsearch("https://elastic:"+es_password+"@es1:9200",ca_certs="/usr/src/app/certs/ca/ca.crt")


# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Fonctions globales  --------------------------------------------
# --------------------------------------------------------------------------------------------------------------


def flatten(listOfList):
    """
        Takes a list of lists and flattens it 

        Parameters
        ----------
        listOfList : [[X],[Y],..,[Z]]

        Returns
        -------
        Returns a flatten list : [X, Y, ..., Z]
        """
    return [item for list in listOfList for item in list]


# ----------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Basic filter -------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------


# Filtre des prédictions BERT (Stopwords / Ponctuation / Nombres / [UNK] prédiction d'un token inconnu)
def filterPredictions(resultsPredictions):
    """
    Filtre les prédictions initiales de la fonction getExtendedWords de predictionTools.py et retourne le même objet en ayant retiré
    les stopwords, la ponctuation, les nombres, et le token "[UNK]".

    Parameters
    ----------
    resultsPredictions : list(list(dict())) (ex. [[{"score":_, "token_str":_},...]] )
        Prédictions pour chacun des mots sélectionnés dans le texte d'origine.

    Returns
    -------
    resultsDoc : list(list(dict())) (ex. [[{"score":_, "token_str":_},...]] )
        Prédictions pour chacun des mots sélectionnés dans le texte d'origine filtrées.

    """
    # resultsDoc = []
    # for wordPredictions in resultsPredictions:
    #     resultsWord = []
    #     for prediction in wordPredictions:
    #         token = prediction["token_str"]
    #         if(token != "[UNK]" and not token in stopwords.words() and not token in string.punctuation and len(re.findall("\d", token)) == 0 and token.strip(string.punctuation) != ""):
    #             resultsWord.append(prediction)
    #     resultsDoc.append(resultsWord)
    # return resultsDoc

    tokens = []
    for token in flatten(resultsPredictions):
        token_str=token["token_str"]
        if token_str != "[UNK]" and not token_str in stopwords.words() and not token_str in string.punctuation and len(re.findall("\d", token_str)) == 0 and token_str.strip(string.punctuation) != "":
            tokens.append(token)
    return tokens


# ----------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Score filter -------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------


# Filtre des prédictions BERT par rapport à un score minimal donné
def getTokensFilterByScore(results, scoreThreshold):
    """
    Filtre les prédictions de chaque mot par rapport à leurs scores et au seuil sélectionné.

    Parameters
    ----------
    results : list(list(dict())) (ex. [[{"score":_, "token_str":_},...]] )
        Prédictions pour chacun des mots sélectionnés dans le texte d'origine.
    scoreThreshold : float [0,1]
        Seuil pour le score des prédictions.

    Returns
    -------
    tokens : list<dict> => [{"score":_, "token_str":_},...] 
        Liste des tokens sélectionnés avec leur score respectif.

    """
    tokens = []
    for token in flatten(results):
        if token["score"] >= scoreThreshold:
            tokens.append(token)
    return tokens


# ---------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Perfect predictions ---------------------------------------------
# ---------------------------------------------------------------------------------------------------------------


# dans la liste des tokens (NLTKTokens ou SwTokens), pour un token donné, si BERT a pu prédire le token exact sur les 10 générés, alors on recup la liste des 10 tokens
# Filtre de prédictions correctes
def getCorrectPredictionTokensFilter(tokens, predictions):
    """
    A partir de la liste des tokens et des prédictions pour chaque d'entre eux, recupère les prédictions des mots qui sont contenus dans leurs
    propres prédictions.

    Parameters
    ----------
    tokens : list<str>
        Liste des tokens d'origine.
    predictions : list(list(dict())) (ex. [[{"score":_, "token_str":_},...]] )
        Prédictions pour chacun des mots sélectionnés dans le texte d'origine.

    Returns
    -------
    results : list(dict()) (ex. [{"score":_, "token_str":_},...] )
        Prédictions filtrées pour chacun des mots sélectionnés dans le texte d'origine.

    """

    results = []
    if(len(tokens)<=len(predictions)):      #on itère toujours sur le plus petit, donc si la longueur de 'tokens' est inférieure ou égale à celle de 'prédictions', on itère sur 'tokens'
        
        for i in range(len(tokens)):
            predictionTokens = [p["token_str"] for p in predictions[i]]
            if(tokens[i] in predictionTokens):
                results = results + predictions[i]

    else:                                   #sinon, ca veut dire que 'predictions' est plus petit, donc on itère sur 'predictions'
        for i in range(len(predictions)):
            predictionTokens = [p["token_str"] for p in predictions[i]]
            if(tokens[i] in predictionTokens):
                results = results + predictions[i]

    return results


# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------- ACP centroid filter --------------------------------------------
# --------------------------------------------------------------------------------------------------------------


def token2vec(token):
    """
    Récupère le vecteur d'un mot (si il existe) dans un modèle word2vec

    Parameters
    ----------
    token : str
        Token recherché.

    Returns
    -------
    array<float>
        Vecteur du token ou vide.

    """
    # if token in wv_from_bin.vocab:
    #     # return model[text].round(3)
    #     return wv_from_bin[token]
    # else:
    #     return []
    vec=[]
    if token in wv_from_bin.vocab:
        vec= wv_from_bin[token]
    return vec


def vec2Matrix(tokens):
    """
    Transforme une liste de tokens en une liste de vecteurs 

    Parameters
    ----------
    tokens : list<str>
        Liste de tokens d'origine.

    Returns
    -------
    list<str>, array<array<float>>
        Liste des mots d'origine et d'une matrice rassemblant les vecteurs de chacun de ces mots.

    """
    vecList = []
    nameList = []
    for word in tokens:
        wordVec = token2vec(word)
        if wordVec != []:
            # if word not in nameList: 	#sans repetion, ça marche moins bien
            vecList.append(wordVec)
            nameList.append(word)
    return nameList, np.array(vecList)


def getCentroid(matrice):
    """
    Retourne le centroid d'une ensemble de vecteurs (moyenne des valeurs pour chaque dimension des vecteurs).

    Parameters
    ----------
    matrice : array<array<float>>
        Matrice.

    Returns
    -------
    array<float>
        Vecteur du centroid.

    """
    return matrice.mean(axis=0)


def getNearestNeighborsOfCentroid(centroid, matrice, pourcentSelected, closest=True):
    distances = []
    topk = int(matrice.shape[0]*(pourcentSelected/100))# combien d'éléments finaux on veut, on prend matrice.shape[0] car 'distances' aura la meme longueur à la fin
    for vector in matrice:  # pour chaque row dans la matrice
        distances.append(np.abs(np.linalg.norm(centroid - vector, axis=0)))# np.linalg.norm calcule la norme d'un vecteur et nb.abs la valeur absolue, retour un float
    if(closest):
        return np.argsort(np.asarray(distances))[:topk]# np.argsort retourne les index des éléments triés, par exemple : x = np.array([3, 1, 2]) => np.argsort(x) => array([1, 2, 0])
    else:
        return np.argsort(np.asarray(distances))[:-topk-1:-1]


def getACPCentroidFilteredExtension(rowDoc, pcaDim, pourcentSelected, closest, index=False):
    """
    Filtre une extension en projetant ses vecteurs dans l'ACP du document d'origine et en mesurant la distance de chacun des mots au centroid du document dans cet espace vectoriel.

    Parameters
    ----------
    rowDoc : Series
        Ligne (Serie) d'un dataframe (pandas). Peut fonctionner avec une liste de dictionnaire.
    pcaDim : int
        Dimension PCA sélectionnée.
    pourcentSelected : int [0,100]
        Pourcentage de l'extension à garder.
    closest : bool
        Garder les éléments les plus proches (true) ou les plus éloignés (false).
    index : bool, optional
        Retourne les indexes des mots sélectionnés dans la liste des vecteurs de l'extension. The default is False.

    Returns
    -------
    list<str> or  list<str>, list<int>
        Liste de mots de l'extension finale.

    """
    # if(len(rowDoc["SwTokensVectors"]) > pcaDim):
    #     pca = PCA(n_components=pcaDim, svd_solver='full')
    #     dataMat = pca.fit_transform(rowDoc["SwTokensVectors"]) # SwTokensVectors est une serie/colonne contenant les vecteurs des mots d'origines
    #     centroid = getCentroid(dataMat)
    #     if(len(rowDoc["SwTokensVectorsExtensionBert"]) != 0):
    #         extensionDataMat = pca.transform(rowDoc["SwTokensVectorsExtensionBert"]) # SwTokensVectorsExtension est une serie/colonne contenant les vecteurs des mots de l'extension
    #         indexesNearest = getNearestNeighborsOfCentroid(centroid, extensionDataMat, pourcentSelected, closest)
    #         finalExtension = np.asarray(rowDoc["SwTokensVectorsNamesExtensionBert"])[indexesNearest].tolist()  # SwTokensNamesExtension est une serie/colonne contenant les mots de l'extension qui ont été transformés en vecteurs
    #     else:
    #         finalExtension = []
    #         indexesNearest = []
    # else:
    #     finalExtension = []
    #     indexesNearest = []
    # if(not index):
    #     return finalExtension
    # else:
    #     return finalExtension, indexesNearest

    finalExtension = []
    indexesNearest = []
    if(len(rowDoc["tokens_vectors"]) > pcaDim and len(rowDoc["bertTokens_vectors"]) > 0):
        pca = PCA(n_components=pcaDim, svd_solver='full')
        dataMat = pca.fit_transform(rowDoc["tokens_vectors"]) #tokens_vectors est une serie/colonne contenant les vecteurs des mots d'origines
        centroid = getCentroid(dataMat)
        extensionDataMat = pca.transform(rowDoc["bertTokens_vectors"]) # bertTokens_vectors est une serie/colonne contenant les vecteurs des mots de l'extension
        indexesNearest = getNearestNeighborsOfCentroid(centroid, extensionDataMat, pourcentSelected, closest)
        finalExtension = np.asarray(rowDoc["bertTokens"])[indexesNearest].tolist() #bertTokens est une serie/colonne contenant les mots de l'extension qui ont été transformés en vecteurs
    if(not index):
        return finalExtension
    else:
        return finalExtension, indexesNearest


# -----------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ ACP axes filter ------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------


def sim(comp, pt):
    """
    Similarité cosinus entre deux vecteurs

    Parameters
    ----------
    comp : array<float>
        Premier vecteur.
    pt : array<float>
        Deuxième vecteur.

    Returns
    -------
    Float
    Similarité cosinus

    """
    return (ps.cosine_similarity(comp, pt)[0][0])


def vecClosestSimilarity(dataMat, dim):
    closestSimilarities = []
    for dataVec in dataMat:
        newDataVec = dataVec[:dim]
        alldimValues = []
        for i in range(0, dim):
            dimVec = np.zeros(dim)
            dimVec[i] = 1
            simr = sim([dimVec], [newDataVec])
            alldimValues.append(abs(simr))
        closestSimilarities.append(np.mean(alldimValues))
    return closestSimilarities


def getNearestNeighborsOfAxis(distances, matrice, pourcentSelected, closest=True):
    """
    Filtre une extension en projetant ses vecteurs dans l'ACP du document d'origine et en mesurant la distance de chacun des mots de l'extension aux axes de l'ACP dans cet espace vectoriel

    Parameters
    ----------
    distances : TYPE
        DESCRIPTION.
    matrice : TYPE
        DESCRIPTION.
    pourcentSelected : int [0,100]
        Pourcentage de l'extension à garder.
    closest : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    list<str> or  list<str>, list<int>
        Liste de mots de l'extension finale.

    """
    topk = int(matrice.shape[0]*(pourcentSelected/100))
    
    if(closest):#c'est correct comme ca, car le calcul de mesure n'est pas le meme
        return np.argsort(np.asarray(distances))[:-topk-1:-1]
    else:
        return np.argsort(np.asarray(distances))[:topk]


def getACPAxisFilteredExtension(rowDoc, pcaDim, pourcentSelected, closest, index=False):
    """
    Filtre l'extension à partir d'une représentation vectorielle de tous les mots de l'extension

    Parameters
    ----------
    rowDoc : Series
        Ligne (Serie) d'un dataframe (pandas). Peut fonctionner avec une liste de dictionnaire.
    pcaDim : int
        Dimension PCA sélectionnée.
    pourcentSelected : int [0,100]
        Pourcentage de l'extension à garder.
    closest : bool
        Garder les éléments les plus proches (true) ou les plus éloignés (false).
    index : bool, optional
        Retourne les indexes des mots sélectionnés dans la liste des vecteurs de l'extension. The default is False.

    Returns
    -------
    list<str> or  list<str>, list<int>
        Liste de mots de l'extension finale.

    """
    # if(len(rowDoc["SwTokensVectors"]) > pcaDim):# SwTokensVectors est une serie/colonne contenant les vecteurs des mots d'origines
    #     pca = PCA(n_components=pcaDim, svd_solver='full')
    #     dataMat = pca.fit_transform(rowDoc["SwTokensVectors"])
    #     if(len(rowDoc["SwTokensVectorsExtension"]) != 0):# SwTokensVectorsExtension est une serie/colonne contenant les vecteurs des mots de l'extension
    #         extensionDataMat = pca.transform(rowDoc["SwTokensVectorsExtension"])
    #         results = vecClosestSimilarity(extensionDataMat, pcaDim)
    #         indexesNearest = getNearestNeighborsOfAxis(results, dataMat, pourcentSelected, closest)
    #         finalExtension = np.asarray(rowDoc["SwTokensVectorsNamesExtension"])[ # SwTokensNamesExtension est une serie/colonne contenant les mots de l'extension qui ont été transformés en vecteurs
    #             indexesNearest].tolist()
    #     else:
    #         finalExtension = []
    #         indexesNearest = []
    # else:
    #     finalExtension = []
    #     indexesNearest = []
    # if(not index):
    #     return finalExtension
    # else:
    #     return finalExtension, indexesNearest

    finalExtension = []
    indexesNearest = []
    if(len(rowDoc["tokens_vectors"]) > pcaDim and len(rowDoc["bertTokens_vectors"]) > 0):# tokens_vectors est une serie/colonne contenant les vecteurs des mots d'origines
        pca = PCA(n_components=pcaDim, svd_solver='full')
        dataMat = pca.fit_transform(rowDoc["tokens_vectors"])
        extensionDataMat = pca.transform(rowDoc["bertTokens_vectors"])# bertTokens_vectors est une serie/colonne contenant les vecteurs des mots de l'extension
        results = vecClosestSimilarity(extensionDataMat, pcaDim)
        indexesNearest = getNearestNeighborsOfAxis(results, dataMat, pourcentSelected, closest)
        finalExtension = np.asarray(rowDoc["bertTokens"])[ indexesNearest].tolist()# bertTokens est une serie/colonne contenant les mots de l'extension qui ont été transformés en vecteurs
    if(not index):
        return finalExtension
    else:
        return finalExtension, indexesNearest


# -----------------------------------------------------------------------------------------------------------
# ---------------------------------------- Code qui update chaque doc ---------------------------------------
# -----------------------------------------------------------------------------------------------------------


def updateDocuments(index):

    # query ES qui permet de récupérer tous les docs qui possèdent déjà les BertTokens (car ils permettent de réaliser les filtres) et où ils manquent les filtres (certains/tous les filtres)
    query = {
    "bool": {
      "must": [
        {"exists": {"field": "extension.bertTokens.resultsBertNltkTokenizer"}}
      ], 
      "must_not": [
        {
          "bool": {
            "must": [
              {
                "match_phrase": {
                  "process_status.filters.status": "basic"
                }
              },
              {
                "match_phrase": {
                  "process_status.filters.status": "perfectPredictions"
                }
              },
              {
                "match_phrase": {
                  "process_status.filters.status": "score"
                }
              }
            #   ,{
            #     "match_phrase": {
            #       "process_status.filters.status": "pca_axes"
            #     }
            #   },
            #   {
            #     "match_phrase": {
            #       "process_status.filters.status": "pca_centroid"
            #     }
            #   }
            ]
          }
        }
      ]
    }
  }
    nbFoisAucuneDonnee=0 #permet de compter le nombre de fois qu'il n'a pas recu de nouvelles données, au bout de 10 fois, on arrete le programme
    nbConnexionEchouee=0 #permet de compter le nombre de fois qu'on a échoué la connexion au cluster, au bout de 10 fois, on arrete le programme
    while nbFoisAucuneDonnee<10 and nbConnexionEchouee<10:
        hits=[]
        while nbConnexionEchouee<10: #boucle qui permet de se connecter au cluster ES quand il est prêt 
            try:
                response=elastic.search(index=index, query=query,size=10000)
                hits=response["hits"]["hits"]
                nbConnexionEchouee=0
                break #on sort de la boucle dès que le cluster est prêt et qu'il a recu une reponse
            except Exception as err:
                print("Echec de connexion : ",err)
                nbConnexionEchouee+=1
                sleep(10)
    
        if hits!=[]: #s'il y a des reponses
            nbFoisAucuneDonnee=0 #on reinitialise le compteur à 0
            for i in range(0,len(hits)): #on parcourt tous les docs un par un pour les update
                try:
                    doc_id=hits[i]["_id"]
                    doc_tokens=hits[i]["_source"]["tokens"]
                    doc_bertTokens=hits[i]["_source"]["extension"]["bertTokens"]["resultsBertNltkTokenizer"]

                    elastic.update(index=index, id=doc_id,doc={"process_status": {"filters":{"start_process": datetime.now(timezone.utc), "status": "adding filters"}}})

                    filters={}
                    process_status=""

                    if("filters" not in hits[i]["_source"]["extension"]):# si le field 'filters' n'existe pas encore, alors on doit créer tous les filtres 
                    
                        basic_filter=filterPredictions(doc_bertTokens)
                        score_filter=getTokensFilterByScore(doc_bertTokens,0.5)
                        perfectPredictions_filter=getCorrectPredictionTokensFilter(doc_tokens,doc_bertTokens)

                        # doc_bertTokens_str=[token["token_str"] for token in flatten(doc_bertTokens)]#on récupère tous les 'token_str' des bertTokens (on retire les scores)
                        # bertTokens,bertTokens_vectors= vec2Matrix(doc_bertTokens_str)
                        # tokens,tokens_vectors=vec2Matrix(doc_tokens)# la variable 'tokens' nous sert à rien mais c'est pour que la fonction soit plus générique
                        # doc={"tokens_vectors":tokens_vectors,"bertTokens":bertTokens,"bertTokens_vectors":bertTokens_vectors}
                        # PCA_centroid_filter=getACPCentroidFilteredExtension(doc, 2, 50, True)
                        # PCA_axes_filter=getACPAxisFilteredExtension(doc, 2, 50, True)
                    
                        # filters={"basic":basic_filter,"score":score_filter,"perfectPredictions":perfectPredictions_filter,"pca_axes":PCA_axes_filter,"pca_centroid":PCA_centroid_filter}
                        filters={"basic":basic_filter,"score":score_filter,"perfectPredictions":perfectPredictions_filter}
                        # process_status=" basic score perfectPredictions pca_axes pca_centroid"
                        process_status=" basic score perfectPredictions"
                    
                    else:# si le field 'filters' existe déjà dans le document, ça veut dire qu'il lui manque uniquement certains filtres, alors on vérifie 1 par 1 lequel est manquant
                        process_status=hits[i]["_source"]["process_status"]["filters"]["status"]#comme au moins un filtre existe déjà, alors le process_status existe déjà, on le récup pour le mettre à jour
        
                        if("basic" not in hits[i]["_source"]["extension"]["filters"]):
                            basic_filter=filterPredictions(doc_bertTokens)
                            filters["basic"]=basic_filter
                            process_status+=" basic"

                        if("score" not in hits[i]["_source"]["extension"]["filters"]):
                            score_filter=getTokensFilterByScore(doc_bertTokens,0.5)
                            filters["score"]=score_filter
                            process_status+=" score"

                        if("perfectPredictions" not in hits[i]["_source"]["extension"]["filters"]):
                            perfectPredictions_filter=getCorrectPredictionTokensFilter(doc_tokens,doc_bertTokens)
                            filters["perfectPredictions"]=perfectPredictions_filter
                            process_status+=" perfectPredictions"

                        # if("pca_centroid" not in hits[i]["_source"]["extension"]["filters"]):
                        #     doc_bertTokens_str=[token["token_str"] for token in flatten(doc_bertTokens)]#on récupère tous les 'token_str' des bertTokens (on retire les scores)
                        #     bertTokens,bertTokens_vectors= vec2Matrix(doc_bertTokens_str)
                        #     tokens,tokens_vectors=vec2Matrix(doc_tokens)# la variable 'tokens' nous sert à rien mais c'est pour que la fonction soit plus générique
                        #     doc={"tokens_vectors":tokens_vectors,"bertTokens":bertTokens,"bertTokens_vectors":bertTokens_vectors}
                        #     PCA_centroid_filter=getACPCentroidFilteredExtension(doc, 2, 50, True)
                        #     filters["pca_centroid"]=PCA_centroid_filter
                        #     process_status+=" pca_centroid"
                    
                        # if("pca_axes" not in hits[i]["_source"]["extension"]["filters"]):
                        #     doc_bertTokens_str=[token["token_str"] for token in flatten(doc_bertTokens)]#on récupère tous les 'token_str' des bertTokens (on retire les scores)
                        #     bertTokens,bertTokens_vectors= vec2Matrix(doc_bertTokens_str)
                        #     tokens,tokens_vectors=vec2Matrix(doc_tokens)# la variable 'tokens' nous sert à rien mais c'est pour que la fonction soit plus générique
                        #     doc={"tokens_vectors":tokens_vectors,"bertTokens":bertTokens,"bertTokens_vectors":bertTokens_vectors}
                        #     PCA_axes_filter=getACPAxisFilteredExtension(doc, 2, 50, True)
                        #     filters["pca_axes"]=PCA_axes_filter
                        #     process_status+=" pca_axes"
                    
                    elastic.update(index=index, id=doc_id,doc={"extension": {"filters": filters},"process_status": {"filters":{"end_process": datetime.now(timezone.utc), "status": "finished adding filters :"+process_status}}})
                
                except Exception as err:
                    print("Erreur :",err)
        else: #s'il n'y a pas de réponse,
            print("pas de réponse")
            nbFoisAucuneDonnee+=1 #alors on augmente le compteur de 1

        sleep(10) #après avoir terminé tous les docs ou s'il n'y a eu aucun nouveau doc, on attend 10 secondes


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------ Main ------------------------------------------------
# ------------------------------------------------------------------------------------------------------


### 1ERE SOLUTION 

# Extraire le PKL du BZ2 puis..
wv_from_bin = Wikipedia2Vec.load("enwiki_20180420_100d.pkl")
wv_from_bin.save_text("test", out_format='word2vec')
wv_from_bin = KeyedVectors.load_word2vec_format("test")
wv_from_bin.save_word2vec_format("enwiki_20180420_100d.bin", binary=True)
wv_from_bin = KeyedVectors.load_word2vec_format("enwiki_20180420_100d.bin", binary=True)

### 2EME SOLUTION MAIS TU VAS DEVOIR TELECHARGER LA VERSION .txt DU WORD2VEC

# Extraire le TXT du BZ2 puis..
# wv_from_bin = KeyedVectors.load_word2vec_format("E:/Data/enwiki_20180420_100d.txt")
# wv_from_bin.save_word2vec_format("enwiki_20180420_100d.bin", binary=True)
# wv_from_bin = KeyedVectors.load_word2vec_format("enwiki_20180420_100d.bin", binary=True)


updateDocuments("antique")
updateDocuments("nfcorpus")
