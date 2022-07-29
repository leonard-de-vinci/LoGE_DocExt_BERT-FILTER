import os
import string
import re
from enum import Enum
from time import sleep
from typing import final

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from elasticsearch import Elasticsearch

import nltk
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download("stopwords")


# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Paramètres de FastAPI ------------------------------------------
# ------------------------------------------------------------------------------------------------------------


app = FastAPI()

#https://fastapi.tiangolo.com/tutorial/cors/

# 'origins' correspond aux adresses qu'on autorise à envoyer des request
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Connexion au cluster --------------------------------------------
# ---------------------------------------------------------------------------------------------------------------


es_password = os.environ.get("ELASTIC_PASSWORD")# permet de récuperer la variable d'environnement
elastic = Elasticsearch("https://elastic:"+es_password +"@es1:9200", ca_certs="/usr/src/app/certs/ca/ca.crt")


# -----------------------------------------------------------------------------------------------------------------
# ---------------- Déclaration des classes (permet à FastAPI de restreindre les valeurs possibles) ----------------
# -----------------------------------------------------------------------------------------------------------------


class IndexOptions(str, Enum):
    antique = "antique"
    antiquebm25="antiquebm25"
    nfcorpus = "nfcorpus"
    nfcorpusbm25 = "nfcorpusbm25"


class FiltersOptions(str, Enum):
    abstract = "abstract"
    bertTokens = "bertTokens"
    basic ="basic"
    score="score"
    perfectPredictions="perfectPredictions"
    pca_centroid="pca_centroid"
    pca_axes="pca_axes"

#permet de match un filter avec son field Elasticsearch
match_filter_with_ESfield = {
    FiltersOptions.abstract: "ABSTRACT",
    FiltersOptions.bertTokens: "extension.bertTokens.resultsBertNltkTokenizer.token_str",
    FiltersOptions.basic: "extension.filters.basic.token_str",
    FiltersOptions.score: "extension.filters.score.token_str",
    FiltersOptions.perfectPredictions: "extension.filters.perfectPredictions.token_str"
}


# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Fonctions globales  --------------------------------------------
# --------------------------------------------------------------------------------------------------------------


def checkToken(token): #considerate also as the tokenizer
    """Evaluate if a token is important enough to be predicted. Returns boolean (True: important, False:unimportant)

    Parameters
    ----------
    token : string
        Token evaluated
    """
    token = token.lower().strip(string.punctuation) 
    return len(token)>1 and len(re.findall("\d", token))==0  and not token in stopwords.words() and not token in string.punctuation # '\d' correspond à un chiffre => [0-9]


def getTextTokens(text):
    return [word.lower() for word in nltk.tokenize.word_tokenize(text)]


def flatten(listOfList):
    """
        Takes a list of lists and flattens it 

        Parameters
        ----------
        listOfList : list(list)
            [[X], [X], ..., [X]]

        Returns
        -------
        list()
            Returns a flatten list => [X, X, ..., X]
        """
    return [item for list in listOfList for item in list]


def getScoreOfWord(word, TokensAndScores):
    """
        Check where the word matches the token_str in the TokensAndScores, and increases the score of the word if that's the case.

        Parameters
        ----------
        word : str
        
        TokensAndScores : list(dict)
            [{'score':X, 'token_str':'string'} , ... , {'score':X, 'token_str':'string'}]

        Returns
        -------
        float
            Returns the score of the word of the query
        """
    score = 0
    for token in TokensAndScores:
        if token["token_str"]==word:
            score+=token["score"]
    return round(score,3)


def getFieldOfDoc(doc,filter):
    """
        Takes an Elasticsearch document and a filter, and returns the corresponding field of this document

        Parameters
        ----------
        doc : dict
            Elasticsearch document

        filter : str 
            The field we are looking for, corresponds to a filter

        Returns
        -------
        list()
            Returns the field of the Elasticsearch document
        """
    # Remarque : On ne peut pas juste faire doc[field] car les fields sont nested et ils n'ont d'ailleurs pas forcément le meme nom que leur filtre (bertTokens != resultsBertNltkTokenizer)
    if filter == FiltersOptions.abstract:
        result = doc["_source"]["abstract"]
    elif filter == FiltersOptions.bertTokens:
        result = doc["_source"]["extension"]["bertTokens"]["resultsBertNltkTokenizer"]
    elif filter == FiltersOptions.basic:
        result = doc["_source"]["extension"]["filters"]["basic"]
    elif filter == FiltersOptions.score:
        result = doc["_source"]["extension"]["filters"]["score"]
    elif filter == FiltersOptions.perfectPredictions:
        result = doc["_source"]["extension"]["filters"]["perfectPredictions"]
    elif filter == FiltersOptions.pca_axes:
        result = doc["_source"]["extension"]["filters"]["pca_axes"]
    elif filter == FiltersOptions.pca_centroid:
        result = doc["_source"]["extension"]["filters"]["pca_centroid"]
    return result


def getTokensAndScores(doc,filter):
    """
        This function allows to retrieve the list of token_str and score in a flatten list from the filter of an Elasticsearch document

        Parameters
        ----------
        doc : dict
            Elasticsearch document

        filter : str 
            bertTokens, basic, score, perfectPredictions, pca_axes, pca_centroid

        Returns
        -------
        list(dict)
            Returns the list of tokens with their respective scores => [{ token_str, score }, ... , { token_str, score }]
        """
        #Remarque : cette fonction permet de récupérer la liste des token_str et score en une flatten list
    if filter==FiltersOptions.bertTokens:
        return flatten(getFieldOfDoc(doc,filter))
    elif filter==FiltersOptions.score or filter==FiltersOptions.perfectPredictions or filter==FiltersOptions.basic:
        return getFieldOfDoc(doc,filter)
    elif filter==FiltersOptions.pca_axes or filter==FiltersOptions.pca_centroid:
        return "doSomething"


def getExistingFiltersInDocument(doc,filtersList):
    """
    Takes a list of filters in parameter and checks if they are all present in the Elasticsearch document, returns only those present

    Parameters
    ----------
    doc : Elasticsearch document
    
    filtersList : list
        List of filters 

    Returns
    -------
    list()
        Returns only the filters present (filters that are already generated in Elasticsearch) in the Elasticsearch document
    """
    # Remarque : je mets chaque filtre dans un try-except pour éviter de crash dans le cas où le field n'existe pas encore
    existingFilters=[]
    for filter in filtersList:
        try:
            getFieldOfDoc(doc,filter)#permet de vérifier si le field existe, s'il crash, c'est que le field n'existe pas, on passe donc dans l'except et on n'ajoute pas le filtre à la liste finale
            existingFilters.append(filter)
        except :
            pass

    return existingFilters


def getPerfectlyPredictedWords(originalTokens,bertTokens):
    """
    Takes the original list of tokens and checks if each one of them are perfectly predicted or not (if Bert successfully predicted the word)

    Parameters
    ----------
    originalTokens : list<str>
        The list of original tokens 
    
    bertTokens : list(list<dict>)
        The list of tokens generated by BERT => [[{token_str, score}, ...], ..., [{token_str, score}, ...]]
        
    Returns
    -------
    list<str>
        Returns the list of perfectly predicted tokens
    """
    perfectlyPredicted=[]
    if len(originalTokens)<=len(bertTokens):
        for i in range(len(originalTokens)):
            if originalTokens[i] in [token["token_str"] for token in bertTokens[i]]:
                perfectlyPredicted.append(originalTokens[i])
    else:
        for i in range(len(bertTokens)):
            if originalTokens[i] in [token["token_str"] for token in bertTokens[i]]:
                perfectlyPredicted.append(originalTokens[i])
    return perfectlyPredicted           


# --------------------------------------------------------------------------------------------------------------------------------------
# ---------------- Endpoint : retourne la liste des documents les plus proches de la question saisie par l'utilisateur  ----------------
# --------------------------------------------------------------------------------------------------------------------------------------


@app.get("/search/{index}/{question}/")
async def getDocs(index: IndexOptions, question: str, filters: list[FiltersOptions] = Query(default=[FiltersOptions.abstract]), limit: int = 10):
    
    # VERIFICATION SUR LIMIT - toutes les autres vérifications sur les query parameters (les data types (int/str) et valeurs prédéfinies sont déjà gérés par fastAPI)

    if limit > 50:
        raise HTTPException(status_code=400, detail="Limit too high")
    elif limit <= 0:
        raise HTTPException(status_code=400, detail="Limit too low")
    
    filters = set(filters)  # permet de s'assurer de l'unicité des éléments
    question_tokens = getTextTokens(question)# transforme la question saisie par l'utilisateur en tokens, permet de chercher dans les filtres dans ES
    question_tokens = [token for token in question_tokens if checkToken(token)]

    # CRÉATION DE LA QUERY - quand l'utilisateur sélectionne des filtres dans le frontend, on doit mettre à jour la query Elasticsearch

    query = {"bool": {
        "should": []
    }} 

    for filter in filters: # pour chaque filtre sélectionné par l'utilisateur dans le frontend, on ajoute un more_like_this dans le should
        if(filter == FiltersOptions.abstract):# si le filtre est abstract, alors on cherche uniquement avec la question 'originale' (str) dans l'abstract
            query["bool"]["should"].append({"more_like_this": {
                "fields": [
                    match_filter_with_ESfield[filter]
                ],
                "like": question,
                "min_term_freq": 1,
                "max_query_terms": 25}})
        else:  # sinon, on cherche avec la question 'tokenisée' (list(str)) dans les bertTokens ou filters 
            if len(question_tokens)>0:
                query["bool"]["should"].append({"more_like_this": {
                    "fields": [
                        match_filter_with_ESfield[filter]
                    ],
                    "like": question_tokens,
                    "min_term_freq": 1,
                    "max_query_terms": 25}})

    print("Query :",query,"\nQuestion_tokens :",question_tokens)

    connectionFailed = 0  # permet de compter le nombre de fois qu'on a échoué la connexion au cluster, au bout de 10 fois, on arrete le programme
    codeError = False
    while connectionFailed < 10 and codeError==False:
        try:
            response = elastic.search(index=index, query=query, size=limit, _source=["ABSTRACT"])
            try:
                hits = response["hits"]["hits"]
                connectionFailed = 0
                final_response = []
                for hit in hits: # pour chaque document Elasticsearch
                    final_response.append( {"doc_id": hit["_id"], "score": hit["_score"], "abstract": hit["_source"]["ABSTRACT"]}) # on récupère uniquement l'ID, le score et l'abstract
                
                return {"documents": final_response}
            except Exception as err:
                print("Erreur dans le code :",err)
                codeError=True

        except Exception as err:
            print("Connexion échouée :", err)
            connectionFailed += 1
            sleep(10)

    raise HTTPException(status_code=404, detail="Cannot connect to the elastic cluster")# si on n'a pas réussi à retourner la liste de documents, alors on raise une erreur


# ----------------------------------------------------------------------------------------------------------------------------------------
# -------------------------- Endpoint : retourne toutes les informations concernant un document en particulier  --------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------


@app.get("/search_doc/{index}/{question}/{doc_id}/")
async def getDoc(index: IndexOptions, question: str, doc_id: str, filters: list[FiltersOptions] = Query(default=[FiltersOptions.abstract])):
    query = {
        "match": {
            "_id": doc_id
        }
    }
    connectionFailed = 0  # permet de compter le nombre de fois qu'on a échoué la connexion au cluster, au bout de 10 fois, on arrete le programme
    codeError = False
    while connectionFailed < 10 and codeError == False:
        try:
            response = elastic.search(index=index, query=query, size=1)
            try:
                connectionFailed = 0 # on le réinitialise à 0 car on a réussi à se connecter au cluster
                hits = response["hits"]["hits"][0] # => [0] car un seul résultat, puisqu'on cherche un seul document avec un ID spécifique
                final_response = {"doc_id": hits["_id"], "abstract": hits["_source"]["ABSTRACT"], "nbWordsInAbstract": len(hits["_source"]["ABSTRACT"].translate(str.maketrans('', '', string.punctuation)).split())}

                # PRÉTRAITEMENT DE LA LISTE DES FILTRES - on traite les filtres sélectionnés par l'utilisateur afin de ne garder que ceux qui sont présents dans le document Elasticsearch et non-nuls

                filters = set(filters)# permet de s'assurer de l'unicité des éléments
                if "abstract" in filters : filters.remove("abstract")# on retire 'abstract' de la liste des filtres

                # FILTRES EXISTANTS - création de la liste des filtres existants et non-vides dans le document Elasticsearch par rapport à ceux demandés

                existingFilters=getExistingFiltersInDocument(hits,filters)# prend la liste des filtres demandés dans le frontend, et supprime tous les filtres qui n'existent pas ou sont vides dans le document Elasticsearch

                # OCCURRENCE DANS ABSTRACT - on calcule l'occurence des mots de la query dans l'abstract
               
                question_without_punctuation = question.lower().translate( str.maketrans('', '', string.punctuation)) # on retire la ponctuation de la question afin de pouvoir correctement compter les occurrences de chaque mot dans l'abstract
                abstract_without_punctuation = hits["_source"]["ABSTRACT"].lower().translate( str.maketrans('', '', string.punctuation))# on retire la ponctuation de l'abstract afin de pouvoir correctement compter les occurrences de chaque mot
                occurrencesOfQueryWords = {"abstract": sorted([{"word": word, "occurrence": abstract_without_punctuation.split().count(word)} for word in set(question_without_punctuation.split())], key=lambda x: x['occurrence'], reverse=True)}

                # OCCURRENCE & SCORE DANS FILTRES - on calcule l'occurence et le score des mots de la query par filtre
                
                occurrenceWordsInFilters = [] # liste des mots avec leurs occurrences par filtre respectifs => [{"word":str , "bertTokens":int ...}, ... , {"word":str , "bertTokens":int ...}]
                scoreWordsInFilters=[] # liste des mots avec leurs scores par filtre respectifs => [{"word":str , "bertTokens":float ...}, ... , {"word":str , "bertTokens":float ...}]
                question_tokens = getTextTokens(question) # transforme la question saisie par l'utilisateur (str) en liste de tokens (list[str])

                if len(filters)> 0 and len(question_tokens)>0: # si l'utilisateur a selectionné au moins 1 filtre, qu'il existe ou pas, dans le document Elasticsearch et que la question n'est pas vide
                    
                    token_str_list_perFilter={} # rassemble tous les token_str de chaque filtre => {"bertTokens":["token_str",..,"token_str"] , "score":["token_str",..,"token_str"]}
                    
                    for filter in existingFilters:
                        token_str_list = [] # correspond a la liste des token_str pour le filtre en question
                        for token in getTokensAndScores(hits,filter): # à partir de la liste des 'token_str' et 'score' retournés par la fonction getTokensAndScores(), on récupère uniquement le token_str de chaque token
                            token_str_list.append(token["token_str"])
                        token_str_list_perFilter[filter]=token_str_list

                    for word in set(question_tokens):
                        currentWord_occurrence={"word":word} # correspond à un mot qu'on va ajouter ensuite à occurrenceWordsInFilters une fois terminé => {"word":str , "bertTokens":int ...}
                        currentWord_score={"word":word} # correspond à un mot qu'on va ajouter ensuite à scoreWordsInFilters une fois terminé => {"word":str , "bertTokens":float ...}
                        for filter in filters: 
                            if filter in existingFilters: # pour chaque filtre qui existe et est non-nul dans le document Elasticsearch, on va calculer l'occurrence et le score du currentWord
                                 
                                currentWord_occurrence[filter]=token_str_list_perFilter[filter].count(word)
                                currentWord_score[filter]=getScoreOfWord(word,getTokensAndScores(hits,filter))

                            else: # si le filtre dans le document Elasticsearch n'existe pas ou est vide, on met l'occurrence et le score à : "-"         
                                    
                                currentWord_occurrence[filter]="-"
                                currentWord_score[filter]="-"

                        occurrenceWordsInFilters.append(currentWord_occurrence)
                        scoreWordsInFilters.append(currentWord_score)

                if occurrenceWordsInFilters!=[] : occurrencesOfQueryWords["filters"] = occurrenceWordsInFilters # si les occurences des mots par filtre sont non nulles, alors on rajoute à la final_response
                final_response["occurrenceOfQueryWords"] = occurrencesOfQueryWords
                if scoreWordsInFilters!=[] : final_response["scoresOfQueryWords"]=scoreWordsInFilters # si les scores des mots par filtre sont non nuls, alors on rajoute à la final_response

                # RETURN FINAL
                
                return {"document": final_response}

            except Exception as err:
                print("Erreur dans le code :", err)
                codeError = True
        except Exception as err:
            print("Connexion échouée : ", err)
            connectionFailed += 1
            sleep(10)

    raise HTTPException(status_code=404, detail="Cannot connect to the elastic cluster") # si on n'a pas réussi à retourner la liste de documents, alors on raise une erreur


# -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------- Endpoint : retourne tous les bertTokens d'un document en particulier  -------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------


@app.get("/tokens/{index}/{doc_id}/")
async def getTokensList(index: IndexOptions, doc_id: str):
    query = {
        "match": {
            "_id": doc_id
        }
    }

    connectionFailed = 0  # permet de compter le nombre de fois qu'on a échoué la connexion au cluster, au bout de 10 fois, on arrete le programme
    codeError = False
    while connectionFailed < 10 and codeError == False:
        try:
            response = elastic.search(index=index, query=query, size=1)
            try:
                hit = response["hits"]["hits"][0] # car un seul résultat
                final_response = {}

                filters=[filter.value for filter in FiltersOptions]
                if "abstract" in filters : filters.remove("abstract")# on retire 'abstract' de la liste des filtres
                existingFilters=getExistingFiltersInDocument(hit,filters)
                
                if "bertTokens" in existingFilters:# si les tokens générés par BERT existent et ne sont pas vides

                    token_str_list_perFilter={} # rassemble tous les token_str de chaque filtre => {"bertTokens":["token_str",..,"token_str"] , "score":["token_str",..,"token_str"]}
                    TotalNbTokens_perFilter={}

                    for filter in filters:
                        if filter in existingFilters:
                            token_str_list = [] # correspond a la liste des token_str pour le filtre en question
                            for token in getTokensAndScores(hit,filter): # à partir de la liste des 'token_str' et 'score' retournés par la fonction getTokensAndScores(), on récupère uniquement le token_str de chaque token
                                token_str_list.append(token["token_str"])
                            token_str_list_perFilter[filter]=token_str_list
                            TotalNbTokens_perFilter[filter]=len(token_str_list)
                        else:
                            TotalNbTokens_perFilter[filter]="-"
                    final_response["totalNbTokensPerFilter"]=TotalNbTokens_perFilter

                    wordList=[]
                    for word in set(token_str_list_perFilter["bertTokens"]):
                        currentWord={"word":word} 
                        for filter in filters:
                            if filter in existingFilters:
                                currentWord[filter]=token_str_list_perFilter[filter].count(word)
                            else:
                                currentWord[filter]="-"
                        wordList.append(currentWord)
                    final_response["tokensList"]=wordList
                return {"tokens":final_response}
            except Exception as err:
                print("Erreur dans le code :", err)
                codeError = True
        except Exception as err:
            print("Connexion échouée :", err)
            connectionFailed += 1
            sleep(10)
        
    raise HTTPException(status_code=404, detail="Cannot connect to the elastic cluster")# si on n'a pas réussi à retourner la liste de documents, alors on raise une erreur






# commande pour start l'API : uvicorn main:app --reload