from time import sleep
from elasticsearch import Elasticsearch
from datetime import datetime,timezone
import os

from itertools import tee
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import torch
from torch.nn import functional as F
from transformers import pipeline
from transformers import logging
logging.set_verbosity_error()

tokenizer = RegexpTokenizer(r'\w+')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download("stopwords")
stop_words = set(stopwords.words('english')) 


# --------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Connexion au cluster ES ------------------------------------------
# --------------------------------------------------------------------------------------------------------------


es_password=os.environ.get("ELASTIC_PASSWORD")#permet de récuperer la variable d'environnement 'ELASTIC_PASSWORD'
elastic = Elasticsearch("https://elastic:"+es_password+"@es1:9200",ca_certs="/usr/src/app/certs/ca/ca.crt")


# -----------------------------------------------------------------------------------------------------------
# -------------------------------------------- Prediction tools ---------------------------------------------
# -----------------------------------------------------------------------------------------------------------


# retourne un tableau contenant des tuples qui shift de 1 en 1( => chaque tuple ayant une taille de 'size' )
# exemple : window([1,2,3,4,5,6,7,8,9,10],3)=[(1,2,3),(2,3,4),(3,4,5),(4,5,6),(5,6,7),(6,7,8),(7,8,9),(8,9,10)]
def window(iterable, size=510):
    """Create list of shifting window of specified size on an iterable list (e.g. for size 2 :  [1, 2, 3] -> [[1, 2], [2, 3]])

    Parameters
    ----------
    iterable : iterable object
        Origin on which the window is shifting
    size : integer
        Size of the shifting window
    """
    iters = tee(iterable, size) 
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return list(zip(*iters))
        
#retourne vrai ou faux : regarde si le token est assez important pour être prédit
def checkToken(token): #considerate also as the tokenizer
    """Evaluate if a token is important enough to be predicted. Returns boolean (True: important, False:unimportant)

    Parameters
    ----------
    token : string
        Token evaluated
    """
    token = token.lower().strip(string.punctuation) 
    return len(token)>1 and len(re.findall("\d", token))==0  and not token in stopwords.words() and not token in string.punctuation # '\d' correspond à un chiffre => [0-9]

#prend une phrase en entrée, masque un mot à la position 'index' et retourne une liste de mots possibles avec pour chacun leurs proba associés
def getPredictions(tokens, index, unmasker):
    """Out of a list of tokens, give top10 words (with their probabilities) predicted by the MASK-model for the word at the index position.
       Returns list of predictions  containing each word and associated probability predicted
    
    Parameters
    ----------
    tokens : list(string)
        Origin list of tokens
    index : integer
        Postion of the masked word
    unmasker : object
        Pipeline for a "fill-mask" model
    """
    baseTokens = tokens.copy()#on créé une copie des tokens
    baseTokens[index] = "[MASK]"#on masque le mot à l'index 'index'
    inputsIds = unmasker.tokenizer.convert_tokens_to_ids(baseTokens)#convertit les tokens (List[str]) en id (List[int])
    inputs = {}
    inputs["input_ids"] = torch.tensor([inputsIds])
    inputs["attention_mask"] = torch.zeros([1, len(baseTokens)], dtype=torch.int32).long()#un tensor avec une longueur égale au nombre de tokens, .long() permet de mettre les nombres en int
    inputs["token_type_ids"] = torch.zeros([1, len(baseTokens)], dtype=torch.int32).long()#répétition ? dtype=torch.int32 et .long() font la meme chose
    outputs = unmasker.model(**inputs)
    mask_index = torch.where(inputs["input_ids"][0] == unmasker.tokenizer.mask_token_id)
    logits = outputs.logits#The logits are the output of the BERT Model before a softmax activation function is applied to the output of BERT. In order to get the logits, we have to specify return_dict = True in the parameters when initializing the model, otherwise, the above code will result in a compilation error
    softmax = F.softmax(logits, dim = -1)#on applique softmax
    mask_word = softmax[0, mask_index, :]
    top_10_score = torch.topk(mask_word, 10, dim = 1)[0][0].tolist()#Returns the 10 largest elements of the given input tensor along a given dimension.
    top_10_token = torch.topk(mask_word, 10, dim = 1)[1][0]
    top_10_token = [unmasker.tokenizer.decode([token]) for token in top_10_token]#reconvertir les ID en texte (List[int]->List[str])
    predictions = []
    for i in range(len(top_10_score)):
        predictions.append({"score":top_10_score[i], "token_str":top_10_token[i]})#liste des tokens et leurs scores respectifs
    return predictions


def getTokens(text, unmaskerBert):
    """Get all the predictions for the input text based on the two "fill-mask" models based on each window. 
    All results are merged into one list ([{"score":_, "token_str":_},...]).
    
    Parameters
    ----------
    text : string
        text of the document
    unmaskerBert : object
        Pipeline for a "fill-mask" model pre-trained on Wiki/Roman
    unmaskerBio : object
        Pipeline for a "fill-mask" model pre-trained on PUBMED/MIMIC
    name : string
        Name of the document (for debugging)
    """
    resultsBert = []
    textTokens = nltk.tokenize.word_tokenize(text) # Good muffins cost $3.88\nin New York.  Please buy me two of them.\n\nThanks. => ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.','Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']
    textTokens = [word.lower() for word in textTokens] #on met tout en minuscule
    if(len(textTokens) < 510): #si la longueur du texte est < 510 , alors pas besoin de fenetrage 'glissant'
        windows = [textTokens]
        for tokens in list(windows): 
            tokens = list(tokens) 
            for i in range(0, len(tokens)//2): #on accede mot par mot
                token = tokens[i]
                if checkToken(token):
                    resultsBert.append(getPredictions(tokens, i, unmaskerBert))
            for i in range(len(tokens)//2, len(tokens)): #pourquoi on divise en 2 ?
                token = tokens[i]
                if checkToken(token):
                    resultsBert.append(getPredictions(tokens, i, unmaskerBert))
    else: #sinon, on doit créer le fenetrage glissant
        windows = window(textTokens)
        indexLast = len(list(windows)) - 1 # index de la derniere fenetre
        for index, tokens in enumerate(list(windows)): #on se balade dans chaque fenetre
            tokens = list(tokens) #la fenetre actuelle
            if(index == 0): #si c'est la premiere fenetre alors,
                for i in range(0, len(tokens)//2 + 1): #on se balade de l'index 0 à la moitié de la fenetre
                    token = tokens[i] #le mot de la phrase 
                    if checkToken(token):
                        resultsBert.append(getPredictions(tokens, i, unmaskerBert))
            if(index == indexLast): #si c'est la derniere fenetre, alors on se balade de la motiié du tableau jusqu'à la fin
                for i in range(len(tokens)//2, len(tokens)):
                    token = tokens[i]
                    if checkToken(token):
                        resultsBert.append(getPredictions(tokens, i, unmaskerBert))
            if(index != indexLast and index != 0): #si on est pas au debut ou fin, alors on prend le mot au milieu de la fenetre actuelle
                token = tokens[len(tokens)//2]
                if checkToken(token):
                    resultsBert.append(getPredictions(tokens, len(tokens)//2, unmaskerBert))
    return {'resultsBertNltkTokenizer' : resultsBert}


# -----------------------------------------------------------------------------------------------------------
# ---------------------------------------- Code qui update chaque doc ---------------------------------------
# -----------------------------------------------------------------------------------------------------------


def updateDocuments(unmaskerBert,index):

    # query ES qui permet de recup tous les docs qui n'ont pas le field 'process_status.bertTokens.end_process', car ce field est ajouté une fois que les tokens sont ajoutés
    query = {
    "bool": {
      "must_not": [
        {
          "exists": {
            "field": "process_status.bertTokens.end_process"
          }
        }
      ]
    }
  }

    nbFoisAucuneDonnee=0#permet de compter le nombre de fois qu'il n'a pas recu de nouvelles données, au bout de 10 fois, on arrete le programme
    nbConnexionEchouee=0#permet de compter le nombre de fois qu'on a échoué la connexion au cluster, au bout de 10 fois, on arrete le programme

    while nbFoisAucuneDonnee<10 and nbConnexionEchouee<10:
        hits=[]
        while nbConnexionEchouee<10:#boucle qui permet de se connecter au cluster ES quand il est prêt 
            try:
                response=elastic.search(index=index, query=query,size=10000)
                hits=response["hits"]["hits"]
                nbConnexionEchouee=0
                break#on sort de la boucle dès que le cluster est prêt et qu'il a recu une reponse
            except Exception as err:
                print("Echec de connexion : ",err)
                nbConnexionEchouee+=1
                sleep(10)

        if hits!=[]:#s'il y a des reponses
            nbFoisAucuneDonnee=0#on reinitialise le compteur à 0
            for i in range(0,len(hits)):#on parcourt tous les docs un par un pour les update
                try:
                    doc_id=hits[i]["_id"]
                    doc_abstract=hits[i]["_source"]["ABSTRACT"]

                    elastic.update(index=index, id=doc_id,doc={"process_status": {"bertTokens":{"start_process": datetime.now(timezone.utc), "status": "adding tokens"}}})
                
                    tokens=getTokens(doc_abstract,unmaskerBert)
                
                    elastic.update(index=index, id=doc_id,doc={"extension": {"bertTokens": tokens},"process_status": {"bertTokens":{"end_process": datetime.now(timezone.utc), "status": "finished adding tokens"}}})

                except Exception as err:
                    print("Erreur :",err)

        else:#s'il n'y a pas de réponse, alors on augmente le compteur 'nbFoisAucuneDonnee' de 1
            print("pas de réponse")
            nbFoisAucuneDonnee+=1

        sleep(10)#après avoir terminé tous les docs ou s'il n'y a eu aucun nouveau doc, on attend 10 secondes


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------ Main ------------------------------------------------
# ------------------------------------------------------------------------------------------------------


# !!!! bertModel : A REMPLACER PAR LES MODELES DEFINITIFS !!!!
bertModel = pipeline('fill-mask', model='google/bert_uncased_L-4_H-256_A-4')
# updateDocuments(bertModel,"antique")
updateDocuments(bertModel,"nfcorpus")





        










