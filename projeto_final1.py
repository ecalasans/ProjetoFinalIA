# -*- coding: utf-8 -*-

import nltk
from nltk.stem import RSLPStemmer
from collections import OrderedDict
import time
from nltk.corpus import stopwords
import fuzzyText

stemmer = RSLPStemmer() #extração dos radicais das palavras

palavras = ['jogador', 'futebol'] # dicioario

arquivo = open('futebol.txt', 'r') #abrindo o arquivo

texto = arquivo.read() # texto a ser comparado

words = (nltk.word_tokenize(texto)) #texto tokenrizsdo

stops = set(stopwords.words("portuguese")) #conjunto de palavras sem importância para a classificação do texto

word_features = ([w for w in words if not w in stops]) #stop words removioda do texto

#Procurando match total
def matchWord(texto, dicionario):
    bag_words ={} # Inicia o dicionario que vai guardar os matchs do dicionário no texto
    for i in dicionario:
        bag_words[i] = 0

    # Procura os matchs entre o dicionario e o texto; incrementa +1 quando encontra na chave do dicionário que
    # se refere à palavra que está sendo analisada
    for palavraTexto in texto:
        for palavraDicionario in dicionario:
            if (palavraTexto == palavraDicionario):
                bag_words[palavraDicionario] +=1

    return(bag_words)

# Procurando match de radical
def macthRadical(texto, dicionario):
    textoStemmer = [stemmer.stem(w.lower()) for w in texto] # Transforma as palavras do dicionario em radicais
    dicStermmer = [stemmer.stem(w.lower()) for w in dicionario] # Transforma as palavras do texto em radicais
    bag_words ={}
    nMatch = {}

    # Inicia o dicionario que sera usado para contabilizar os radicais e as palavras que nao tiveram match
    for i in dicStermmer:
        bag_words[i] = 0
        nMatch[i] = 0

    # Procura os radicais como feito no metodo anterior
    for palavraTexto in textoStemmer:
        for palavraDicionario in dicStermmer:
            if (palavraTexto == palavraDicionario):
                bag_words[palavraDicionario] += 1

    # Procura as palavras que nao estao presente no texto e estao no dicionario
    for palavraDicionario in dicStermmer:
        if (not palavraDicionario in textoStemmer):
            nMatch[palavraDicionario] += 1

    return(bag_words, nMatch)
    
# Relevância de cada palavra
def relevancia():
    total = matchWord(word_features, palavras)
    radical, nMatch = macthRadical(word_features, palavras)

    relevancia_total_sum = 0
    relevancia_radical_sum = 0
    relevancia_nMatch_sum = 0

    # calcula a relevancia dos 3 casos com a seguinte formula: value_matchs_found / total_matchs
    for matchPalavra, key in enumerate(total):
        relevancia_total_sum += total[key]

    for matchPalavra, key in enumerate(radical):
        relevancia_radical_sum += radical[key]
    
    for matchPalavra, key in enumerate(nMatch):
        relevancia_nMatch_sum += nMatch[key]

    relevancia_total = {}
    relevancia_radical = {}
    relevancia_nMatch = {}

    for matchPalavra, key in enumerate(total):
        if(relevancia_total_sum == 0):
            relevancia_total_sum = 1
        relevancia_total[key] = total[key] / relevancia_total_sum

    for matchPalavra, key in enumerate(radical):
        if(relevancia_radical_sum == 0):
            relevancia_radical_sum = 1
        relevancia_radical[key] = radical[key] / relevancia_radical_sum

    for matchPalavra, key in enumerate(nMatch):
        if(relevancia_nMatch_sum == 0):
             relevancia_nMatch[key] = 0
        else:
            if(relevancia_nMatch_sum == 0):
                relevancia_nMatch_sum = 1
            relevancia_nMatch[key] = nMatch[key] / relevancia_nMatch_sum

    print(relevancia_total)
    print(relevancia_radical)
    print(relevancia_nMatch)

    return(relevancia_total, relevancia_radical, relevancia_nMatch)

(total, radical, nMatch) = relevancia()
rele = 0

# a relevancia final do texto é calculada somando as relevancias de cada palavra do dicionario dividido pela quantidade
# de palavras no dicionario
for word, key in enumerate(total):
    rele += fuzzyText.fuzzyRelText(total[key], radical[stemmer.stem(key)], nMatch[stemmer.stem(key)])

print(rele/len(total))  
