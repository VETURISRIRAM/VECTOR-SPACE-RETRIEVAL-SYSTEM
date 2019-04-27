"""
@author : Sriram Veturi (UIN : 651427659)
@title  : Basic Vector Space Retrieval System
"""

import os
import re
import math
import nltk
import copy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from nltk import word_tokenize
from pathlib import Path


def initializations():

    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()
    print("Guidelines to Enter the Input for File/Directory names!\n")
    print("1) You can just enter the directory/file name (with extension) if present in the same directory as this file!")
    print("2) No need to include any slashes if in the same directory. Path module would take care of it!")
    print("3) If the required Directory/File is in same Directory, You can Enter the contents in the Brackets!\n")
    print("Now, please enter the following!\n")
    directory = input("Path to Directory containing the Data Text Files : (cranfieldDocs) : ")  # cranfieldDocs
    queriesFilePath = input("Path to 'queries.txt' file : (queries.txt) : ")  # queries.txt
    relevanceFilePath = input("Path to 'relevance.txt' file : (relevance.txt) : ")  # relevance.txt

    return stopWords, ps, directory, queriesFilePath, relevanceFilePath


# Function traverse and get the text from files
def traversal(directory):

    directory = Path(directory)
    # Dictionary to store the file with words in it
    fileWordsDict = dict()
    # To store the text of files
    contents = list()
    for file in os.listdir(directory):
        with open(os.path.join(directory, file)) as f:
            c = f.read()
            contents = contents + word_tokenize(c)
            fileName = int(file.replace('cranfield', ''))
            fileWordsDict[fileName] = word_tokenize(c)

    return contents, fileWordsDict


def removeSGMLTags(contents, fileWordsDict):

    corpus = list()
    corpusDict = dict()
    cleanContent = re.compile('<.*?>')

    for content in contents:
        if '\n' in content:
            content = content.replace('\n', '')
        cleanString = re.sub(cleanContent, '', content)
        tags = ['DOC', 'DOCNO', 'TITLE', 'AUTHOR', 'BIBLIO', 'TEXT']
        for tag in tags:
            cleanString = cleanString.replace(tag, '')
        corpus.append(cleanString)

    for file, words in fileWordsDict.items():
        temp = list()
        for word in words:
            if '\n' in word:
                word = word.replace('\n', '')
            cleanString = re.sub(cleanContent, '', word)
            tags = ['DOC', 'DOCNO', 'TITLE', 'AUTHOR', 'BIBLIO', 'TEXT']
            for tag in tags:
                cleanString = cleanString.replace(tag, '')
            temp.append(cleanString)
        corpusDict[file] = temp

    return corpus, corpusDict


# Function to preprocess the text
def preprocessText(contents, fileWordsDictionary):
    # To store the corpus of words
    corpus = list()
    # Create corpus, remove special chars and lowercase operation.
    for content in contents:
        content = re.sub('[^A-Za-z]+', '', content).lower()
        corpus.append(content)
    # Remove Stopwords before stemming
    corpus = [word for word in corpus if word not in stopWords]
    # Integrate Porter Stemmer
    corpus = [ps.stem(word) for word in corpus]
    # Remove Stopwords after stemming
    corpus = [word for word in corpus if word not in stopWords]
    # Remove unnecessary empty strings from corpus
    corpus = [word for word in corpus if word != '' and len(word) > 2]
    # Preprocess the dictionary of words
    corpusDict = dict()
    for file, words in fileWordsDictionary.items():
        preprocessedWordList = list()
        for word in words:
            word = re.sub('[^A-Za-z]+', '', word).lower()
            preprocessedWordList.append(word)
        # Remove Stopwords before stemming
        preprocessedWordList = [word for word in preprocessedWordList if word not in stopWords]
        # Integrate Porter Stemmer
        preprocessedWordList = [ps.stem(word) for word in preprocessedWordList]
        # Remove Stopwords after stemming
        preprocessedWordList = [word for word in preprocessedWordList if word not in stopWords]
        # Remove unnecessary empty strings from corpus
        preprocessedWordList = [word for word in preprocessedWordList if word != '' and len(word) > 2]
        corpusDict[file] = preprocessedWordList

    return corpus, corpusDict


def generateTF(corpus, corpusDict):
    tfDict = dict()
    for documentNumber, wordsInIt in corpusDict.items():
        tfDictEach = dict()
        for docWord in wordsInIt:
            tfDictEach[docWord] = wordsInIt.count(docWord) / len(wordsInIt)
        tfDict[documentNumber] = tfDictEach

    return tfDict


def generateIDF(corpus, corpusDict, totalDocuments=1400, totalQueries=10, task=1):

    # Compute the number of documents in which the words exist.
    idfDict = dict()
    for word in corpus:
        count = 0
        for documentNumber, wordsInIt in corpusDict.items():
            if word in wordsInIt:
                count += 1
        if count == 0:
            continue
        else:
            if task == 1:
                idfDict[word] = math.log(totalDocuments / count)
            else:
                idfDict[word] = math.log(totalQueries / count)

    return idfDict


def generateTFIDF(corpus, corpusDict, tfDict, idfDict):

    tfidfDict = dict()
    for documentNumber, wordsInIt in corpusDict.items():
        tfidfDictEach = dict()
        for word in wordsInIt:
            tfidfDictEach[word] = tfDict[documentNumber][word] * idfDict[word]
        tfidfDict[documentNumber] = tfidfDictEach

    return tfidfDict


def documentsCleaning():

    # Traverse the directory to retrieve text
    contents, fileWordsDict = traversal(directory)
    # Clean the tags and remove '\n' from the text
    contents, removedSGMLDict = removeSGMLTags(contents, fileWordsDict)
    print("Removed SGML Tags!")
    # Stop Words Removal, Perter Stemmer and other text preprocessing
    corpus, corpusDict = preprocessText(contents, removedSGMLDict)
    print("Documents Preprocessing Done!")
    # Calculate TF for each word in each document
    tfDict = generateTF(corpus, corpusDict)
    print("Documents Term Frequencies Generated!")
    # Calculate IDF for each word in the corpus
    idfDict = generateIDF(corpus, corpusDict)
    print("Documents Inverse Document Frequencies Generated")
    # Calculate TF-IDF for each word in the document
    tfidfDict = generateTFIDF(corpus, corpusDict, tfDict, idfDict)
    print("Documents TF-IDFs Generated!\n")

    return tfidfDict


def queryPreprocessing(queriesFilePath):

    queriesFilePath = Path(queriesFilePath)
    contents, queriesByLines, line = list(), list(), list()
    with open(queriesFilePath) as f:
        c = f.read()
        contents = contents + word_tokenize(c)
    for query in contents:
        if query != '.':
            line.append(query)
        else:
            queriesByLines.append(line)
            line = list()
    queryPreprocess = list()
    for contents in queriesByLines:
        # To store the corpus of words
        corpus = list()
        # Create corpus, remove special chars and lowercase operation.
        for content in contents:
            content = re.sub('[^A-Za-z]+', '', content).lower()
            corpus.append(content)
        # Remove Stopwords before stemming
        corpus = [word for word in corpus if word not in stopWords]
        # Integrate Porter Stemmer
        corpus = [ps.stem(word) for word in corpus]
        # Remove Stopwords after stemming
        corpus = [word for word in corpus if word not in stopWords]
        # Remove unnecessary empty strings from corpus
        corpus = [word for word in corpus if word != '' and len(word) > 2]
        queryPreprocess.append(corpus)

    return queryPreprocess


def queriesCleaning(queriesFilePath):

    preprocessedQueries = queryPreprocessing(queriesFilePath)
    queriesCorpus = list(set(list(sum(preprocessedQueries, []))))
    queriesCorpusDict = dict()
    count = 1
    for query in preprocessedQueries:
        queriesCorpusDict[count] = query
        count += 1
    queriesTFDict = generateTF(queriesCorpus, queriesCorpusDict)
    queriesIDFDict = generateIDF(queriesCorpus, queriesCorpusDict, task=2)
    queriesTFIDFDict = generateTFIDF(queriesCorpus, queriesCorpusDict, queriesTFDict, queriesIDFDict)

    return queriesTFIDFDict


def cosineSimilarityCalculator(documentsTFIDFDict, queriesTFIDFDict):

    queriesDocumentsSimilarity = dict()
    for query, queryWordsAndWeightDict in queriesTFIDFDict.items():
        queryDocumentsSimilarity = dict()
        for document, documentWordsAndWeightsDict in documentsTFIDFDict.items():
            intersection = [i for i in list(queryWordsAndWeightDict.keys()) if
                            i in list(documentWordsAndWeightsDict.keys())]
            if len(intersection) == 0:
                continue
            else:
                numerator = 0
                for commonWord in intersection:
                    numerator += queryWordsAndWeightDict[commonWord] * documentWordsAndWeightsDict[commonWord]
                docTerm = 0
                for docWords, wordWeights in documentWordsAndWeightsDict.items():
                    docTerm += (wordWeights) ** 2
                docTerm = (docTerm) ** (1 / 2)
                queryTerm = 0
                for queryWord, wordWeights in queryWordsAndWeightDict.items():
                    queryTerm += (wordWeights) ** 2
                queryTerm = (queryTerm) ** (1 / 2)
                denominator = docTerm * queryTerm
                cosineSimilarity = numerator / denominator
            index = '(' + str(query) + ',' + str(document) + ')'
            queryDocumentsSimilarity[index] = cosineSimilarity

        # Sort the similarities in descending order
        queryDocumentsSimilarity = sorted(queryDocumentsSimilarity.items(), key=lambda kv: kv[1])[::-1]
        temp = dict()
        for x in queryDocumentsSimilarity:
            temp[x[0]] = x[1]
        if len(temp) != 0:
            queriesDocumentsSimilarity[query] = temp

    return queriesDocumentsSimilarity


def readRelevance(filePath):

    filePath = Path(filePath)
    relevanceList = dict()
    contents = list()
    with open(filePath) as f:
        c = f.read()
        contents = contents + word_tokenize(c)
    separate, temp = list(), list()
    each = 0
    for index in range(0, len(contents), 2):
        if contents[index] == str(each):
            temp.append(contents[index])
            temp.append(contents[index+1])
        else:
            separate.append(temp)
            each += 1
            temp = list()
            temp.append(contents[index])
            temp.append(contents[index + 1])
    separate = separate[1:]
    tenth = list()
    for x in contents[len(list(sum(separate, []))) :]:
        tenth.append(x)
    separate.append(tenth)
    count = 1
    for each in separate:
        temp = list()
        for c in range(0, len(each), 2):
            temp.append(str('('+str(each[c]+','+str(each[c+1]+')'))))
        relevanceList[count] = temp
        count += 1

    return relevanceList

def calculatePrecisionRecall(relevantDocuments, topRetrieved, top):

    # Precision-Recall Score
    precisionsList, recallsList = list(), list()

    for i in range(1, 11):
        docList = topRetrieved[i]
        relList = relevantDocuments[i]
        numberRelevant = len(relList)
        numberRetrieved = len(docList)
        numberRelevantRetrieved = len([i for i in docList if i in relList])
        recallTop = numberRelevantRetrieved / numberRelevant
        precisionTop = numberRelevantRetrieved / numberRetrieved
        precisionsList.append(precisionTop)
        recallsList.append(recallTop)
        averagePrecisionTop = sum(precisionsList) / 10
        averageRecallTop = sum(recallsList) / 10
    print("\nAverage Scores for Top-{0} documents :\n1) Recall : {1}\n2) Precision : {2}".format(top,
                                                                                                    averagePrecisionTop,
                                                                                                    averageRecallTop))
    print("\nRecall Score for all the queries    : ", recallsList)
    print("\nPrecision Score for all the queries : ", precisionsList)

def standardizeForPrecisionRecall(relevantDocuments, retrievedDocuments):

    retrievedDocs = dict()
    for documentNumber, retrieved in retrievedDocuments.items():
        retrievedDocs[documentNumber] = list(retrieved.keys())
    topTenRetrieved, topFiftyRetrieved, topHundredRetrieved, topFiveHundredRetrieved = {}, {}, {}, {}
    for documentNumber, retrieved in retrievedDocs.items():
        if len(retrieved) >= 10:
            topTenRetrieved[documentNumber] = retrieved[:10]
        else:
            topTenRetrieved[documentNumber] = retrieved
        if len(retrieved) >= 50:
            topFiftyRetrieved[documentNumber] = retrieved[:50]
        else:
            topFiftyRetrieved[documentNumber] = retrieved
        if len(retrieved) >= 100:
            topHundredRetrieved[documentNumber] = retrieved[:100]
        else:
            topHundredRetrieved[documentNumber] = retrieved
        if len(retrieved) >= 500:
            topFiveHundredRetrieved[documentNumber] = retrieved[:500]
        else:
            topFiveHundredRetrieved[documentNumber] = retrieved

    calculatePrecisionRecall(relevantDocuments, topTenRetrieved, top=10)
    calculatePrecisionRecall(relevantDocuments, topFiftyRetrieved, top=50)
    calculatePrecisionRecall(relevantDocuments, topHundredRetrieved, top=100)
    calculatePrecisionRecall(relevantDocuments, topFiveHundredRetrieved, top=500)


if __name__ == "__main__":

    stopWords, ps, directory, queriesFilePath, relevanceFilePath = initializations()
    print("###################### Execution Starts ########################\n")
    documentsTFIDFDict = documentsCleaning()
    print("################### Documents Cleaning Done ####################\n")
    queriesTFIDFDict = queriesCleaning(queriesFilePath)
    print("#################### Queries Cleaning Done #####################\n")
    queriesDocumentsSimilarity = cosineSimilarityCalculator(documentsTFIDFDict, queriesTFIDFDict)
    print("################### Cosine Similarity Found ####################\n")
    relevanceList = readRelevance(relevanceFilePath)
    print("##################### Relevance File Read ######################\n")
    print('############################ RESULTS ###########################')
    standardizeForPrecisionRecall(relevanceList, queriesDocumentsSimilarity)
