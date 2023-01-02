from multiprocessing.spawn import prepare
from typing import Dict
import gensim
import random
import codecs
import string
from nltk.stem.porter import PorterStemmer
from nltk import FreqDist

# 1.0
random.seed(123)


def openAndLoadFile(filename):
    # 1.1
    f = codecs.open(filename, 'r', 'utf-8')

    # Read from file
    return f.read()


def preProcessing(text, wordsToIgnore):

    # List of paragraphs (1.2)
    def splitTextIntoParagraphs(text):
        textList = text.split('\n\n')
        # Remove empty paragraphs
        return list(filter(None, textList))

    # Filter out paragraphs containing a word in wordsToIgnore (1.3)
    def removeParagraphsToIgnore(textList, wordsToIgnore):
        tempList = []
        for paragraph in textList:
            ignore = False
            for word in wordsToIgnore:
                if word in paragraph:
                    ignore = True
            if not ignore:
                tempList.append(paragraph)
        return tempList

    # Tokenize paragraphs (1.4)
    def tokenizeParagraphs(textList):
        tempList = []
        for paragraph in textList:
            tempList.append(paragraph.split())
        return tempList

    # Remove text punctuation and convert to lowercase (1.5)
    def removePunctuationAndLowercase(textList):
        tempList = []
        for paragraph in textList:
            tempParagraph = []
            for word in paragraph:
                # Remove punctuation and whitecharacters
                word = word.strip(string.punctuation + "\n\r\t")

                # Convert to lowercase
                word = word.lower()
                tempParagraph.append(word)
            tempList.append(tempParagraph)
        return tempList

    # Stem words (1.6)
    def stemWords(textList):
        stemmer = PorterStemmer()
        tempList = []
        for paragraph in textList:
            tempParagraph = []
            for word in paragraph:
                tempParagraph.append(stemmer.stem(word))
            tempList.append(tempParagraph)
        return tempList

    return stemWords(removePunctuationAndLowercase(tokenizeParagraphs(removeParagraphsToIgnore(splitTextIntoParagraphs(text), wordsToIgnore))))

# Count word frequencies (1.7)


def countWordFrequencies(textList):
    freqDist = FreqDist()
    for paragraph in textList:
        for word in paragraph:
            freqDist[word] += 1
    return freqDist


# Build a dictionary (2.0)
def Dictionary(textList, stopWordsText):

    # Build a dictionary (2.0)
    dictionary = gensim.corpora.Dictionary(textList)

    # Filter out stopwords (2.1)
    stopwords = stopWordsText.split(",")

    stopword_ids = []
    for stopword in stopwords:
        try:
            stopword_id = dictionary.token2id[stopword]
            stopword_ids.append(stopword_id)
        except KeyError:
            pass

    dictionary.filter_tokens(stopword_ids)

    # Map paragraphs into bag-of-words (2.2)
    corpus = []
    for paragraph in textList:
        corpus.append(dictionary.doc2bow(paragraph))

    return corpus, dictionary


def TFIDF(corpus):

    # Build a TF-IDF model (3.1)
    def buildTFIDFModel(corpus):
        return gensim.models.TfidfModel(corpus)

    # Map Bags-of-words into TF-IDF weights (3.2)
    def mapIntoTFIDF(tfidfModel, corpus):
        return tfidfModel[corpus]

    # Construct MatrixSimilarity object (3.3)
    def constructMatrixSimilarity(tfidfCorpus):
        return gensim.similarities.MatrixSimilarity(tfidfCorpus)

    return mapIntoTFIDF(buildTFIDFModel(corpus), corpus), constructMatrixSimilarity(mapIntoTFIDF(buildTFIDFModel(corpus), corpus))


# Repeat for LSI (3.4)
def LSI(corpus, dictionary, num_topics):
    def buildLSIModel(corpus, dictionary, num_topics):
        return gensim.models.LsiModel(TFIDF(corpus)[0], id2word=dictionary, num_topics=num_topics)

    def mapIntoLSI(lsiModel, corpus):
        return lsiModel[corpus]

    def constructMatrixSimilarity(lsiCorpus):
        return gensim.similarities.MatrixSimilarity(lsiCorpus)

    return buildLSIModel(corpus, dictionary, num_topics), constructMatrixSimilarity(mapIntoLSI(buildLSIModel(corpus, dictionary, num_topics), corpus))


# 3.5
textList = preProcessing(openAndLoadFile("pg3300.txt"), ["Gutenberg"])
stopWordsText = openAndLoadFile("common-english-words.txt")
corpus = Dictionary(textList, stopWordsText)[0]
dictionary = Dictionary(textList, stopWordsText)[1]
num_topics = 100


def First3LSITopics():
    return LSI(corpus, dictionary, num_topics)[0].show_topics(3)


# 4.1
query = preProcessing("What is the function of money?", [])
query = Dictionary(query, "")[1].doc2bow(query[0])

# 4.2 Convert BOW into TF-IDF representation


def convertBOWToTFIDF(query):
    return TFIDF(corpus)[0][query]

# 4.3 Report the top 3 most similar paragraphs
# Stuck here, and ran out of time. Got error codes when using the gensim functions, and couldn't figure out how to fix them.
