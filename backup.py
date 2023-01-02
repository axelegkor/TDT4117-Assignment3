import gensim
import random
import codecs
import string
from nltk.stem.porter import PorterStemmer
from nltk import FreqDist

# 1.0
random.seed(123)

# 1.1
f = codecs.open('pg3300.txt', 'r', 'utf-8')

# List of paragraphs (1.2)
# Read from file
textList = f.read().split('\n\n')

# Remove empty paragraphs
textList = list(filter(None, textList))

# Filter out "Gutenberg" paragraphs (1.3)
tempList = []
for paragraph in textList:
    if 'Gutenberg' not in paragraph:
        tempList.append(paragraph)
textList = tempList

# Tokenize paragraphs (1.4)
tempList = []
for paragraph in textList:
    tempList.append(paragraph.split())
textList = tempList

# Remove text punctuation and convert to lowercase (1.5)
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
textList = tempList

# Stem words (1.6)
stemmer = PorterStemmer()
tempList = []
for paragraph in textList:
    tempParagraph = []
    for word in paragraph:
        tempParagraph.append(stemmer.stem(word))
    tempList.append(tempParagraph)
textList = tempList

freqDist = FreqDist()
# Count word frequencies (1.7)
for paragraph in textList:
    for word in paragraph:
        freqDist[word] += 1


# Build a dictionary (2.0)

dictionary = gensim.corpora.Dictionary(textList)


# Filter out stopwords (2.1)
g = codecs.open('common-english-words.txt', 'r', 'utf-8')
stopwords = g.read().split(",")

stopword_ids = []
for stopword in stopwords:
    try:
        stopword_id = dictionary.token2id[stopword]
        stopword_ids.append(stopword_id)
    except KeyError:
        pass

dictionary.filter_tokens(stopword_ids)

# Map paragraphs into bag-of-words (2.2)
corpus = [dictionary.doc2bow(paragraph) for paragraph in textList]

# Build a TF-IDF model (3.1)
tfidf = gensim.models.TfidfModel(corpus)

# Map Bags-of-words into TF-IDF weights (3.2)
corpus_tfidf = tfidf[corpus]

# Construct MatrixSimilarity object (3.3)
matrixSimilarity_tfidf = gensim.similarities.MatrixSimilarity(corpus_tfidf)

# Repeat for LSI (3.4)
lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)

corpus_lsi = lsi[corpus_tfidf]

matrixSimilarity_lsi = gensim.similarities.MatrixSimilarity(corpus_lsi)

# Report and try to interpret first 3 LSI topics (3.5)
print(lsi.show_topics(3))
