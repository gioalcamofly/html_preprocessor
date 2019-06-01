import csv
import html2text
from os import listdir
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import FreqDist
import nltk
from nltk.stem import PorterStemmer
from os.path import isfile, join
import sys
import re

reload(sys)
sys.setdefaultencoding('iso-8859-15')

path = "../BankSearch-reduced/"
preprocessed_path = "../BankSearch-reduced/preprocessed_text/"
stop_words = set(stopwords.words('english'))
stop_words.add('http')
stop_words.add('html')
stop_words.add('gif')
stop_words.add('com')
stop_words.add('www')
stop_words.add('co')
stop_words.add('uk')
stop_words.add('images')

def getFiles(path):

    files = [open(join(path, f)) for f in listdir(path) if isfile(join(path, f))]
    return files

def html_to_text(files):

    raw_texts = [f.read() for f in files]

    h = html2text.HTML2Text()
    h.ignore_links = True

    texts = [re.sub(r"[^a-zA-Z]+", ' ', h.handle(text).lower()) for text in raw_texts]

    return texts

def saveTexts(texts, files):

    for file in files:
        w = open(preprocessed_path + 'out_' + file.name.replace(path, ''), 'w')
        w.write(texts.pop(0).encode("utf-8"))

def stemmer(texts):
    stemmed_text = []
    ps = PorterStemmer()

    for text in texts:
        # words = nltk.word_tokenize(text)
        stem_text = ""

        for w in text:
            stem_text = stem_text + " " + ps.stem(w)

        stemmed_text.append(stem_text)

    return stemmed_text

files = getFiles(preprocessed_path)
if len(files) > 0:
    print('entro aqui')
    texts = [f.read() for f in files]
else:
    files = getFiles(path)
    texts = html_to_text(files)

    saveTexts(texts, files)



tokenized_text = []
count = 0
for text in texts:
    tokens = nltk.word_tokenize(text)
    fdist = FreqDist(tokens)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [w for w in tokens if (fdist[w] > 3 or len(tokens) < 100)] #It's better to do this after the stopwords are eliminated, because they affect the length
    tokenized_text.append(tokens)
    if len(set(tokens)) == 0:
        count += 1
    print len(set(tokens))

print count


vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(stemmer(tokenized_text))

feature_names = vectorizer.get_feature_names()
tfidf = X.toarray()

w = open('out.txt', 'a')

# for weight in tfidf[0]:
#     print(weight)

for word in feature_names:
    print word
