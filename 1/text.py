# Natural Language Toolkit
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
# libary for ru text
from razdel import tokenize, sentenize

# create tokens from text
data = "NLTK is undergoing continual development as new modules are added and existing ones are improved. Since the book was published in 2009, we upgraded NLTK to Python 3 and updated all the code samples in the online version of the book."
tokens = word_tokenize(data)
print(tokens)

text = "Для определения интерфейса для работы с сервером нередко используются html-страницы, то есть cтатические файлы с кодом html, которые могут использовать какие-то другие статические файлы - файлы стилей css, изображений, скриптов javascript и т.д."
lst = list(tokenize(text))
print(lst)

# N-grams(unigrams, bigrams, trigrams)

unigram = list(nltk.ngrams(tokens, 1))
bigrams = list(nltk.ngrams(tokens, 2))
print(unigram[:5])
print(bigrams[:5])

# find most commons n-grams
from nltk import FreqDist

print(FreqDist(unigram).most_common(5))
print(FreqDist(bigrams).most_common(5))

# stop words we can delete sometimes
nltk.download('stopwords')
from nltk.corpus import stopwords

stopwords = set(stopwords.words('russian'))
print(stopwords)

# стоп слова лучше не убирать при оценке тональности

# пунктуация, которая может быть токеном
import string

print(string.punctuation)

# стемминг отбрасывает часть слов, а лематизация находит корень

from nltk.stem import PorterStemmer, SnowballStemmer

words = ["game", "gaming", "gamer", "computer"]
words_ru = ['корова', 'мальчики', 'мужчины', 'столом', 'убежала']

ps = PorterStemmer()
print(list(map(ps.stem, words)))

ss = SnowballStemmer(language='russian')
print(list(map(ss.stem, words_ru)))

# pymorhy - lemmatisation

raw = """DENNIS: Listen, strange women lying in ponds distributing swords
is no basis for a system of government.  Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""

raw_ru = """Не существует научных доказательств в пользу эффективности НЛП, оно 
признано псевдонаукой. Систематические обзоры указывают, что НЛП основано на 
устаревших представлениях об устройстве мозга, несовместимо с современной 
неврологией и содержит ряд фактических ошибок."""

import pymorphy2

morph = pymorphy2.MorphAnalyzer()
pymorphy_results = list(map(lambda x: morph.parse(x), raw_ru.split(' ')))
print(pymorphy_results)

# spacy english - context

# PyMystem3 - another lib for russian text
# for all tasks use different methods
