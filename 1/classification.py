# TODO shape, %%time

import nltk
from sklearn.datasets import fetch_20newsgroups

nltk.download('punkt')

newsgroups_train = fetch_20newsgroups(subset='train')

print("Весь датасет:", newsgroups_train.target_names, "\n", newsgroups_train.filenames.shape)

# рассмотрим следующую выборку

categories = ['alt.atheism', 'comp.graphics', 'misc.forsale', 'rec.autos', 'sci.crypt']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
print(newsgroups_train.filenames.shape)

# Пример документа
# print(newsgroups_train.data[0])


# Векторизация с помощью TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

# Перебор параметров

# lowercase
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors.shape

# Пример вектора - очень много нулей
# print(vectorizer.get_feature_names_out()[:10])

# min_df, max_df, ngram_range
vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=0.03, max_df=0.9)
vectorizer.fit_transform(newsgroups_train.data)
print(vectors.shape)

# stopwords, preprocessing(preproc)
from nltk.corpus import stopwords
from nltk import word_tokenize

nltk.download('stopwords')

stopWords = set(stopwords.words('english'))
nltk.download('wordnet')
wnl = nltk.WordNetLemmatizer()


def preproc_nltk_wnl(text):
    return ' '.join([wnl.lemmatize(word) for word in word_tokenize(text.lower()) if word not in stopWords])


st = "Oh, I think I ve landed Where there are miracles at work,  For the thirst and for the hunger Come the conference of birds"
print(preproc_nltk_wnl(st))

vectorizer = TfidfVectorizer(preprocessor=preproc_nltk_wnl)
vectors = vectorizer.fit_transform(newsgroups_train.data)

# preproc_spacy
import spacy

nlp = spacy.load("en_core_web_sm")
texts = newsgroups_train.data.copy()


def preproc_nltk_spacy(text):
    spacy_result = nlp(text)
    return ' '.join([token.lemma_ for token in spacy_result if token.lemma_ not in stopWords])


print(preproc_nltk_spacy(st))

new_texts = []
for doc in nlp.pipe(texts, batch_size=32, n_process=3, disable=["parser", "ner"]):
    new_texts.append(' '.join([tok.lemma_ for tok in doc if tok.lemma not in stopWords]))
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(new_texts)

# WordNetLemmatizer is faster than spacy but

# Ended model

vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.5, max_features=1000)
vectors = vectorizer.fit_transform(new_texts)
vectorizer.get_feature_names_out()[::100]

# косинусное сходство
vector = vectors.todense()[0]
(vector != 0).sum()
np.mean(list(map(lambda x: (x != 0).sum(), vectors.todense())))