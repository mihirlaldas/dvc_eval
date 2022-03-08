import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.svm import LinearSVC
from pickle import load, dump
import os

pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,2), stop_words='english', sublinear_tf=True)),
                     ('chi', SelectKBest(chi2, k=10000)),
                     ('clf', LinearSVC(C=1.0, penalty='l1',max_iter=3000, dual=False))
                    ])

# train_x = pd.read_csv('data/prepared/train_x.csv')
# train_y = pd.read_csv('data/prepared/train_y.csv')

# test_x = pd.read_csv('data/prepared/test_x.csv')
# test_y = pd.read_csv('data/prepared/test_y.csv')

train_x = load(open('data/prepared/train_x.pkl', 'rb'))
train_y = load(open('data/prepared/train_y.pkl', 'rb'))

test_x = load(open('data/prepared/test_x.pkl', 'rb'))
test_y = load(open('data/prepared/test_y.pkl', 'rb'))
print(train_x.head())
print(train_y.shape)

model = pipeline.fit(train_x, train_y)
print('accuracy score: '+ str(model.score(test_x, test_y)))

os.makedirs("model", exist_ok=True)
dump(model, open('model/model.pkl', 'wb'))
