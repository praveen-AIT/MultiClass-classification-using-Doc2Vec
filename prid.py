#import all modules
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from gensim.models import doc2vec
from collections import namedtuple

def get_tag_and_training_data(filename):
    '''takes the input file and returns  tokenized sentences and document tags as separate lists'''
    tags=[]
    documents=[]
    line_counter=1
    with open(filename) as f:
        for line in f:
            #skip first line
            if line_counter==1:
                line_counter=line_counter+1
                continue
            #Initialize the token list for line
            tags.append(int(line[:1]))
            documents.append(line[2:])
    return tags,documents
    
Y,X=get_tag_and_training_data('training_data.txt')

#75:25 training test split
Y_train,Y_test=Y[:2200],Y[2200:]

# data already loaded as lists of sentences in X and Y

docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(X):
    words = text.lower().split()
    tags = [i]
    docs.append(analyzedDocument(words, tags))

# Train model (set min_count = 1, if you want the model to work with the provided example data set)
model = doc2vec.Doc2Vec(docs, size = 160, window = 10, min_count = 7, workers = 4)

#making training and test sets
wb_Y_train,wb_Y_test=Y_train,Y_test
wb_X=[]
for i in range(len(X)):
    wb_X.append(model.docvecs[i])
wb_X_train=wb_X[:2200]
wb_X_test=wb_X[2200:]

# Word Embeddings Logistic Regression

wb_logreg = linear_model.LogisticRegression(C=1e4)
wb_logreg.fit(wb_X_train,wb_Y_train)
wb_pred=wb_logreg.predict(wb_X_test)
print(accuracy_score(wb_Y_test, wb_pred))

# Word Embeddings Naive Bayes

wb_clf = GaussianNB()
wb_clf.fit(wb_X_train,wb_Y_train)
wb_nb_pred=wb_clf.predict(wb_X_test)
print(accuracy_score(wb_Y_test, wb_nb_pred))
