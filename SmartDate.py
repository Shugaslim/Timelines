
import pandas as pd
import numpy as np
from numpy.linalg import norm
from bs4 import BeautifulSoup
import spacy
from EventCollect import retrieve_Sents, parse
from numba import jit, cuda
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
import time
# import plotly.express as plt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class SmartDate:
    def __init__(self):
        #Load training data
        #Train model
        self.nlp = spacy.load('en_core_web_md')
        self.model = GradientBoostingRegressor(n_estimators=20,max_depth=8,random_state=0, max_features="sqrt", learning_rate=0.1)
        self.discrim_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=0)
        self.km = KMeans(n_clusters=3, random_state=0)
        self.vec_dict = {}


    def getDate(self, phrase):
        self.compute()
    
    def loadData(self):
        #load Datacsv
        #return training and testing data
        data = pd.read_csv("Data.csv")
        w2vec = {}
        Vecs = []
        new_y = []

        data_sup = data[~data['y'].isna()]
        data_sup.y.astype('int32')
        data_unsup = data[data['y'].isna()]

        # X = data.X.to_list()
        # y = data.y.to_list()

        # y = np.nan_to_num(y)
        # y = y.astype('int32')

        #loading supervised data
        for index, row in data_sup.iterrows():
            doc = self.nlp(row.X)
            if row.X in w2vec.keys():
                continue
            else:
                w2vec[row.X] = (doc.vector.astype('float64'), row.y)
                Vecs.append(doc.vector.astype('float64'))
                new_y.append(row.y)

        
        #loading unspervised data 
        for index, row in data_unsup.iterrows():
            #  if index > 10000:
            #      break
             doc = self.nlp(row.X)
             if row.X in w2vec.keys():
                continue
             else:
                w2vec[row.X] = (doc.vector.astype('float64'))
                Vecs.append(doc.vector.astype('float64'))
        
        self.vec_dict = w2vec

        return np.array(Vecs), np.array(new_y)
    
    def buildDataset(self):
        HTMLFile = open("ListofHisstory.html", "r")
        index = HTMLFile.read()
        S = BeautifulSoup(index, features="html.parser")
        pages = []
        data = []
        for i in S.find_all('li'):
            link = i.text
            link = link.lower()
            if "history" in link:
                if ',' in link:
                    val = link.split(",")
                    new_s = "history of "+ val[0]
                    pages.append(new_s)
                else:
                    pages.append(link)

            for i in pages:
                print(i)
                sents = parse(i)
                finsents = retrieve_Sents(sents, False)
                for tup in finsents:
                    data.append(tup[1])

            df = pd.DataFrame(data=data, columns=["X"])
            df.to_csv("Data.csv")
    
    def getWords(self, x):
        words = []
        for i in x:
            ind = -1
            vals = list(self.vec_dict.values())
            for j in range(0, len(vals)):
                if (i == vals[j][0]).any():
                    ind = j
            val = list(self.vec_dict.keys())[ind]
            words.append(val)
        return words
    
    def bestPred(self, d_X, d_us_X, d_y):
        #Make variables
        fin_X = []
        threshhold = 100
        
        for i in d_us_X:
            i_dub = np.array(i, dtype=np.double)
            lab = self.km.predict([i_dub])
            if lab[0] == 2:
                phrase = self.getWords([i_dub])[0]
                pred = self.model.predict([i_dub])
                print(phrase + "\t" + str(pred))
                fin_X.append(i)

        return np.array(fin_X)

    def noDup(self, us_test_X, X):
        batch_size = 100
        rng = np.random.default_rng()
        rand_ind =rng.choice(len(us_test_X), batch_size, replace=False)
        us_X = []
        for j in rand_ind:
            if us_test_X[j] in X:
                continue
            else:
                us_X.append(us_test_X[j])
        return us_X
    
    def easyPop(self, us_X):
        new_X, new_y = [], []
        for i in us_X:
            label = self.km.predict([i])
            if label == 2:
                word = self.getWords([i])[0]
                if word.isnumeric():
                    new_X.append(i)
                    new_y.append(int(word))
        
        return np.array(new_X), np.array(new_y)


        


    def compute(self):
        print("Loading Data")
        trainInput, TrainOutput = self.loadData()
        print("Unique data points: " + str(len(trainInput)))
        n = len(TrainOutput)
        p = int((n * 9) / 10)
        X = trainInput[:p]
        y = TrainOutput[:p]
        test_X = trainInput[p:n-1]
        test_y = TrainOutput[p:n-1]
        us_test_X = trainInput[n:]
        iter = 1
        scores = []
        mse_points = []

        #train Discrimanate model
        self.km.fit(X)
        X_s, y_s = self.easyPop(us_test_X)
        if len(X_s) > 0:
                X = np.concatenate((X, X_s))
                y = np.concatenate((y, y_s.astype('int32')))

        
        while True:
            if iter == 2:
                break
            print("iter: " + str(iter))
            shape = X.shape
            print("Training data size: "+ str(shape))
            self.model.fit(X, y)
            score = self.model.score(test_X, test_y)
            print("score is: "+ str(score))
            #scores.append(score)


            # new_us_X = []
            # while len(new_us_X) == 0:
            #     new_us_X = self.noDup(us_test_X, X)

            # new_test_X =  self.bestPred(X, new_us_X, y)
            # # words = self.getWords(new_test_X, vec_dict)
            # if len(new_test_X) > 0:
            #     new_pred_y = self.model.predict(new_test_X).astype('int32')
            #     X = np.concatenate((X, new_test_X))
            #     y = np.concatenate((y, new_pred_y))
            
            # s_pred_y = self.model.predict(test_X)
            # mse = mean_squared_error(test_y, s_pred_y)
            # mse_points.append(mse)
            iter+=1
        
        #plt.plot(scores)
        #plt.show()
    


sm = SmartDate()
sm.compute()
