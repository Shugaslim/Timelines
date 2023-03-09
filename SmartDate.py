
import pandas as pd
from sklearn.cluster import Birch
import numpy as np
from bs4 import BeautifulSoup
import spacy
from EventCollect import retrieve_Sents, parse
from sklearn.ensemble import GradientBoostingRegressor


class SmartDate:
    def __init__(self):
        #Load training data
        #Train model
        self.nlp = spacy.load('en_core_web_sm')
        self.model = GradientBoostingRegressor(random_state=0)


    def getDate(self, phrase):
        self.compute()

    
    def loadData(self):
        #load Datacsv
        #return training and testing data
        data = pd.read_csv("Data.csv")
        freq_d = {}
        w2vec = {}
        w2id = {}
        Vecs = []
        stop = [ 'stop', 'the', 'to', 'and', 'a', 'in', 'it', 'is', 'I', 'that', 'had', 'on', 'for', 'were', 'was']

        X = data.X.to_list()
        y = data.y.to_list()

        y = np.nan_to_num(y)
        y = y.astype('int32')
        n = len(X)
        index = 0
        for i in X[0:1000]:
            print(str(index) + " / " + str(n))
            doc = self.nlp(i)
            # for j in doc:
            #     if j.text in stop:
            #         continue
            #     else:
            #         if j.text in freq_d.keys():
            #             freq_d[j.text] += 1
            #         else:
            #             freq_d[j.text] = 1.0
            w2vec[i] = doc.vector
            index+=1
        
        # for i in freq_d.keys():
        #     freq_d[i] /= len(freq_d.keys())

        # vocab = freq_d.keys()
        # vocab.sort()

        # id = 1
        # for i in vocab:
        #     if i in w2id.keys():
        #         continue
        #     else:
        #         w2id[i] = id
        #         id+=1

        for i in X:
            if i in w2vec.keys():
                Vecs.append(w2vec[i])

        return np.array(Vecs), y
    
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
    
    def compute(self):
       print("Loading Data")
       trainInput, TrainOutput = self.loadData()

       X = trainInput[0:90]
       y = TrainOutput[0:90]
       test_X = trainInput[91:96]
       test_y = TrainOutput[91:96]

       print("Training data") 
       self.model.fit(X, y)

       print("Making Predictions")
       pred_y = self.model.predict(test_X)

       print("Actual:")
       print(test_y)
       print("prediction:")
       print(pred_y)


sm = SmartDate()
sm.compute()
