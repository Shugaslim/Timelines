
import pandas as pd
import sklearn
import html
from bs4 import BeautifulSoup
from EventCollect import retrieve_Sents, parse


class SmartDate:
    def init(self):
        #Load training data
        #Train model
        return ''   

    def getDate(self, phrase):
        date = -1
        return date
    
    def loadData(self):
        #load Datacsv
        #return training and testing data
        return [], []
    
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
       trainData, testData = self.loadData()
       #train model



