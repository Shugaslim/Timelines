import wikipedia
import spacy
nlp = spacy.load('en_core_web_sm')
import time


def retrieve_Sents(sents, nO):
	finalSents = []
	for i in sents:
		res = isDate_Spacy(i, nO)
		if res[0]:
			for d in res[1]:
				result = (i, d.text)
				finalSents.append(result)
	return finalSents

def isDate_Spacy(sent, numOnly):
	doc = nlp(sent)
	ent_dates = []

	for ent in doc.ents:
		if ent.label_ == "DATE":
			ent_dates.append(ent)
	dates = []
	if numOnly:
		for i in ent_dates:
			if i.text.isdigit():
				dates.append(i)
			else:
				continue
	else:
		dates = ent_dates

	if len(dates) > 0:
		return (True, dates)
	else:
		return (False, [])



def common_tokens(input):
	page = wikipedia.page(input)
	pageC = page.content
	doc = nlp(pageC)
	ProperToken = {}
	for i in doc.ents:
		if i.text in ProperToken.keys():
			ProperToken[i.text] += 1.0
		else:
			ProperToken[i.text] = 1.0

	length = len(ProperToken.keys())
	for i in ProperToken:
		ProperToken[i] /= float(length)
	
	Sorted_PT = sorted(ProperToken.items(), key=lambda x:x[1], reverse=True)

	return Sorted_PT


def analyze(common, sents1):
	finalList = []
	for i in sents1:
		weight = 0
		year = 0
		for j in common:
			if j[0] in i[0]:
				weight += 1000*j[1]
		year = int(i[1])
		result = (i[0], weight, year)
		finalList.append(result)
	return finalList

def parse(input):
	try:
		page = wikipedia.page(title=input, pageid=None)
	except:
		return []

	pageC = page.content
	doc = nlp(pageC)
	sents = []
	for s in doc.sents:
		sents.append(s.text)
	return sents