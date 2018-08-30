import wikipedia
import nltk
import re
import operator

def retrieve_Sents(sents):
	finalSents = []
	for i in sents:
		tokens = nltk.word_tokenize(i)
		for w in tokens:
			w.lower()
		res = isDate(tokens)
		if res[0]:
			result = (i, res[1])
			finalSents.append(result)
	return finalSents

def isDate(sent):
	for i in range(1, len(sent)):
		if sent[i].isdigit() and len(sent[i]) <= 4:
			if sent[i-1] == 'on' or sent[i-1] == 'in' or sent[i-1] == 'until':
				if len(sent[i]) > 3:
					if sent[i+1] == "ce" or sent[i+1] =="ad" or sent[i+1] == "bc" or sent[i+1] == "bce":
						return (True, sent[i], sent[i+1])
					else:
						return (True, sent[i], "")
 				if len(sent[i]) <= 3:
					if sent[i+1] == "ce" or sent[i+1] =="ad" or sent[i+1] == "bc" or sent[i+1] == "bce":
						return (True, sent[i], sent[i+1])
					else:
						return (True, sent[i], "")
	return (False, sent[i])



def common_tokens(token):
	pos_tok = nltk.pos_tag(token)
	ProperToken = []
	for i in pos_tok:
		if i[1] == 'NNP' or i[1] == 'NNPS':
			ProperToken.append(i[0])

	legth = len(ProperToken)
	freq = nltk.FreqDist(ProperToken)
	most_comm_50 = freq.most_common(50)

	common_token = []
	for i in most_comm_50:
		 common_token1 = (i[0], (float(i[1]) /float(legth)))
		 common_token.append(common_token1)

	return common_token


def analyze(common, sents1):
	finalList = []
	for i in sents1:
		weight = 0
		year = 0
		tokens = nltk.word_tokenize(i[0])
		for j in tokens:
			for k in common:
				if (j == k[0]):
					weight += 1000*float(k[1])
					year = int(i[1])

		result = (i[0], weight, year)
		finalList.append(result)
	return finalList

def parse(input):
	page = wikipedia.page(input)
	pageC = page.content
	pageFinal = re.sub('=+', "", pageC) 
	sents = nltk.sent_tokenize(pageFinal)
	return sents

def tokenize(input):
	page = wikipedia.page(input)
	pageC = page.content
	pageFinal = re.sub('=+', "", pageC) 
	sents = nltk.sent_tokenize(pageFinal)
	token = nltk.word_tokenize(pageFinal)
	return token








